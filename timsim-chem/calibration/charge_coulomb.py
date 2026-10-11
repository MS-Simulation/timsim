"""Fit and validate the site-specific charge model's Coulomb term (and p(His)) on real DIA-NN runs.

Usage:
    python charge_coulomb.py FIT_REPORT MZ_LO MZ_HI [CHECK_REPORT CHECK_LO CHECK_HI] [--fixture OUT.tsv]

FIT_REPORT is a DIA-NN report of a REAL tryptic run; MZ_LO/MZ_HI bound the precursor m/z it could
observe (its DIA window range). CHECK_REPORT is an independent run used only for validation.

What is compared, and what is not
---------------------------------
* Peptides: unmodified only (carbamidomethyl C allowed — the mass below assumes it), so the m/z
  censoring uses the right mass and no modform pooling happens. Charges 2-4 on both sides (DIA-NN's
  usual precursor charge range).
* Model: exactly `ChargeModel::distribution_with_loss` — Poisson-binomial over N-term/R/K/H, the
  penalty exp(-coulomb * (z(z-1)/2 - 1) / length) on z >= 3, renormalised over z >= 1, cap 4 —
  then restricted to charges 2-4 whose m/z the run could observe and renormalised over those.
* Real: Precursor.Quantity per charge, per peptide.
* Two metrics, per (K+R count 1-3, His count 0 / 1 / >=2) cell:
  - dominant: distribution of each peptide's most intense charge (the fit objective; cell-balanced);
  - shares: mean absolute error between the model's and the observed charge-share vectors. The
    observed vector is censored by identification (an undetected minor state reads as 0), so this
    is a consistency check, not an unbiased fraction estimate.
* Validation: the fit uses half of FIT_REPORT's peptides (hash split); every number printed for
  "held-out" uses the other half, and CHECK_REPORT is never fitted.

Agreement is printed twice and labelled: weighted like the run (every held-out peptide counts once)
and cell-balanced (every cell counts equally).
"""
import hashlib
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AA = dict(G=57.02146, A=71.03711, S=87.03203, P=97.05276, V=99.06841, T=101.04768, C=160.03065,
          L=113.08406, I=113.08406, N=114.04293, D=115.02694, Q=128.05858, K=128.09496, E=129.04259,
          M=131.04049, H=137.05891, F=147.06841, R=156.10111, Y=163.06333, W=186.07931)
SITE = dict(n=0.93, R=0.97, K=0.95)
ZS = (2, 3, 4)


def model_shares(seq, p_h, coulomb, lo, hi, max_charge=4):
    """Observable share vector over charges 2..4 (zeros where unobservable); None if none observable."""
    probs = [SITE["n"]] + [SITE[a] if a in "RK" else p_h for a in seq if a in "RKH"]
    pmf = np.zeros(len(probs) + 1)
    pmf[0] = 1.0
    for p in probs:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= 1 - p
    if coulomb > 0:
        for z in range(3, len(pmf)):
            pmf[z] *= np.exp(-coulomb * (z * (z - 1) / 2 - 1) / len(seq))
    pmf = pmf / pmf[1:].sum()
    mass = sum(AA[a] for a in seq) + 18.01056
    v = np.array([pmf[z] if z < len(pmf) and z <= max_charge and lo <= (mass + z * 1.007276) / z <= hi
                  else 0.0 for z in ZS])
    return v / v.sum() if v.sum() > 0 else None


def load(report):
    r = pq.read_table(report, columns=["Modified.Sequence", "Stripped.Sequence", "Precursor.Charge",
                                       "Q.Value", "Precursor.Quantity"]).to_pandas()
    r = r[(r["Q.Value"] <= 0.01) & r["Precursor.Charge"].isin(ZS)]
    r = r[r["Modified.Sequence"].str.replace("(UniMod:4)", "", regex=False) == r["Stripped.Sequence"]]
    r = r[r["Stripped.Sequence"].map(lambda s: set(s) <= set(AA))]
    q = r.pivot_table(index="Stripped.Sequence", columns="Precursor.Charge", values="Precursor.Quantity",
                      aggfunc="sum", fill_value=0.0).reindex(columns=list(ZS), fill_value=0.0)
    q = q[q.sum(axis=1) > 0]
    d = pd.DataFrame({"seq": q.index.to_numpy(), "kr": q.index.str.count("[KR]").to_numpy(),
                      "h": np.minimum(q.index.str.count("H").to_numpy(), 2)})
    d["shares"] = list(q.div(q.sum(axis=1), axis=0).to_numpy())
    d["dominant"] = [ZS[i] for i in q.to_numpy().argmax(axis=1)]
    d["half"] = d.seq.map(lambda s: hashlib.sha256(s.encode()).digest()[0] & 1)
    return d[d.kr.between(1, 3)].reset_index(drop=True)


def evaluate(d, p_h, coulomb, lo, hi):
    rows = []
    for s, kr, h, sh, dom in d[["seq", "kr", "h", "shares", "dominant"]].itertuples(index=False):
        m = model_shares(s, p_h, coulomb, lo, hi)
        if m is not None:
            rows.append((kr, h, ZS[int(m.argmax())], dom, float(np.abs(m - sh).sum())))
    e = pd.DataFrame(rows, columns=["kr", "h", "model", "real", "share_err"])
    loss, cell_agree, cells = 0.0, [], {}
    for key, c in e.groupby(["kr", "h"]):
        if len(c) < 50:
            continue
        real = c.real.value_counts(normalize=True).reindex(ZS, fill_value=0)
        mod = c.model.value_counts(normalize=True).reindex(ZS, fill_value=0)
        loss += float(((real - mod) ** 2).sum())
        cell_agree.append(float((c.model == c.real).mean()))
        cells[key] = (len(c), real.round(2).tolist(), mod.round(2).tolist(), round(c.share_err.mean(), 3))
    return dict(loss=loss, agree_run=float((e.model == e.real).mean()),
                agree_cells=float(np.mean(cell_agree)), share_err=float(e.share_err.mean()), cells=cells)


def sample(d, per_cell, seed=0):
    return pd.concat([c.sample(min(len(c), per_cell), random_state=seed)
                      for _, c in d.groupby(["kr", "h"])])


def main(argv):
    fixture = None
    if "--fixture" in argv:
        i = argv.index("--fixture")
        fixture = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    fit = load(argv[0])
    lo, hi = float(argv[1]), float(argv[2])
    train, test = fit[fit.half == 0], fit[fit.half == 1]
    # p(His) is held below p(K) = 0.95: gas-phase basicity R > K > H is the model's premise, and an
    # unconstrained fit lands on p(His) = p(K) (0.95) — within the fit's noise of 0.90 and chemically wrong.
    grid = [(p_h, c) for p_h in (0.7, 0.8, 0.85, 0.9) for c in (0, 8, 10, 12, 14, 16, 18, 22)]
    tr = sample(train, 2000)
    scored = sorted((evaluate(tr, p_h, c, lo, hi)["loss"], p_h, c) for p_h, c in grid)
    print("fit (train half, cell-balanced dominant-charge loss), best 4:")
    for loss, p_h, c in scored[:4]:
        print(f"  loss {loss:.3f}  histidine {p_h}  coulomb {c}")
    _, p_h, c = scored[0]
    checks = [("held-out half", test, lo, hi)]
    if len(argv) >= 6:
        checks.append(("independent run", load(argv[3]), float(argv[4]), float(argv[5])))
    for name, d, l, h in checks:
        for label, ph, cc in (("independent sites (pre-0.3)", 0.80, 0.0), (f"fitted (H {p_h}, coulomb {c})", p_h, c)):
            r = evaluate(d, ph, cc, l, h)
            print(f"{name}, {label}: n={len(d)}  agreement run-weighted {r['agree_run']:.3f}, "
                  f"cell-balanced {r['agree_cells']:.3f}; dominant loss {r['loss']:.3f}; "
                  f"share MAE {r['share_err']:.3f}")
            if label.startswith("fitted"):
                for key, v in sorted(r["cells"].items()):
                    print(f"    K/R {key[0]}, His {key[1]}{'+' if key[1] == 2 else ''}: n={v[0]} "
                          f"real 2/3/4+ {v[1]}  model {v[2]}  share MAE {v[3]}")
    if fixture:
        fx = sample(test, 25, seed=1)[["seq", "dominant"]]
        fx.to_csv(fixture, sep="\t", index=False)
        print(f"fixture: {len(fx)} held-out peptides -> {fixture}")


if __name__ == "__main__":
    main(sys.argv[1:])
