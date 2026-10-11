"""Fit the site-specific charge model's Coulomb term (and p(His)) to a real DIA-NN run.

Usage:  python charge_coulomb.py REPORT.parquet MZ_LO MZ_HI

REPORT is a DIA-NN report of a REAL tryptic run (Stripped.Sequence, Precursor.Charge, Q.Value,
Precursor.Quantity). MZ_LO/MZ_HI bound the precursor m/z the run could observe (its DIA window
range): a charge state outside it can never be the observed dominant one, so the model is held to
the same window. Target: per peptide, the charge carrying the most Precursor.Quantity; cells =
(K+R count 1-3, His count 0-2) with >= 50 peptides; loss = squared error of the dominant-charge
distribution per cell. Mirrors `ChargeModel::distribution_with_loss` (penalty on z >= 3 only).

Result recorded in `ChargeModel::realistic_with` (2026-10-11, G20513 HeLa diaPASEF, DIA-NN 2.3,
window 300-1165): coulomb 14, histidine 0.90. Per-peptide agreement 0.847 on this script's
cell-balanced sample; 0.89 on a sample weighted like the run (0.49 with independent sites).
"""
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

AA = dict(G=57.02146, A=71.03711, S=87.03203, P=97.05276, V=99.06841, T=101.04768, C=160.03065,
          L=113.08406, I=113.08406, N=114.04293, D=115.02694, Q=128.05858, K=128.09496, E=129.04259,
          M=131.04049, H=137.05891, F=147.06841, R=156.10111, Y=163.06333, W=186.07931, U=150.95364)
SITE = dict(n=0.93, R=0.97, K=0.95)


def dominant(seq, p_h, coulomb, lo, hi, max_charge=4):
    probs = [SITE["n"]] + [SITE[a] if a in "RK" else p_h for a in seq if a in "RKH"]
    pmf = np.zeros(len(probs) + 1)
    pmf[0] = 1.0
    for p in probs:
        pmf[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf[0] *= 1 - p
    mass = sum(AA.get(a, 0.0) for a in seq) + 18.01056
    best = None
    for z in range(1, min(len(pmf) - 1, max_charge) + 1):
        w = pmf[z] * (np.exp(-coulomb * (z * (z - 1) / 2 - 1) / len(seq)) if z >= 3 else 1.0)
        if lo <= (mass + z * 1.007276) / z <= hi and (best is None or w > best[1]):
            best = (z, w)
    return best[0] if best else None


def main(report, lo, hi):
    r = pq.read_table(report, columns=["Stripped.Sequence", "Precursor.Charge", "Q.Value",
                                       "Precursor.Quantity"]).to_pandas()
    r = r[r["Q.Value"] <= 0.01]
    g = r.groupby(["Stripped.Sequence", "Precursor.Charge"])["Precursor.Quantity"].sum().reset_index()
    top = g.loc[g.groupby("Stripped.Sequence")["Precursor.Quantity"].idxmax()]
    top = top.assign(kr=top["Stripped.Sequence"].str.count("[KR]"),
                     h=top["Stripped.Sequence"].str.count("H").clip(upper=2))
    rng = np.random.default_rng(0)
    cells = {}
    for (kr, h), sub in top[top.kr.between(1, 3)].groupby(["kr", "h"]):
        if len(sub) >= 50:
            cells[(kr, h)] = sub.sample(min(len(sub), 3000), random_state=rng.integers(1 << 31))
    res = []
    for p_h in (0.7, 0.8, 0.9):
        for c in (0, 8, 10, 12, 14, 16, 18, 22):
            err = agree = n = 0
            for sub in cells.values():
                m = [dominant(s, p_h, c, lo, hi) for s in sub["Stripped.Sequence"]]
                real = sub["Precursor.Charge"].value_counts(normalize=True).reindex([2, 3, 4], fill_value=0)
                mod = pd.Series(m).value_counts(normalize=True).reindex([2, 3, 4], fill_value=0)
                err += ((real - mod) ** 2).sum()
                agree += int((np.array(m) == sub["Precursor.Charge"].to_numpy()).sum())
                n += len(sub)
            res.append((err, p_h, c, agree / n))
    for err, p_h, c, a in sorted(res)[:5]:
        print(f"loss {err:.3f}  histidine {p_h}  coulomb {c}  per-peptide agreement {a:.3f}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]), float(sys.argv[3]))
