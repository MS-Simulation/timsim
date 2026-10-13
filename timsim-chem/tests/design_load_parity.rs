//! Byte-identity gate for the per-condition load (`design::Condition::load_ng`, 0.4.0).
//!
//! The field is optional, and leaving it unset must change nothing: every number
//! `design::resolve` emits for a design that sets no per-condition load has to be bit for bit what
//! the last release without the field (0.3.0) emitted. The code says it does — an unset load
//! resolves to the design's own `load_ng`, so every product and quotient is the one 0.3.0
//! computed, and every load ratio is exactly 1 — but an argument is not a test.
//!
//! So this runs both. The old side is the published 0.3.0 crate itself, pinned as a
//! dev-dependency, not a copy of its code that could drift; and because both sides run in one
//! process against one libm, "bit-identical" is a statement that holds on any platform.

use std::collections::BTreeMap;
use timsim_chem::design as new;
use timsim_chem_v030::design as old;

const AAS: &[u8] = b"ACDEFGHIKLMNPQRSTVWY";

/// Three organisms of different size and protein length, from a fixed LCG (no RNG crate, so the
/// corpus is identical on every run), plus one protein with a residue we cannot weigh.
fn proteome() -> Vec<(String, String, String)> {
    let mut s: u64 = 0x9e37_79b9_7f4a_7c15;
    let mut next = || {
        s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (s >> 33) as usize
    };
    let mut v = Vec::new();
    for (org, prefix, n, spread) in [
        ("HUMAN", "HUM", 60, 400),
        ("YEAST", "YST", 35, 300),
        ("ECOLI", "ECO", 15, 150),
    ] {
        for i in 0..n {
            let len = 50 + next() % spread;
            let seq: String = (0..len).map(|_| AAS[next() % AAS.len()] as char).collect();
            v.push((format!("{prefix}{i}"), seq, org.to_string()));
        }
    }
    // Skipped and reported, never guessed — and the report must say so identically.
    v.push(("BAD1".into(), "PEPTXDEK".into(), "HUMAN".into()));
    v
}

fn condition(
    name: &str,
    mix: &[(&str, new::Share)],
    regulate: Vec<new::Regulation>,
) -> new::Condition {
    new::Condition {
        name: name.into(),
        mix: mix.iter().map(|(o, s)| (o.to_string(), *s)).collect(),
        load_ng: None,
        replicates: 3,
        technical_replicates: 2,
        regulate,
    }
}

/// Every code path the load touches, plus the ones it must leave alone: three abundance profiles,
/// biological variance with per-protein heterogeneity and class multipliers, explicit and
/// generative regulation, `rest`, a zero share (an undefined fold change, i.e. NaN), forcing and
/// displacement under `n_proteins`, and a skipped protein.
fn variants() -> Vec<(&'static str, new::DesignSpec)> {
    use new::Share::{Fraction, Rest};
    let mut abundance = BTreeMap::new();
    abundance.insert(
        "HUMAN".to_string(),
        new::AbundanceProfile::LogNormal { sigma: 2.0 },
    );
    abundance.insert(
        "YEAST".to_string(),
        new::AbundanceProfile::HockeyStick {
            decay: 0.06,
            tail: 1e-4,
        },
    );
    abundance.insert(
        "ECOLI".to_string(),
        new::AbundanceProfile::Table(
            (0..15)
                .map(|i| (format!("ECO{i}"), 1.0 + 0.37 * i as f64))
                .collect(),
        ),
    );
    let base = new::DesignSpec {
        reference: "A".into(),
        load_ng: 200.0,
        complexity: new::Complexity::default(),
        abundance,
        conditions: vec![
            condition(
                "A",
                &[
                    ("HUMAN", Fraction(0.65)),
                    ("YEAST", Fraction(0.30)),
                    ("ECOLI", Fraction(0.05)),
                ],
                vec![],
            ),
            condition(
                "B",
                &[
                    ("HUMAN", Fraction(0.65)),
                    ("YEAST", Fraction(0.20)),
                    ("ECOLI", Rest),
                ],
                vec![
                    new::Regulation::Explicit {
                        proteins: [("HUM3".to_string(), 1.6), ("YST2".to_string(), -0.7)]
                            .into_iter()
                            .collect(),
                    },
                    new::Regulation::Generative {
                        fraction: 0.1,
                        log2fc_sd: 1.0,
                    },
                ],
            ),
            condition(
                "C",
                &[
                    ("HUMAN", Fraction(0.5)),
                    ("YEAST", Rest),
                    ("ECOLI", Fraction(0.0)),
                ],
                vec![],
            ),
        ],
        variance: new::Variance {
            biological: 0.2,
            biological_heterogeneity: 0.5,
            technical: 0.05,
            cv_multipliers: [("HUM1".to_string(), 0.5), ("YST4".to_string(), 1.4)]
                .into_iter()
                .collect(),
        },
        seed: 42,
    };

    let mut subset = base.clone();
    subset.complexity = new::Complexity {
        n_proteins: Some(30),
    };
    subset.conditions[1]
        .regulate
        .push(new::Regulation::Explicit {
            proteins: (10..18).map(|i| (format!("HUM{i}"), 0.8)).collect(),
        });

    let mut flat = base.clone();
    flat.variance = new::Variance::default();

    let mut other = base.clone();
    other.load_ng = 37.5;
    other.seed = 7;
    other.reference = "B".into();

    vec![
        ("base", base),
        ("subset", subset),
        ("no variance", flat),
        ("37.5 ng, seed 7, ref B", other),
    ]
}

fn to_old(s: &new::DesignSpec) -> old::DesignSpec {
    old::DesignSpec {
        reference: s.reference.clone(),
        load_ng: s.load_ng,
        complexity: old::Complexity {
            n_proteins: s.complexity.n_proteins,
        },
        abundance: s
            .abundance
            .iter()
            .map(|(org, a)| {
                let a = match a {
                    new::AbundanceProfile::LogNormal { sigma } => {
                        old::AbundanceProfile::LogNormal { sigma: *sigma }
                    }
                    new::AbundanceProfile::HockeyStick { decay, tail } => {
                        old::AbundanceProfile::HockeyStick {
                            decay: *decay,
                            tail: *tail,
                        }
                    }
                    new::AbundanceProfile::Table(t) => old::AbundanceProfile::Table(t.clone()),
                };
                (org.clone(), a)
            })
            .collect(),
        conditions: s
            .conditions
            .iter()
            .map(|c| {
                assert!(
                    c.load_ng.is_none(),
                    "0.3.0 has no per-condition load to compare against"
                );
                old::Condition {
                    name: c.name.clone(),
                    mix: c
                        .mix
                        .iter()
                        .map(|(org, share)| {
                            let share = match share {
                                new::Share::Fraction(f) => old::Share::Fraction(*f),
                                new::Share::Rest => old::Share::Rest,
                            };
                            (org.clone(), share)
                        })
                        .collect(),
                    replicates: c.replicates,
                    technical_replicates: c.technical_replicates,
                    regulate: c
                        .regulate
                        .iter()
                        .map(|r| match r {
                            new::Regulation::Explicit { proteins } => old::Regulation::Explicit {
                                proteins: proteins.clone(),
                            },
                            new::Regulation::Generative {
                                fraction,
                                log2fc_sd,
                            } => old::Regulation::Generative {
                                fraction: *fraction,
                                log2fc_sd: *log2fc_sd,
                            },
                        })
                        .collect(),
                }
            })
            .collect(),
        variance: old::Variance {
            biological: s.variance.biological,
            biological_heterogeneity: s.variance.biological_heterogeneity,
            technical: s.variance.technical,
            cv_multipliers: s.variance.cv_multipliers.clone(),
        },
        seed: s.seed,
    }
}

/// Everything a resolved design emits, one line per row, every float by its BITS — so equal means
/// bit-identical, NaN included (an undefined `true_log2fc` is a defined answer, not a wildcard).
/// A macro because the old and new `Design` are different types with the same fields.
macro_rules! canon {
    ($d:expr) => {{
        let d = &$d;
        let b = |x: f64| format!("{:016x}", x.to_bits());
        let mut v: Vec<String> = Vec::new();
        for s in &d.samples {
            v.push(format!(
                "sample {} {} {}",
                s.sample_id, s.condition, s.replicate
            ));
        }
        for r in &d.runs {
            v.push(format!(
                "run {} {} {}",
                r.run_id, r.technical_replicate, r.injection_order
            ));
        }
        for m in &d.sample_runs {
            v.push(format!(
                "map {} {} {:?} {}",
                m.sample_id,
                m.run_id,
                m.channel,
                b(m.mix_fraction)
            ));
        }
        for q in &d.protein_quantities {
            v.push(format!(
                "quantity {} {} {} {} {:?} {}",
                q.protein_id,
                q.sample_id,
                b(q.amount_amol),
                b(q.true_log2fc),
                q.requested_log2fc.map(b),
                q.is_regulated
            ));
        }
        let r = &d.report;
        for o in &r.organisms {
            let per = |m: &BTreeMap<String, f64>| {
                m.iter()
                    .map(|(c, x)| format!("{c}={}", b(*x)))
                    .collect::<Vec<_>>()
            };
            v.push(format!(
                "organism {} mean_mw {} mass_fraction {:?} true_log2fc {:?}",
                o.organism,
                b(o.mean_mw),
                per(&o.mass_fraction),
                per(&o.true_log2fc)
            ));
        }
        v.push(format!(
            "load {} balance {}",
            b(r.load_ng),
            b(r.mass_balance_error_ng)
        ));
        v.push(format!(
            "skipped {:?} forced {:?} displaced {:?}",
            r.skipped_proteins, r.forced_present, r.displaced
        ));
        v
    }};
}

fn assert_same(label: &str, got: &[String], want: &[String]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: a different number of output rows"
    );
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert_eq!(g, w, "{label}: first difference at output row {i}");
    }
}

#[test]
fn no_per_condition_load_is_bit_identical_to_0_3_0() {
    let ps = proteome();
    let new_dp: Vec<new::DesignProtein> = ps
        .iter()
        .map(|(id, seq, org)| new::DesignProtein {
            id,
            sequence: seq,
            organism: org,
        })
        .collect();
    let old_dp: Vec<old::DesignProtein> = ps
        .iter()
        .map(|(id, seq, org)| old::DesignProtein {
            id,
            sequence: seq,
            organism: org,
        })
        .collect();

    for (label, spec) in variants() {
        let got = new::resolve(&spec, &new_dp).unwrap();
        let want = old::resolve(&to_old(&spec), &old_dp).unwrap();
        assert_same(label, &canon!(got), &canon!(want));

        // Nothing overrides, so every condition reports the design's load, bit for bit.
        assert_eq!(got.report.condition_load_ng.len(), spec.conditions.len());
        for (c, l) in &got.report.condition_load_ng {
            assert_eq!(
                l.to_bits(),
                spec.load_ng.to_bits(),
                "{label}: condition {c}"
            );
        }
    }
}

/// The comparison above is only as strong as what the variants exercise, so assert that they do.
#[test]
fn the_parity_variants_exercise_what_they_claim_to() {
    let ps = proteome();
    let dp: Vec<new::DesignProtein> = ps
        .iter()
        .map(|(id, seq, org)| new::DesignProtein {
            id,
            sequence: seq,
            organism: org,
        })
        .collect();
    let v = variants();
    let base = new::resolve(&v[0].1, &dp).unwrap();
    assert_eq!(base.report.skipped_proteins, vec!["BAD1".to_string()]);
    assert!(base
        .protein_quantities
        .iter()
        .any(|q| q.requested_log2fc.is_some()));
    assert!(base
        .protein_quantities
        .iter()
        .any(|q| q.true_log2fc.is_nan()));
    assert!(base
        .report
        .organisms
        .iter()
        .any(|o| o.true_log2fc.values().any(|x| x.is_nan())));
    let subset = new::resolve(&v[1].1, &dp).unwrap();
    assert!(!subset.report.forced_present.is_empty() && !subset.report.displaced.is_empty());
}

/// Setting a condition's load to the design's own value is the same design, so it must be the same
/// bits — the override path and the default path do not differ in anything but where the number
/// came from.
#[test]
fn an_override_equal_to_the_design_load_is_bit_identical() {
    let ps = proteome();
    let dp: Vec<new::DesignProtein> = ps
        .iter()
        .map(|(id, seq, org)| new::DesignProtein {
            id,
            sequence: seq,
            organism: org,
        })
        .collect();
    for (label, spec) in variants() {
        let mut explicit = spec.clone();
        for c in &mut explicit.conditions {
            c.load_ng = Some(spec.load_ng);
        }
        let a = new::resolve(&spec, &dp).unwrap();
        let b = new::resolve(&explicit, &dp).unwrap();
        assert_same(label, &canon!(b), &canon!(a));
        assert_eq!(a.report.condition_load_ng, b.report.condition_load_ng);
    }
}
