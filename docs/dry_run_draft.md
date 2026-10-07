# Phase 3 dry run on the assistant-drafted data (v2: project-manager focus)

**This is an integration test, not a result.** The 300 draft items were written by an AI assistant (`source: assistant_draft`), so they are not the native-speaker dataset, they never enter a test split, and the author of the drafts knew the system's rules. The run shows that every stage works end to end on realistic data: split, training and dev-only search, freezing with hashes, evaluation, statistics, robustness and calibration. The numbers say nothing reliable about the research question.

Split: `data/draft/manifest.json`, train 260 / dev 40 items, 40 dev items in 17 families. Contract v1.0. Freeze `18f0dfa1e5d7`. Bootstrap/permutation: 10,000 / 10,000.

## Dev results (95% family-bootstrap CI)

| System | Safe Task Success | intent macro-F1 | route macro-F1 | unsafe execution | unnecessary clarification |
|---|---|---|---|---|---|
| S1 | 0.90 [0.78, 1.00] | 0.85 [0.65, 1.00] | 0.85 [0.65, 1.00] | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] |
| S2 | 0.97 [0.92, 1.00] | 0.95 [0.75, 1.00] | 0.96 [0.79, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| S3 | 0.97 [0.92, 1.00] | 0.95 [0.75, 1.00] | 0.96 [0.79, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| S5 | 0.97 [0.92, 1.00] | 0.95 [0.83, 1.00] | 0.96 [0.87, 1.00] | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] |
| S6-s0 | 0.93 [0.80, 1.00] | 0.84 [0.58, 1.00] | 0.86 [0.63, 1.00] | 0.05 [0.00, 0.13] | 0.00 [0.00, 0.00] |
| S6-s1 | 0.97 [0.92, 1.00] | 0.95 [0.83, 1.00] | 0.96 [0.87, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| S6-s2 | 0.95 [0.84, 1.00] | 0.90 [0.60, 1.00] | 0.91 [0.63, 1.00] | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] |
| S6-s3 | 0.97 [0.92, 1.00] | 0.95 [0.83, 1.00] | 0.96 [0.87, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| S6-s4 | 0.95 [0.84, 1.00] | 0.90 [0.60, 1.00] | 0.91 [0.63, 1.00] | 0.03 [0.00, 0.08] | 0.00 [0.00, 0.00] |
| oracle | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |

**S6 over 5 seeds:** STS 0.955 ± 0.021 (range 0.925-0.975); intent macro-F1 0.909 ± 0.046.

## By domain (95% family-bootstrap CI)

| System | project-manager STS | general STS | project-manager intent F1 | general intent F1 |
|---|---|---|---|---|
| S1 | 0.91 [0.76, 1.00] (n=32) | 0.88 [0.62, 1.00] (n=8) | 0.87 [0.63, 1.00] (n=32) | 0.86 [0.56, 1.00] (n=8) |
| S2 | 0.97 [0.90, 1.00] (n=32) | 1.00 [1.00, 1.00] (n=8) | 0.92 [0.56, 1.00] (n=32) | 1.00 [1.00, 1.00] (n=8) |
| S3 | 0.97 [0.90, 1.00] (n=32) | 1.00 [1.00, 1.00] (n=8) | 0.92 [0.56, 1.00] (n=32) | 1.00 [1.00, 1.00] (n=8) |
| S5 | 1.00 [1.00, 1.00] (n=32) | 0.88 [0.62, 1.00] (n=8) | 1.00 [1.00, 1.00] (n=32) | 0.86 [0.56, 1.00] (n=8) |
| S6-s0 | 0.97 [0.89, 1.00] (n=32) | 0.75 [0.25, 1.00] (n=8) | 0.92 [0.56, 1.00] (n=32) | 0.61 [0.25, 1.00] (n=8) |
| S6-s1 | 1.00 [1.00, 1.00] (n=32) | 0.88 [0.62, 1.00] (n=8) | 1.00 [1.00, 1.00] (n=32) | 0.86 [0.56, 1.00] (n=8) |
| S6-s2 | 1.00 [1.00, 1.00] (n=32) | 0.75 [0.25, 1.00] (n=8) | 1.00 [1.00, 1.00] (n=32) | 0.61 [0.25, 1.00] (n=8) |
| S6-s3 | 1.00 [1.00, 1.00] (n=32) | 0.88 [0.62, 1.00] (n=8) | 1.00 [1.00, 1.00] (n=32) | 0.86 [0.56, 1.00] (n=8) |
| S6-s4 | 1.00 [1.00, 1.00] (n=32) | 0.75 [0.25, 1.00] (n=8) | 1.00 [1.00, 1.00] (n=32) | 0.61 [0.25, 1.00] (n=8) |
| oracle | 1.00 [1.00, 1.00] (n=32) | 1.00 [1.00, 1.00] (n=8) | 1.00 [1.00, 1.00] (n=32) | 1.00 [1.00, 1.00] (n=8) |

## Pre-registered comparisons (on dev, for illustration only)

| Comparison | STS difference [95% CI] | permutation p | Holm p | McNemar p |
|---|---|---|---|---|
| S2 vs S1 | +0.075 [-0.031, +0.205] | 0.5013 | 1.0000 | 0.3750 |
| S3 vs S1 | +0.075 [-0.031, +0.205] | 0.5013 | 1.0000 | 0.3750 |
| S3 vs S2 | +0.000 [+0.000, +0.000] | 1.0000 | 1.0000 | 1.0000 |

## Robustness: invariance rate under label-preserving perturbations

| System | taa_marbuta | hamza | alef_maqsura | digits | diacritics | elongation | punctuation | politeness | english_trigger |
|---|---|---|---|---|---|---|---|---|---|
| S1 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |
| S2 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.87 |
| S3 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.90 | 0.83 |
| S5 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 1.00 | 0.70 |
| S6-s0 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 0.97 | 0.96 |
| oracle | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 | 1.00 |

## Calibration of the intent probabilities

| System | ECE | Brier | AURC |
|---|---|---|---|
| S2 | 0.225 | 0.135 | 0.001 |
| S3 | 0.269 | 0.175 | 0.001 |
| S5 | 0.221 | 0.145 | 0.010 |
| S6-s0 | 0.067 | 0.135 | 0.004 |
| S6-s1 | 0.021 | 0.049 | 0.002 |
| S6-s2 | 0.046 | 0.098 | 0.004 |
| S6-s3 | 0.028 | 0.053 | 0.001 |
| S6-s4 | 0.042 | 0.088 | 0.003 |

## Selected settings and cost (CPU, from `freeze.json`)

| System | selected on dev | train time | predict per item |
|---|---|---|---|
| S1 | none (rules) | - | 0.02 ms |
| S2 | C=3, word [1, 1] | 0.06 s CPU | 0.05 ms |
| S3 | C=1, word [1, 1], char [2, 4] | 0.16 s CPU | 0.11 ms |
| S5 | C=10 | 8 s wall (incl. embedding) | 19.3 ms |
| S6 | fixed (no search) | 29.8 min wall per seed | 31.7 ms |

Machine: Intel64 Family 6 Model 154 Stepping 4, GenuineIntel, 12 logical CPUs, 31.6 GB RAM; process peak RSS 5933 MB. Tokenizer check: S6 maps 0.0% of training word pieces to [UNK].

## What the dry run established

- The whole Phase 3 path runs on real model weights (pinned revisions), with no manual steps.
- The dev-only search behaves as pre-registered: flat grids are resolved by the simplicity tie-break.
- With 40 dev items, STS intervals are 8-22 points wide, and dev rankings change between drafts (v1 general vs v2 project-manager). This is the small-sample problem the protocol's power analysis anticipates (section 5.1). The test set is therefore 120 items, decisions rest on pre-registered tests, and no system is chosen on these numbers.
- Dev STS on this draft (point estimates only): S2 0.97, S3 0.97, S5 0.97, S1 0.90; S6 mean over seeds 0.95.
- It caught a measurement bug before any real data was touched. Invariance compared the user's spelling of the reminder text instead of its canonical form. It is fixed, a regression test was added, and the protocol states the definition.
- It also caught a bias in small-sample macro-F1. A label absent from a bootstrap resample was scored 0 instead of being left out, which pulled subgroup intervals down (e.g. 1.00 [0.33, 1.00]). It is fixed in both implementations, tested against each other, and the protocol states the rule.
- Frozen models can be re-scored on dev after a scoring-code fix (model files stay hash-checked). Only the locked test run requires an unchanged freeze.
