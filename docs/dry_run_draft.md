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

## P7 dry run: LLM baselines on Kaggle (S4, S7), same draft dev split

Run on Kaggle 2x T4, 2026-10-08, `notebooks/kaggle_s4_llm.ipynb`; outputs in `runs/s4/draft-dev/` (not committed). Same 40 dev items as above, so the same caveats apply: these are integration-test numbers, not results. Jais 2 overflowed in fp16 and ran in `nf4-fp32` (revision `da0e1639cd92`); Nile-Chat ran in fp32, Qwen3 in fp16.

| System | STS [95% CI] | intent macro-F1 | unsafe exec. | ECE | PM STS (n=32) | PM intent F1 |
|---|---|---|---|---|---|---|
| S4 Jais 2 8B, 0-shot, calibrated | 0.85 [0.67, 1.00] | 0.60 | 0.05 | 0.12 | 0.88 | 0.61 |
| S4 Jais 2 8B, 6-shot, calibrated | 0.53 [0.33, 0.71] | 0.45 | 0.20 | 0.46 | 0.56 | 0.47 |
| S4 Nile-Chat 4B, 0-shot, calibrated | 0.68 [0.48, 0.85] | 0.67 | 0.18 | 0.12 | 0.66 | 0.64 |
| S4 Nile-Chat 4B, 6-shot, calibrated | 0.88 [0.73, 0.98] | 0.77 | 0.03 | 0.08 | 0.91 | 0.75 |
| S4 Nile-Chat 4B, 6-shot, raw | 0.95 [0.83, 1.00] | 0.91 | 0.00 | 0.09 | 0.94 | 0.85 |
| S4 Qwen3 8B, 0-shot, calibrated | 0.55 [0.34, 0.78] | 0.51 | 0.20 | 0.42 | 0.50 | 0.43 |
| S4 Qwen3 8B, 6-shot, calibrated | 0.85 [0.70, 0.97] | 0.83 | 0.05 | 0.14 | 0.84 | 0.77 |
| S7 Jais 2 8B + QLoRA, raw | 0.93 [0.84, 1.00] | 0.87 | 0.03 | 0.16 | 0.94 | 0.87 |
| S7 Jais 2 8B + QLoRA, calibrated | 0.88 [0.71, 1.00] | 0.70 | 0.05 | 0.09 | 0.91 | 0.73 |

`calibrated` is the pre-registered primary S4 number; `raw` is shown where it differs notably. All rows are in `summary.csv`.

What the run established:
- The P7 path runs end to end on Kaggle: gated-model access check, fp16 overflow detection with a whole-model rerun in `nf4-fp32`, QLoRA training (loss 0.26 -> 0.13 over 3 epochs, 80 min) and per-domain reporting.
- `generate` gave the same label as `raw` on all 240 S4 items (0 unparseable answers): each verbalizer starts with a distinct token, so greedy decoding agrees with label scoring here.
- Contextual calibration did not help on this draft: it lowered STS in 4 of 6 model/shot settings. Few-shot prompts helped Nile-Chat and Qwen3 but hurt Jais 2.
- No LLM beat S2/S3/S5 (0.97) on these 40 items, and every interval overlaps. Nothing is decided on these numbers.
- The MASSIVE out-of-distribution cell failed in this first run (the S7 model was still on the GPUs). Fixed, and rerun in v2 below.

## P7 dry run v2: confidence gate and out-of-distribution checks (Kaggle, 2026-10-09)

Same notebook with the confidence gate, MASSIVE and the native Egyptian check; outputs in `runs/s4/draft-dev-v2/` (not committed). Same 40 draft dev items, so these are integration-test numbers, not results.

- **Reproducibility.** All 18 S4 rows are identical to the first run. **S7 is not**: STS 1.00 for both scoring methods (first run 0.93 / 0.88) with the same seed, because GPU fine-tuning is not bit-exact. On 40 items that is 3 items. S7 needs several seeds before any number is quoted.
- **Gate on dev.** τ is chosen and read on the same 40 items here, which is optimistic. When some τ reaches unsafe execution ≤ 0.02 the rule takes the smallest one, whatever it costs: Nile-Chat 0-shot raw goes from STS 0.83 / unsafe 0.08 to 0.53 / 0.00 at τ 0.8, asking for confirmation on 38% of items. When unsafe execution stays above the target (extractor errors the gate cannot see), the rule takes the τ with the lowest rate: Jais 2 0-shot goes from 0.85 / 0.05 to 0.83 / 0.03 at τ 0.95 (5% confirmations). The best settings need no gate: Nile-Chat 6-shot raw (0.95 / 0.00) and S7 get τ = 0.
- **Out of distribution, intent macro-F1 on MASSIVE ar-SA (6-shot, calibrated):** S1 0.94, Qwen3 0.61, Nile-Chat 0.42, Jais 2 0.41.
- **Native Egyptian (`Alhasan/egyptian-nlu`, 196 items), out-of-scope requests correctly refused:** S1 0.99, Qwen3 0.91, Jais 2 0.40, Nile-Chat 0.21. Every LLM error goes in the same direction: an out-of-scope request (smart home, audio, email, Q&A) taken as a reminder or an alarm. The gate would stop most of Nile-Chat's (109 of 151 at τ 0.6), which are low-confidence, but almost none of Jais 2's (2 of 115): Jais 2 is confidently wrong.
- **A likely scoring bias, to check.** The label score is the *summed* log-probability of the verbalizer tokens, and `غير مدعوم` is longer than the in-scope labels: Nile-Chat 4 tokens vs 3 (`تذكير`) and 2 (`منبه`); Qwen3 4 vs 4 and 2 (Jais 2's tokenizer is gated and was not checked). Qwen3, where `unsupported` is no longer than `reminder`, abstains far more often. Contextual calibration should remove a constant label bias but did not fix this. Before the real run: an ablation with length-normalised scores (mean log-probability per token) and single-token verbalizers, logged as a deviation.

## S4 scoring ablation (Kaggle, 2026-10-10)

`notebooks/kaggle_s4_llm.ipynb`, ablation cell only (Jais 2 116 min, Nile-Chat 51, Qwen3 28); outputs in `runs/s4/ablation/` (not committed). 6-shot prompt; dev = the 40 draft dev items (no perturbations); Egyptian = out-of-scope recall on the 192 native-written out-of-scope items of `Alhasan/egyptian-nlu`; MASSIVE = intent macro-F1.

| Model | Variant | Egyptian recall raw → calibrated | MASSIVE F1 raw → calibrated | dev STS raw → calibrated |
|---|---|---|---|---|
| Jais 2 | sum | 0.77 → 0.40 | 0.57 → 0.41 | 0.58 → 0.53 |
| Jais 2 | mean | 0.77 → 0.39 | 0.57 → 0.42 | 0.58 → 0.53 |
| Jais 2 | number | 0.47 → 0.32 | 0.34 → 0.27 | 0.78 → 0.78 |
| Nile-Chat | sum | 0.62 → 0.21 | 0.57 → 0.42 | 0.95 → 0.88 |
| Nile-Chat | mean | 0.77 → 0.18 | 0.59 → 0.37 | 0.95 → 0.88 |
| Nile-Chat | number | 0.88 → 0.12 | 0.64 → 0.24 | 0.88 → 0.83 |
| Qwen3 | sum | 0.93 → 0.91 | 0.63 → 0.61 | 0.85 → 0.85 |
| Qwen3 | mean | 0.93 → 0.89 | 0.63 → 0.60 | 0.85 → 0.85 |
| Qwen3 | number | 0.95 → 0.91 | 0.63 → 0.62 | 0.75 → 0.78 |

What it shows:
- **Contextual calibration is what suppresses `unsupported`.** It lowers Egyptian out-of-scope recall in 9 of 9 model × variant cells (by up to 77 points) and MASSIVE F1 in 9 of 9, and raises dev STS once (Qwen3 number, one item). A likely mechanism: the content-free inputs (`N/A`, empty, `[MASK]`) look out of scope, so the model gives them a high `غير مدعوم` probability, and dividing by it pushes real out-of-scope requests to the in-scope labels. The content-free probabilities were not logged, so the mechanism is not confirmed.
- **Length was not the main cause.** Mean-per-token scoring gives the same predictions as the sum for Jais 2 and Qwen3 (raw), and helps only Nile-Chat raw (0.62 → 0.77).
- **One-token number answers are inconsistent.** Jais 2 gains on dev (0.58 → 0.78) but loses Egyptian recall and never predicts alarm on the Egyptian items (alarm recall 0 of 4); Nile-Chat raw gains on Egyptian recall (0.88) but loses on dev; Qwen3 loses on dev.
- **Caveat.** The scoring choice is now informed by the Egyptian and MASSIVE items, so for S4 they are no longer untouched out-of-distribution evidence. The unbiased evidence is the locked native test split.

## Native evaluation of S1-S3 (`athar-ood`, 2026-10-10)

Draft-trained frozen S1-S3 (dry-run freeze) on the 192 human-written out-of-scope items of `Alhasan/egyptian-nlu`, and on the same requests in their Saudi/MSA original (protocol 4.1). 95% Wilson intervals; exact McNemar on the paired outcomes (not Holm-adjusted here).

| System | Egyptian out-of-scope recall | Same requests in MSA | Gap (MSA − Egyptian) | McNemar p | Alarm recall (n = 4) |
|---|---|---|---|---|---|
| S1 rules | 0.990 [0.963, 0.997] | 0.979 | −0.010 | 0.50 | 4/4 |
| S2 word TF-IDF | 0.812 [0.751, 0.861] | 0.786 | −0.026 | 0.38 | 2/4 |
| S3 + char n-grams | 0.786 [0.723, 0.839] | 0.714 | −0.073 | 0.007 | 2/4 |

No system loses on Egyptian; S3 does *better* on Egyptian than on identical MSA content. The models were trained on Egyptian drafts, so for them MSA is the unfamiliar variety. The learned models' main weakness is not dialect but coverage: about a fifth of real out-of-scope requests (smart home, audio, email) are taken as in scope, because the drafts contain few out-of-scope negatives of those kinds.

## S4 diagnostics and S7 seeds (Kaggle commit run, 2026-10-11)

`runs/s4/diagnostics/` (not committed; `report.md` from `scripts/analyse_s4_diagnostics.py`). 6-shot prompts with three train exemplar sets (shot seeds 0-2), each scored `raw`, `calibrated` (content-free) and `prior` (label prior from 60 unlabeled train texts). Dev = 40 draft items; Egyptian = the 192 native out-of-scope items. Mean ± SD over exemplar sets.

| Model | Method | dev STS | Egyptian out-of-scope recall |
|---|---|---|---|
| Jais 2 | raw | 0.57 ± 0.19 | 0.82 ± 0.04 |
| Jais 2 | calibrated | 0.49 ± 0.23 | 0.49 ± 0.08 |
| Jais 2 | prior | 0.61 ± 0.24 | **0.87 ± 0.03** |
| Nile-Chat | raw | 0.91 ± 0.04 | 0.67 ± 0.06 |
| Nile-Chat | calibrated | 0.86 ± 0.01 | 0.19 ± 0.02 |
| Nile-Chat | prior | **0.95 ± 0.04** | **0.99 ± 0.01** |
| Qwen3 | raw | 0.78 ± 0.06 | 0.96 ± 0.02 |
| Qwen3 | calibrated | 0.78 ± 0.09 | 0.89 ± 0.01 |
| Qwen3 | prior | 0.78 ± 0.07 | 0.96 ± 0.01 |

**Label distribution on content-free inputs** (what contextual calibration divides by), `unsupported` share: Jais 2 0.99-1.00, Nile-Chat 0.60-0.72, Qwen3 0.50-0.93.

What it shows:
- **Contextual calibration's failure is explained.** Empty inputs are out of scope, so the models give them mostly `unsupported`, and dividing by that suppresses abstention. In all 9 model × exemplar-set cells, `calibrated` never abstained on an item `raw` missed; the loss is significant in 8 of 9 (exact McNemar, Holm over 36 tests).
- **`prior` fixes it.** It abstained wherever `raw` did in all 9 cells, and more often: significantly for Nile-Chat in 3 of 3 sets (+31 points on average) and Jais 2 in 2 of 3; Qwen3 is at the ceiling. No dev STS difference is significant (40 items). `prior` becomes the S4 primary for the test run (protocol section 10).
- **Few-shot sensitivity dominates Jais 2.** Its dev STS spans 0.375-0.85 across exemplar sets, more than any method difference, so a single-prompt Jais 2 number is not interpretable. Nile-Chat and Qwen3 vary by 5-10 points.
- **S7 over seeds (Jais 2 + QLoRA, raw):** dev STS 0.825 / 0.950 / 0.900, mean 0.89 ± 0.06; the earlier single runs (0.93, 1.00) were within that spread.
- **Not run yet:** the LLM paired dialect gap. This run used a notebook from before the Saudi/MSA versions were added.

## Repeat run (Kaggle, 2026-10-11, same notebook as the diagnostics run)

`runs/s4/repeat-2026-10-11/` (not committed). The S4 and diagnostics cells ran a second time with the same models, revisions, prompts and seeds.

- **Inference is exactly reproducible.** All 6,372 diagnostics predictions and all content-free probabilities are identical to the first run; so is every S4 summary row.
- **S7 fine-tuning is not, even with the same seed** (and cuDNN determinism set): seed 0 gave dev STS 0.825 then 0.875, seed 1 0.950 then 0.875, seed 2 0.900 then 0.950. 4-bit QLoRA training on GPU is not bit-exact, so run-to-run variance is as large as seed-to-seed variance. **S7 is reported pooled over all 6 runs: dev STS 0.896 ± 0.049 (raw; range 0.825-0.950), intent macro-F1 0.81 ± 0.08, unsafe execution 0.025 ± 0.027.**

## LLM paired dialect gap (Kaggle, 2026-10-12, notebook 2026-10-11c)

`runs/s4/diagnostics/msa-*` (not committed). The 192 native Egyptian out-of-scope requests and their Saudi/MSA MASSIVE originals, 3 exemplar sets × 3 methods; exact McNemar per row, Holm over 18 rows. Jais 2 was not run (optional).

| Model | Method | Out-of-scope recall, MSA → Egyptian (mean of 3 sets) | Holm-significant rows |
|---|---|---|---|
| Nile-Chat | prior | 0.988 → 0.986 | 0 of 3 |
| Nile-Chat | raw | 0.694 → 0.672 | 0 of 3 |
| Nile-Chat | calibrated | 0.186 → 0.194 | 0 of 3 |
| Qwen3 | prior | 0.979 → 0.964 | 0 of 3 |
| Qwen3 | raw | 0.974 → 0.957 | 0 of 3 |
| Qwen3 | calibrated | 0.929 → 0.894 | 0 of 3 |

- **No significant dialect penalty** for either model, with any method. Qwen3 is slightly better on MSA in all 9 rows (1-4 points), a consistent but small direction; Nile-Chat shows none. Out-of-scope failures come from the scoring method (calibration) far more than from the dialect.
- The content-free probabilities in this run are identical to the first diagnostics run (inference is reproducible).
