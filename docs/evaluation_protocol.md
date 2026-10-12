# ATHAR Chat evaluation protocol (pre-registered, v1.0 draft)

This document fixes, **before any system is trained on the annotated data**, what will be
measured, how uncertainty is quantified, which comparisons are tested and how results are
read. Its SHA-256, together with `configs/eval.yaml`, the task contract and the scoring code,
is recorded in `models/freeze.json` by `athar-train fit`. The one-shot test run
(`athar-eval --split test --unlock-test`) refuses to start if any of them changed. Departures
after freezing go to section 10 with a date and a reason. They are never applied silently.

## 1. Research question and hypotheses

**RQ.** All systems share the normalizer, argument extractor, execution policy and response
templates. Under that condition, does lightweight supervised intent classification improve
**Safe Task Success (STS)** over hand-written keyword rules for single-turn Egyptian Arabic
reminder/alarm requests?

| | Comparison | Hypothesis | Role |
|---|---|---|---|
| H1 | S2 vs S1 | STS(S2) ≠ STS(S1) | primary |
| H2 | S3 vs S1 | STS(S3) ≠ STS(S1) | primary |
| H3 | S3 vs S2 | STS(S3) ≠ STS(S2): do character n-grams add value? | primary |

Tests are two-sided. The family-wise error rate is held at α = 0.05 with Holm-Bonferroni
over H1-H3. Secondary analyses (section 6) are descriptive and carry no significance claims.

> **Status (2026-10-10, section 10).** No contributor data was collected. The native test
> data is instead **written by the project author**, a native Egyptian Arabic speaker
> (section 4.2): 120 requests (50 ACT / 50 CLARIFY / 20 ABSTAIN) plus up to 30 spare. **H1-H3
> are tested on that split once it is complete**; until then they are untested, and the
> project reports the drafts as development evidence only plus the external native evaluation
> of section 4.1.

## 2. Systems

| | System | Learns from | Status |
|---|---|---|---|
| S1 | keyword rules (`athar.intent.rules`), lists written from the contract scope | nothing (lists may be extended after reading train only) | core |
| S2 | word TF-IDF (1-2 g) + logistic regression | train | core |
| S3 | S2 + character `char_wb` n-grams | train | core |
| Oracle | gold intent + the shared pipeline | gold labels | ceiling: isolates extractor/policy error |
| S4 | instruction-tuned LLMs, intent only: `inception42/Jais-2-8B-Chat` (primary), `MBZUAI-Paris/Nile-Chat-4B` (Egyptian), `Qwen/Qwen3-8B` (multilingual reference); 0-shot and 6-shot | few-shot exemplars from train | extended, reported separately |
| S7 | Jais 2 8B + QLoRA (r 16, α 32, 3 epochs, LR 2e-4, fixed) | train | exploratory |
| S5 | frozen `intfloat/multilingual-e5-small` embeddings (revision `614241f`) + logistic regression | train | extended, reported separately |
| S6 | `UBC-NLP/MARBERTv2` (revision `fe88db9`) fine-tuned with a 3-way head, 5 seeds | train | extended, reported separately |

Every system predicts **only** the intent label (`create_reminder`, `create_alarm`,
`unsupported`). The route always comes from the shared deterministic policy. The extended
systems answer the question a reviewer will ask ("would a pretrained model do better?").
They are outside H1-H3 and get no significance tests against the core systems.

**Model selection (dev only).** S2/S3: grid over C ∈ {0.1, 0.3, 1, 3, 10, 30}, word n-gram
upper bound ∈ {1, 2}, (S3) character n-gram range ∈ {2-4, 2-5, 3-5}. The winner has the highest
dev intent macro-F1; ties go to the simpler model (smaller C, then shorter n-grams). It is then
refitted on train only. The vectorizers never see dev or test text during fitting.

S5 searches only C ∈ {0.1, 0.3, 1, 3, 10, 30}, by the same rule. The encoder was chosen in
advance: multilingual, CPU-sized, with L2-normalised mean-pooled embeddings and the `query: `
prefix, and it is never tuned.

S4 scores each label's Arabic verbalizer (`تذكير`, `منبه`, `غير مدعوم`) after a fixed chat
prompt (`athar.intent.llm`). The **primary S4 number for the locked test run is the 6-shot
`prior` method** (final; section 10: pre-registered `calibrated`, then `raw` on 2026-10-10,
then `prior` on 2026-10-11, all before the test split existed): two train exemplars per label,
chosen with a fixed seed from distinct families; summed verbalizer log-probabilities, divided
by the mean label distribution the same prompt gives 60 unlabeled train requests (after batch
calibration, Zhou et al. 2024), then renormalised. Also reported: `raw` (no calibration),
contextual calibration with content-free inputs (`calibrated`, Zhao et al. 2021), and greedy
`generate`-then-parse,
because first-token scores and generated answers of chat models can disagree (Wang et al.
2024). Few-shot sensitivity is reported over three exemplar sets (shot seeds `seed`, `seed+1`,
`seed+2`, all from train) as mean ± SD; the primary number uses the first. A generated answer that names no label is a
failure. The models run on Kaggle T4 GPUs (`notebooks/kaggle_s4_llm.ipynb`) with the same
`athar` scoring code, in fp16, or fp32 / 4-bit NF4 (fp32 compute) when fp16 overflows, and the model revision
is recorded with each result.

S6 has no dev search. Every setting is fixed in `configs/train.yaml` before training, following
the small-data recipe of Mosbach et al. (2021): AdamW with learning rate 2e-5 and weight decay
0.01, 20 epochs, batch size 16, linear warm-up over 10% of steps then linear decay, gradient
clipping at 1.0, class-balanced cross-entropy (the counterpart of `class_weight="balanced"` in
S2/S3) and a maximum length of 64 word pieces. It is trained with seeds 0-4, and the report
gives the mean and SD over seeds (Reimers & Gurevych 2017). Before training, the tokenizer must
map at most 5% of the training word pieces to `[UNK]`. On the draft data MARBERTv2 maps 0%.
Models are loaded from the Hugging Face Hub at the pinned revisions above.

## 3. Metrics (item level, then aggregated)

An output that fails the response schema scores as a failure and never as an execution.

**Safe Task Success (primary).** An item succeeds when:
- gold `ABSTAIN`: the system abstains;
- gold `CLARIFY`: the route is CLARIFY, the intent equals gold, and the clarification-target
  set equals the gold set exactly;
- gold `ACT`: the route is ACT with the gold tool, the datetime is the same instant (compared in
  UTC), and for reminders the canonical reminder text equals the gold text or an annotated
  alternative.

STS is the mean over items. It is reported overall and per gold route.

**Secondary metrics.**

| Metric | Definition |
|---|---|
| intent macro-F1 | over the three intent labels; a label absent from both gold and prediction in a (re)sample is left out, not scored 0 |
| route macro-F1 | over ACT / CLARIFY / ABSTAIN, same rule |
| unsafe-execution rate | items where a tool call was issued and the item is not a success, ÷ all items |
| unsafe execution on non-ACT | tool calls on gold CLARIFY/ABSTAIN items ÷ those items |
| wrong-argument execution | gold-ACT items executed with a wrong tool, time or text ÷ gold-ACT items |
| execution precision | successful executions ÷ executions |
| unnecessary clarification | gold-ACT items answered with CLARIFY ÷ gold-ACT items |
| over-abstention / missed abstention | ABSTAIN on supported items; non-ABSTAIN on gold ABSTAIN |
| pair success | complete/incomplete pairs with both items correct ÷ pairs |
| per-class P/R/F1, confusion matrices | intent and route |

Implementation: `athar.eval.metrics` (unit-tested; frozen).

## 4. Data

- 300 single-turn requests written and labelled by native speakers following
  `docs/annotation_guide.md` and contract v1.0. Agreement on 60 items is reported with Cohen's
  κ before adjudication.
- Family-level split (`athar-data split`): train 140, dev 40, test 120 (50 ACT, 50 CLARIFY,
  20 ABSTAIN). Assistant-drafted items never enter test. Each item is hashed in the manifest.
- The test split is read exactly once, by the evaluation run, after `athar-train fit` has
  frozen all systems.

### 4.1 What is evaluated beside it (2026-10-10)

No contributor data was collected (section 10). The data used:

| Data | Written by | Role | What can be claimed |
|---|---|---|---|
| `data/draft/athar_draft.jsonl` (300; train 260 / dev 40) | an AI assistant (`source: assistant_draft`) | training; development of every component | nothing about native users: the author of the drafts knew the system rules |
| `Alhasan/egyptian-nlu` human test split (196 after mapping) | a native Egyptian speaker, independently of this project | **native evaluation**, intent level | out-of-scope recall (n = 192) and its Wilson CI; alarm recall (n = 4, descriptive) |
| the same 196 requests in their Saudi/MSA MASSIVE original (`egy_test.csv`) | MASSIVE annotators | **paired dialect gap** | the change in out-of-scope recall from MSA to Egyptian on identical content (exact McNemar) |
| MASSIVE ar-SA test (balanced, ≤100 per label) | MASSIVE annotators | out-of-distribution, MSA | intent macro-F1 |

All systems are trained on drafts and evaluated on human-written data, so every native result
is also a **synthetic-to-real transfer** result. The native set covers abstention well and
execution hardly at all: it has no reminders, 4 alarm requests and no reference times, so
route, argument and STS claims are limited to the drafts. The native set was inspected during
development (S1-S3 dry runs; the S4 scoring choice, section 10), so it is reported as
external evidence, not as an untouched test set.

### 4.2 Author-written test data (2026-10-10)

- **Who and how.** The author writes the wave-1 requests (`contrib/author_sheet_wave1.xlsx`,
  `athar-data author-sheet`): situations in English with no Arabic example wording, the 120
  core requests first, in 2-3 dated sittings, without looking at the drafts or the pilot.
  Items are `source: author`; every card is one family and its two versions are a pair.
- **Split.** `data/athar.jsonl` holds the author items and the drafts. The splitter puts only
  non-draft families in test (50 / 50 / 20); dev (40) and train come from the rest, so
  training is mostly on drafts.
- **Labels.** The author labels with `docs/annotation_guide.md`. Agreement on 60 items is
  measured with a second annotator, or, failing that, a self-review at least 7 days later
  (guide 7.2); κ is reported before adjudication.
- **Threats.** (1) One writer: the test measures one native speaker's style, not the
  population of users. (2) The writer knows the contract and the systems; S1's rules may be
  favoured on in-scope phrasing (section 9). (3) Training on drafts and testing on native text
  is a domain shift for the learned systems. The external native set (4.1, a different
  writer) is reported beside it as a partial check on (1) and (2).

## 5. Statistical analysis

**Unit of resampling: the scenario family.** Paraphrases and complete/incomplete pairs share a
family, so items are not independent. Every interval and test resamples or permutes whole
families.

- **Confidence intervals.** 95% percentile intervals from a family (cluster) bootstrap with
  10,000 resamples, for STS, intent and route macro-F1, unsafe-execution rate and unnecessary
  clarification, per system.
- **Paired tests (H1-H3).** Approximate randomization: 10,000 permutations, swapping the two
  systems' outputs per family. The statistic is the STS difference; the p-value counts the
  observed assignment. The effect size is the STS difference with a 95% family-bootstrap CI.
- **Multiplicity.** Holm-Bonferroni over H1-H3 at α = 0.05.
- **Sensitivity.** Exact McNemar test on item-level success, which ignores families. It is
  reported next to each primary test and does not decide significance.
- **Seeds.** S1-S3 are deterministic. Bootstrap and permutation seeds are fixed in
  `configs/eval.yaml`. S6 reports mean ± SD over 5 training seeds.

Implementation: `athar.eval.stats` (unit-tested, including agreement with known results).

### 5.1 Power and how results are read

Simulated power of the paired permutation test at the planned test size (120 items in about
60 families of two; `athar.eval.stats.simulate_power`, 400 simulations). *Coupling* is the
share of items on which the two systems' outcomes are drawn independently. Lower coupling
means the systems mostly succeed and fail together, which is expected here because they share
most of the pipeline.

| STS S1 | STS S2/S3 | difference | power at coupling 0.25 | 0.50 | 1.00 |
|---|---|---|---|---|---|
| 0.70 | 0.75 | 5 points | 0.22 | 0.15 | 0.11 |
| 0.70 | 0.80 | 10 points | 0.68 | 0.51 | 0.38 |
| 0.70 | 0.85 | 15 points | 0.96 | 0.90 | 0.73 |
| 0.80 | 0.90 | 10 points | 0.81 | 0.68 | 0.50 |

Consequences, fixed in advance:
1. Differences of about 15 STS points or more are detectable. Differences of 5 points are not.
2. A non-significant result is **inconclusive**, not evidence that the systems are equal. The
   report then gives the difference's 95% CI and states which effect sizes it rules out.
3. No claim of superiority on a secondary metric without its CI.

## 6. Secondary analyses (descriptive)

- **Oracle gap.** STS(oracle) − STS(system) splits failures into intent errors and
  extraction/policy errors.
- **Robustness** (CheckList INV tests). These are label-preserving perturbations of every test
  item: `ة→ه`, hamza removal, final `ي→ى`, Western ↔ Arabic-Indic digits, diacritics,
  elongation, punctuation, a `لو سمحت` prefix, and `فكرني → remind me`. For each, the report
  gives STS and the invariance rate: the share of items whose decision is unchanged, i.e. the same route, intent, target set, tool, instant and *canonical* reminder text. Echoing the user's own spelling of the reminder is correct behaviour, not a change. The minimal pairs serve as the
  directional tests.
- **Calibration and selective prediction** (S2, S3, S5, S6). ECE (10 equal-width bins),
  multi-class Brier score, and the area under the risk-coverage curve, for the intent
  probabilities.
- **Confidence gate** (S2, S3, S5, S6, and S4/S7 on Kaggle). The confidence of a prediction is
  the probability of the chosen intent. Below a threshold τ, an execution becomes a
  confirmation question (contract 3, target `confirm`); CLARIFY and ABSTAIN are never changed,
  so the gate can only remove executions. τ is chosen **on dev only** at `athar-train fit`,
  by a rule fixed in `configs/train.yaml`: the smallest τ in {0, .4, .5, .6, .7, .8, .9, .95}
  whose dev unsafe-execution rate is ≤ 0.02 (if none, the τ with the lowest rate; ties: the
  smaller τ). It is frozen with the models. On test the report gives, for each probabilistic
  system: all metrics with CIs at the frozen τ, and the risk-coverage table (coverage, gated
  share, STS, unsafe execution, execution precision, unnecessary clarification) at every τ.
  A confirmation scores as a failure on every gold route, so the gate trades STS (more
  unnecessary clarification) for fewer unsafe executions; both sides are reported. **H1-H3 are
  tested on the ungated systems**; the gate is descriptive.
- **Domain.** STS and intent macro-F1 for **project-manager requests** (`domain:
  project_management`, the main target users, about 70% of the data) and for general requests
  separately, with CIs. This is the first subgroup reported for every system.
- **Out of distribution.** Intent macro-F1 on MASSIVE ar-SA test (Saudi-localised MSA, CC BY
  4.0). Mapping: `alarm_set` → alarm; `alarm_remove`, `calendar_remove`, `weather_query`,
  `play_music`, `email_sendemail`, `calendar_query` → unsupported; `calendar_set` excluded (it
  mixes reminders with calendar events). Balanced sample of up to 100 per label, seed 0.
  Intent only, because MASSIVE has no datetime gold.
- **Out of distribution, native Egyptian** (`athar-ood`; LLMs in the Kaggle notebook). The
  200-item human test split of `Alhasan/egyptian-nlu` (CC BY 4.0; commands written and
  annotated by a native Egyptian speaker, MASSIVE schema), pinned revision `5094a703` and
  SHA-256. Same mapping as MASSIVE (`calendar_set` excluded, 196 items). Only 4 items set an
  alarm, so the reported numbers are **unsupported recall** (n = 192: does the system abstain
  on real out-of-scope Egyptian requests?) with a 95% Wilson interval, and alarm recall
  (n = 4, descriptive), plus the confidence of the false in-scope predictions (would the gate
  have stopped them?), and the **paired dialect gap**: the same items in their Saudi/MSA
  original, exact McNemar on the paired out-of-scope outcomes (section 4.1).
- **Subgroups.** STS for code-switched vs Arabic-only items, and for reminders vs alarms, with
  CIs. Subgroups are small, so these are descriptive only.
- **Resources.** CPU time and peak memory for training and per-item prediction latency, from
  `models/freeze.json`.

## 7. Error analysis

Up to 30 test failures, sampled across systems and stratified by gold route. Each is coded
with one primary cause: intent, normalization, temporal extraction, reminder-text extraction,
policy/routing, or annotation. A second pass by the second annotator codes 10 of them, and
their agreement is reported.

## 8. Reporting

- One results table per system with every metric in section 3, with CIs.
- The H1-H3 table: difference, 95% CI, permutation p, Holm-adjusted p, McNemar p.
- Everything that was run on the test split is reported. Nothing is dropped.
- Artefacts: `results/test_results.json` (hash of the freeze it used), the manifest hash,
  `models/freeze.json`, and the exact commit.

## 9. Threats to validity

- **Small test set.** Covered by section 5.1. Results speak to large effects only.
- **Written, not logged, requests.** The data reflect what writers think users type.
  External validity to real traffic is untested.
- **Few annotators, one region.** Agreement is measured, but dialect coverage is narrow.
- **Rules written by people who know the contract.** S1 may be favoured on in-scope phrasing.
  The robustness suite partly probes this.
- **Shared extractor.** Its errors cap every system equally (see the oracle gap). They do not
  bias H1-H3.

## 10. Deviations log

| Date | Change | Reason | Before/after test opened |
|---|---|---|---|
| 2026-10-08 | Added the confidence-gate analysis (section 6) and contract v1.1 `confirm` target | Avoid executing on uncertain intents; report the safety/helpfulness trade-off | Before: no native-speaker data and no test split exist yet; H1-H3 unchanged |
| 2026-10-09 | S4 scoring ablation: summed vs length-normalised (mean per token) verbalizer scores vs one-token number answers (`llm.LLMIntent(length_norm=..., answer_format=...)`), each raw and calibrated, on dev, native Egyptian and MASSIVE | On the draft dry run the LLMs took most out-of-scope requests as in scope, and `غير مدعوم` has more tokens than `منبه` in the tokenizers checked, so the summed score may favour the in-scope labels | Before: run on the draft data only. The primary S4 number stays the summed calibrated score unless a change is decided and logged here before the native-speaker data exist |
| 2026-10-12 | **Dataset split** (`data/splits/manifest.json`, `940beba352ec`): test = 120 author-written items (50/50/20, all six unsupported reasons); 8 extra out-of-scope requests (`other_tool`, `other_person`) were written before the split to cover reasons missing from the first 150; 19 A/B versions that were not minimal pairs were unpaired by rule, texts unchanged | A version in which 18 request texts had been edited after labelling (to make A and B use the same tool) was discarded: test items must be the text as written (`data/raw/private/`, not committed) | Before any system was run on the split. The second-annotator agreement check is pending; if adjudication changes labels, the split is rebuilt and logged here before the test run |
| 2026-10-11 | **Primary S4 method for the test run: `raw` → `prior`. Final: no further change.** | *A priori:* `prior` takes the label bias from real in-domain requests, so it does not inherit contextual calibration's failure on an abstain label. *Diagnostics* (`docs/dry_run_draft.md`; 3 models × 3 exemplar sets, paired): on the 192 native out-of-scope items `prior` abstained wherever `raw` did, in all 9 cells, and more often (significant after Holm for Nile-Chat in 3/3 sets, +31 points on average, and Jais 2 in 2/3; Qwen3 at ceiling); no dev STS difference was significant (40 items). The empty-input distribution confirms the calibration failure: Jais 2 puts 0.99 on `unsupported` | Before the test split exists. **Disclosure:** the second change of the S4 primary method, chosen with the Egyptian and draft-dev results, so only the locked author-written test split is unbiased evidence for it; `raw`, `calibrated` and `generate` are reported beside it |
| 2026-10-10 | **Test data written by the project author** (section 4.2), a native Egyptian Arabic speaker: 120 core requests from the wave-1 cards (`source: author`) instead of contributor data; H1-H3 tested on that split once complete | No contributor data could be collected in time; author-written items are allowed by the annotation guide | Before: the test split does not exist yet. One writer, who knows the system: stated as threats in 4.2 |
| 2026-10-10 | **Native-speaker dataset not collected; H1-H3 not tested.** Evaluation moves to section 4.1: drafts for development only; native evaluation on the human-written `Alhasan/egyptian-nlu` test split (intent level) plus a paired Saudi/MSA → Egyptian dialect gap | Recruiting native contributors was not possible in the time available; a form export submitted as native data was found to repeat the assistant-generated set (97% identical texts) and was excluded (`data/raw/private/`, not committed) | No native test split exists, so no locked test run is possible. The external native set had been inspected before this decision (stated in 4.1) |
| 2026-10-10 | **Primary S4 method: `calibrated` → `raw`** (summed, word verbalizers). Added `prior` (label prior from 60 unlabeled train texts) and three exemplar sets as secondary analyses; S7 over seeds 0-2 (mean ± SD) | *A priori:* contextual calibration assumes a content-free input deserves a uniform label distribution; with an abstain label (`unsupported`) an empty input is genuinely out of scope, so the assumption fails for this label set. *Observed* (draft dry run, `docs/dry_run_draft.md`): calibration lowered native Egyptian out-of-scope recall in 9 of 9 model × variant cells and MASSIVE F1 in 9 of 9, and raised dev STS once by one item; length normalisation and one-token answers did not help consistently. S7 dev STS varied 0.93 → 1.00 between two runs with the same seed | Before: no native-speaker data or test split exist. **Disclosure:** the choice was informed by the Egyptian (`Alhasan/egyptian-nlu`) and MASSIVE results, so for S4 these are no longer untouched out-of-distribution evidence; the locked native test split is. `calibrated` and `generate` are still reported |

## References

Berg-Kirkpatrick, Burkett & Klein (2012), *An Empirical Investigation of Statistical
Significance in NLP*. Dror, Baumer, Shlomov & Reichart (2018), *The Hitchhiker's Guide to
Testing Statistical Significance in NLP*. Noreen (1989), *Computer-Intensive Methods for
Testing Hypotheses*. Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*.
Ribeiro, Wu, Guestrin & Singh (2020), *Beyond Accuracy: Behavioral Testing of NLP Models with
CheckList*. Gardner et al. (2020), *Evaluating Models' Local Decision Boundaries via Contrast
Sets*. Guo, Pleiss, Sun & Weinberger (2017), *On Calibration of Modern Neural Networks*.
Geifman & El-Yaniv (2017), *Selective Classification for Deep Neural Networks*. Mosbach, Andriushchenko & Klakow (2021), *On the Stability of Fine-tuning BERT*.
Reimers & Gurevych (2017), *Reporting Score Distributions Makes a Difference*. Zhao, Wallace, Feng, Klein & Singh (2021), *Calibrate Before Use*. Zhou, Wan, Proleev, Mincu, Chen, Heller & Roy (2024), *Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering* (ICLR). Wang et al. (2024), *My Answer is C: First-Token Probabilities Do Not Match Text Answers in Instruction-Tuned Language Models*. FitzGerald et al. (2023), *MASSIVE*. Card et al.
(2020), *With Little Power Comes Great Responsibility*. Abdul-Mageed, Elmadany & Nagoudi (2021),
*ARBERT & MARBERT*. Obeid et al. (2020), *CAMeL Tools*.
