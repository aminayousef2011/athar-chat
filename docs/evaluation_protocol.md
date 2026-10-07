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

## 2. Systems

| | System | Learns from | Status |
|---|---|---|---|
| S1 | keyword rules (`athar.intent.rules`), lists written from the contract scope | nothing (lists may be extended after reading train only) | core |
| S2 | word TF-IDF (1-2 g) + logistic regression | train | core |
| S3 | S2 + character `char_wb` n-grams | train | core |
| Oracle | gold intent + the shared pipeline | gold labels | ceiling: isolates extractor/policy error |
| S4 | instruction-tuned Arabic LLM, zero-shot intent only (e.g. Jais 2, Nile-Chat) | nothing | extended, reported separately |
| S5 | frozen multilingual or Arabic sentence embeddings + logistic regression | train | extended, reported separately |
| S6 | fine-tuned dialect encoder (MARBERTv2), 5 seeds | train | extended, reported separately |

Every system predicts **only** the intent label (`create_reminder`, `create_alarm`,
`unsupported`). The route always comes from the shared deterministic policy. The extended
systems answer the question a reviewer will ask ("would a pretrained model do better?").
They are outside H1-H3 and get no significance tests against the core systems.

**Model selection (dev only).** S2/S3: grid over C ∈ {0.1, 0.3, 1, 3, 10, 30}, word n-gram
upper bound ∈ {1, 2}, (S3) character n-gram range ∈ {2-4, 2-5, 3-5}. The winner has the highest
dev intent macro-F1; ties go to the simpler model (smaller C, then shorter n-grams). It is then
refitted on train only. The vectorizers never see dev or test text during fitting. S6: learning
rate, epochs and seeds are fixed in its config before training. The reported value is the mean
and standard deviation over 5 seeds.

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
| intent macro-F1 | over the three intent labels |
| route macro-F1 | over ACT / CLARIFY / ABSTAIN |
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
  gives STS and the invariance rate (unchanged output). The minimal pairs serve as the
  directional tests.
- **Calibration and selective prediction** (S2, S3, S5, S6). ECE (10 equal-width bins),
  multi-class Brier score, and the area under the risk-coverage curve, for the intent
  probabilities.
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
| _none yet_ | | | |

## References

Berg-Kirkpatrick, Burkett & Klein (2012), *An Empirical Investigation of Statistical
Significance in NLP*. Dror, Baumer, Shlomov & Reichart (2018), *The Hitchhiker's Guide to
Testing Statistical Significance in NLP*. Noreen (1989), *Computer-Intensive Methods for
Testing Hypotheses*. Holm (1979), *A Simple Sequentially Rejective Multiple Test Procedure*.
Ribeiro, Wu, Guestrin & Singh (2020), *Beyond Accuracy: Behavioral Testing of NLP Models with
CheckList*. Gardner et al. (2020), *Evaluating Models' Local Decision Boundaries via Contrast
Sets*. Guo, Pleiss, Sun & Weinberger (2017), *On Calibration of Modern Neural Networks*.
Geifman & El-Yaniv (2017), *Selective Classification for Deep Neural Networks*. Card et al.
(2020), *With Little Power Comes Great Responsibility*. Abdul-Mageed, Elmadany & Nagoudi (2021),
*ARBERT & MARBERT*. Obeid et al. (2020), *CAMeL Tools*.
