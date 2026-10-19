# ATHAR Chat: Results Report

**Safe tool routing for Egyptian Arabic: keyword rules, lexical classifiers and Jais 2 / Jais 3**

Amina Yousef, Applied NLP course project. Final results, 2026-10-19.

> **Status.** All pre-registered evaluations are complete. The locked test split (120 author-written
> requests) was scored once per system:
> - core systems and the oracle, `athar-eval`, 2026-10-18;
> - S4, commits 7a and 7b;
> - S7 and S8, commit 7c+7d.
>
> The error analysis (section 6) was done afterwards. It reads only those finished outputs and is
> labelled post hoc where it goes beyond the protocol. Every number below is read from the files listed
> in section 13.

---

## 1. Summary

**Research question.** All systems share the normalizer, argument extractor, execution policy and
templates. Does lightweight supervised intent classification (S2: word TF-IDF + logistic regression;
S3: + character n-grams) improve **Safe Task Success (STS)** over hand-written keyword rules (S1)?

**Answer on the locked test: no. It significantly worsened it.**

| | Comparison | Difference in STS [95% CI] | Holm p |
|---|---|---|---|
| H1 | S2 vs S1 | **−0.167** [−0.250, −0.084] | **0.002** |
| H2 | S3 vs S1 | **−0.142** [−0.222, −0.061] | **0.005** |
| H3 | S3 vs S2 | +0.025 [0.000, +0.057] | 0.25 |

Main findings:

1. **The classifiers lost to the rules because of a distribution shift we can measure.** In the
   AI-drafted training data, 88% of reminders use «فكرني» and 5% the noun «تذكير». In the
   author-written test, the figures are 50% and 42%. The classifiers learned the drafting style:
   - their intent accuracy is 0.76-0.77, against 0.96 for S1;
   - S2 turns 15 of 62 test reminders into alarms;
   - dev had no noun-form reminders at all, so it could not reveal this.

   One threat qualifies the result: the test author also designed the system and S1's lexicon
   (section 11).
2. **The shared extractor and policy cap every pipeline system at 0.808 (the gold-intent oracle).**
   Its 23 failures are specific and explainable (section 6.3):
   - 13 are a single text-extraction bug: the request verb and connector are left in the reminder
     text, while the time and the task are correct;
   - 7 are clarification-slot errors;
   - 3 are executions of reminders with vague content.

   **No pipeline system executed at a wrong time, or executed an out-of-scope request.**
3. **The best systems approach that ceiling.** These are S7, Jais 2 + QLoRA (0.786 ± 0.024 over three
   frozen seeds; intent accuracy 0.93-0.99), S1 (0.767) and S4 Jais 2 (0.758). Their intervals
   overlap; no pairwise test among them was pre-registered.
4. **End-to-end Jais 3 tool calling (S8, 0.725 ± 0.033) fails in the complementary place.** It
   recovers 12-14 of the 15 items that every prompted LLM fails through the shared core. But it
   handles incomplete requests poorly: it executes 8-9 of 50 with a guessed value, and answers 8-12
   in plain text instead of calling a tool.
5. **A pre-registered secondary hypothesis was confirmed for one model, not the other.** Exemplars
   listed in the user message instead of as chat turns raise Jais-3-7B from 0.550 to 0.733
   (+0.183, Holm p = 0.0002). Jais 2 shows no such effect (−0.025, p = 0.52).
6. **The 40-item, AI-drafted dev set did not predict test.**
   - Every system dropped from dev to test, for example S2 from 0.950 to 0.600.
   - Rankings reversed, both S2 vs S1 and among the LLMs.
   - The protocol's rule that nothing is concluded from dev was necessary.

---

## 2. Task and systems

**Task.** A single-turn Egyptian Arabic request, often code-switched with English, goes to one of three
routes:
- **ACT:** a simulated `create_reminder(datetime, reminder_text)` or `create_alarm(datetime)`, with
  every argument resolved;
- **CLARIFY:** a templated question naming the missing slots (`date`, `time`, `am_pm`, `past`,
  `reminder_text`);
- **ABSTAIN:** a templated decline for out-of-scope requests (deletion, modification, recurrence,
  another person, several actions, other tools).

Contract: [task_contract.md](task_contract.md).

**Safe Task Success (STS)** counts an item as a success only if:
- **ABSTAIN:** the system abstains;
- **CLARIFY:** it asks with the right intent and exactly the gold slot set;
- **ACT:** it executes the gold tool at the gold instant and, for reminders, with the gold reminder text,
  compared exactly after canonicalisation (normalisation and removal of framing words; [annotation
  guide](annotation_guide.md)).

Unsafe execution is any execution on a non-ACT item, plus executions on ACT items that fail.

**Design.** All pipeline systems share the normalizer, the rule-based time and argument extractor, the
execution policy and the templates (`athar.timeexpr`, `athar.extract`, `athar.policy`). They differ only in
the intent recognizer, so H1-H3 isolate intent recognition. S8 is the exception: it does the whole task
itself.

| System | Intent recognizer | Trained on | Role |
|---|---|---|---|
| S1 | keyword rules | none | baseline, H1-H3 |
| S2 | word TF-IDF + logistic regression | train | H1, H3 |
| S3 | S2 + character 2-4-grams | train | H2, H3 |
| S4 | instruction-tuned LLM scoring Arabic label verbalizers: **Jais 2 8B** (proposal), **Jais-3-7B**, **Jais-3-3.7B**, Nile-Chat 4B (Egyptian), Qwen3 8B | 6 exemplars from train | extended, reported separately |
| S7 | Jais 2 8B + QLoRA | train | exploratory |
| S8 | Jais-3-7B end to end through native tool calling | none | exploratory |
| oracle | gold intent + shared core | | ceiling of the shared core |

**S4 scoring.** S4 scores the summed log-probability of each label's Arabic verbalizer (`تذكير`, `منبه`,
`غير مدعوم`) after a fixed chat prompt. **Primary setting:** 6-shot with method `prior`. Label scores are
divided by the mean label distribution the same prompt gives 60 unlabeled train requests, then
renormalised (in the spirit of batch calibration, Zhou et al. 2024). `raw`, contextual calibration
(`calibrated`, Zhao et al. 2021) and greedy `generate` are also reported.

**Jais 2 → Jais 3.** The proposal named Jais 2 8B. Jais 3 was released on 2026-10-15, after the systems were
frozen and before the test split was opened. It was added as an extension (protocol log 2026-10-15).
Jais 2 remains the pre-registered S4 model, and H1-H3, the data and the metrics did not change.

---

## 3. Data

| Set | Size | Written by | Use |
|---|---|---|---|
| train | 298 | 261 AI-drafted + 37 by the author | training, exemplars, prior texts |
| dev | 40 | 39 AI-drafted + 1 by the author | selection, τ, development decisions |
| **test** | **120** (50 ACT / 50 CLARIFY / 20 ABSTAIN) | **the author, a native Egyptian speaker**; no drafts | the locked, one-shot evaluation |
| native Egyptian (`Alhasan/egyptian-nlu`, CC BY 4.0) | 196 (192 out of scope, 4 alarms) | an independent native speaker | external check of out-of-scope recall |

- **Split by scenario family.** Paraphrases and complete/incomplete minimal pairs share a family, and
  the manifest is hash-locked (`940beba352ec`). About 70% of requests are project-management requests,
  the target users.
- **Agreement** ([agreement.md](agreement.md)). A second native annotator independently labelled 60 items,
  **45 of them test items**:
  - intent κ = 0.916 and route κ = 1.000;
  - exact target sets 92.3% (CLARIFY); datetimes and reminder texts 100% (ACT).
  - The 5 disagreements were kept as they are; no gold label was changed.
- **The native contributor collection planned in the proposal did not happen** (protocol log 2026-10-10).
  The test split is written by one person, who also designed the system (section 11).

**Trigger wording in reminders** (share of reminder requests containing the cue):

| Cue | train (203) | dev (20) | test (62) |
|---|---|---|---|
| «فكرني» | 0.88 | 0.95 | 0.50 |
| «ذكرني» | 0.02 | 0.00 | 0.08 |
| the noun «تذكير» (`عايز / حطلي / اعمللي تذكير`) | 0.05 | 0.00 | **0.42** |
| «حطلي / اعمللي / ظبطلي» | 0.02 | 0.00 | 0.26 |

This table is the key to sections 5 and 6.

---

## 4. Evaluation discipline

- **Pre-registration.** [evaluation_protocol.md](evaluation_protocol.md) fixed the hypotheses, metrics,
  statistics and reading rules before training. Every later change is in its dated deviations log, with
  its reason and its timing relative to the test.
  - Twelve entries predate any test commit.
  - The S7 freeze (2026-10-18, 06:07 UTC) was written as the first test commit was being started,
    before any test output existed (08:34 UTC).
  - Its amendment (adapter lookup by content hash) and a correcting entry (2026-10-19) were written
    after the S4 test results had been seen. Neither changes a system, τ or the scoring.
- **Freeze and integrity.** `athar-train fit` hashes the models, configuration, protocol, contract and every
  shared-core source into `models/freeze.json`, and the locked run refuses a changed copy. S7's adapters
  are frozen by SHA-256.
  - S1's lexicon and rules were last changed on 2026-10-07, before any test item was written.
  - The S2/S3 pickles are re-written at each re-freeze. We verified that the models used in the locked
    run give **identical test predictions** to the models frozen before any test commit (git `a26d6b4`).
  - They also give bit-identical probabilities on all 338 train+dev items. The shared-core code has not
    changed since then.
- **Statistics.**
  - 95% percentile intervals from a family-level bootstrap (10,000 resamples).
  - Paired approximate randomization swapping whole families (10,000 permutations).
  - Holm-Bonferroni within each declared family, with exact McNemar as a sensitivity check.
- **GPU test runs.** The notebook refuses dev-only analyses on the test split. The confidence-gate
  threshold τ is taken from dev, never chosen on test.

Changes that shaped the LLM results (full log: protocol section 10):

| Date | Change | Basis |
|---|---|---|
| 2026-10-10 | S4 primary `calibrated` → `raw` | A priori: content-free calibration assumes an empty input deserves a uniform label, but here an empty input is genuinely out of scope. It lowered native out-of-scope recall in 9/9 cells |
| 2026-10-11 | S4 primary `raw` → `prior` (final) | More abstention on native out-of-scope requests (significant after Holm for Nile-Chat 3/3 and Jais 2 2/3 exemplar sets). **Disclosed:** chosen with the native set, which is therefore supporting, not untouched, evidence |
| 2026-10-15 | Jais 3 (S4), S8, precision policy, Jais 2 few-shot role fix | Release of Jais 3; prompts checked against each model's own chat template |
| 2026-10-16 | Secondary S4 analysis: inline exemplars | Dev result; declared with its test before the test was opened |
| 2026-10-17 | S7 optimisation set to the pre-registered S6 recipe; S7 headline `prior` | Run 6 seed instability; no search |
| 2026-10-18 | S7 frozen at the run-6b adapters | The test must score the weights evaluated on dev |

**Researcher degrees of freedom.** The S4 primary method changed twice and S7's optimisation once, all
before any test output and all disclosed. Together they are still analytic flexibility, and a reader
should weigh the S4/S7 results with that in mind. H1-H3 and their systems never changed.

---

## 5. Main result: H1-H3 on the locked test

| System | STS [95% CI] | Intent accuracy | Intent macro-F1 | Unsafe exec. | ECE |
|---|---|---|---|---|---|
| oracle (gold intent) | 0.808 [0.725, 0.885] | 1.000 | 1.000 | 0.133 | |
| **S1 keyword rules** | **0.767** [0.678, 0.847] | 0.958 | 0.936 | 0.133 | |
| S3 TF-IDF + char n-grams | 0.625 [0.538, 0.711] | 0.767 | 0.751 | 0.167 | 0.308 |
| S2 TF-IDF | 0.600 [0.513, 0.686] | 0.758 | 0.738 | 0.200 | 0.264 |

Paired tests (family-level randomization; Holm over H1-H3; McNemar discordant counts as sensitivity):

| | Difference [95% CI] | p | Holm p | McNemar (only A / only B right) |
|---|---|---|---|---|
| H1 S2 vs S1 | −0.167 [−0.250, −0.084] | 0.0007 | **0.002** | 5 / 25 (p = 0.0003) |
| H2 S3 vs S1 | −0.142 [−0.222, −0.061] | 0.0024 | **0.005** | 5 / 22 (p = 0.0015) |
| H3 S3 vs S2 | +0.025 [0.000, +0.057] | 0.25 | 0.25 | 3 / 0 (p = 0.25) |

- **Robust to the reminder-text rule** (post hoc sensitivity). Counting a reminder whose only error is its
  text as a success gives H1 −0.175 (p = 0.0006), H2 −0.158 (p = 0.0013) and H3 +0.017 (p = 0.62). The
  conclusions do not depend on the strict text match.
- **Where the classifiers fail** (intent confusion; rows: gold reminder / alarm / out of scope):

  | System | Reminder row | Alarm row | Out-of-scope row |
  |---|---|---|---|
  | S1 | 62 / 0 / 0 | 0 / 38 / 0 | 3 / 2 / 15 |
  | S2 | 41 / **15** / 6 | 0 / 37 / 1 | 6 / 1 / 13 |
  | S3 | 40 / **13** / 9 | 0 / 37 / 1 | 4 / 1 / 15 |

  S2 sends 15 of 62 reminders to the alarm tool and declines 6. These are mostly noun-form requests such as
  «حطلي تذكير الخميس الجاي الساعة 7 الصبح عشان احضر الستاند اب» and «ذكرني السبت الجاي الساعة 6
  بالليل…». Those are the cues that are rare in training (section 3).
- **Subgroups** (descriptive). The classifiers do *better* on code-switched items (S2 0.708, S3 0.750)
  than on Arabic-only ones (0.573, 0.594). S1 is flat (0.750 vs 0.771).

**Reading.**
- With 298 mostly drafted training items, a lexical classifier learned the drafting style rather than the
  task, and did not transfer to a native speaker's phrasing.
- S1 transfers well. However, its lexicon and the test were written by the same person, which the
  protocol named in advance as a threat (section 11).
- The independent native set supports S1's robustness only for out-of-scope requests: recall 0.990 vs
  0.870 for S2 (section 8.3).
- The result is therefore best read as "a small classifier trained on synthetic drafts did not beat
  rules on real requests". It is not a general claim that rules beat classifiers.

---

## 6. Error analysis (post hoc)

### 6.1 Method

The shared pipeline is deterministic given an intent. `scripts/error_analysis.py` therefore rebuilds
every pipeline system's full test output (tool, instant, reminder text, slots) from the intent it predicted
in the locked run. It stops unless the rebuild reproduces every stored item-level result. **All 17 runs
(S1-S3, oracle, 10 S4, 3 S7) are reproduced exactly.** Outputs: `results/error_analysis/`
(`summary.csv`, `failures.csv`).

### 6.2 Failure causes

| System | STS | Text-only mismatch | Should-not-have-executed (wrong tool / incomplete request executed) | Other failures |
|---|---|---|---|---|
| oracle | 0.808 | 13 | 3 (0 / 3) | 7 |
| S1 | 0.767 | 13 | 3 (0 / 3) | 12 |
| S2 | 0.600 | 12 | 12 (**9** / 3) | 24 |
| S3 | 0.625 | 11 | 9 (**7** / 2) | 25 |
| S4 Jais 2 | 0.758 | 12 | 5 (2 / 3) | 12 |
| S4 Jais-3-3.7B | 0.742 | 10 | 3 (0 / 3) | 18 |
| S4 Nile-Chat 4B | 0.742 | 13 | 3 (0 / 3) | 15 |
| S4 Qwen3 8B | 0.700 | 13 | 5 (2 / 3) | 18 |
| S4 Jais-3-7B | 0.550 | 10 | 2 (0 / 2) | 42 (34 in-scope requests declined) |
| S7 seeds 0 / 1 / 2 | 0.758 / 0.800 / 0.800 | 13 / 13 / 13 | 3 / 3 / 3 | 13 / 8 / 8 |

- **No pipeline system executed at a wrong instant, and none executed an out-of-scope request.** The
  "unsafe execution" rate of 0.13-0.20 is mostly text-only mismatches, which run at the right time on the
  right task.
- Genuinely wrong executions are 3 for the oracle and most LLMs, 9-12 for S2/S3. Wrong-tool errors (an
  alarm instead of a reminder) occur almost only with the classifiers.

### 6.3 The shared core's 23 failures (the oracle)

| Cause | n | Example | What happens |
|---|---|---|---|
| Request verb and connector left in the reminder text | 13 | «عايز تذكير بكرة الساعة 5 العصر علشان احدث سجل المخاطر» | gold «احدث سجل المخاطر», extracted «عايز علشان احدث سجل المخاطر». When the time expression sits between «تذكير» and the task, the framing verb (`عايز`, `اعمللي`) and the connector (`علشان`, `عشان`) survive, against the contract's canonicalisation rules. **All 13 are this one construction**, which is absent from the drafts |
| A time word inside a purpose clause | 3 | «ظبطلي منبه الساعة 7 عشان اصحى لصلاة الفجر» | «الفجر» is read as a second time expression, so a complete request gets a time question |
| Wrong clarification slots | 4 | «اعمللي تذكير» | asks only for `time`; gold also asks for `reminder_text` (leftover words such as «كده», «لو ينفع» are taken as content) |
| A vague reminder executed | 3 | «فكرني بكرة الساعة 8 الصبح بحاجة محتاجه ضروري» | executed at the right time with text «بحاجة محتاجه ضروري»; gold asks for the content |

Four of these items are double-annotated (c008-b, c053-b, c102-b, c110-a), and the second annotator agreed
with gold on all four. Two items without a second annotation have gold labels a reader may question. Both
are kept, because labels cannot change after the test:
- **c017-b**, «اعمللي تذكير بعد بكرة 7 بالليل»: gold marks `time` as missing, though «7 بالليل» is a usable
  time.
- **c173-a**: gold marks `reminder_text` as missing, though the request names an action.

**Implication.**
- The extractor was developed on drafted text, scored 1.000 on dev, and fails on one frequent native
  construction (noun request + time + purpose clause).
- A single canonicalisation fix for that construction would remove 13 of the oracle's 23 failures. The
  ceiling would rise from 0.808 to 0.917 (the lenient-text figure in section 5).
- That fix belongs to future work. Changing the extractor now would invalidate the locked comparison.

---

## 7. LLM results on the locked test

All test commits ran on Kaggle 2× T4 with pinned revisions and the frozen manifest. Each model ran at its
dev precision. Every report covers exactly the 120 test items, and S7's adapter hashes match the frozen
ones (all verified).

### 7.1 S4 intent recognizers

| Model (precision) | **Primary 6-shot `prior`** [95% CI] | 0-shot `prior` | 6-shot inline (secondary) | Intent acc. | ECE |
|---|---|---|---|---|---|
| Jais 2 8B (nf4, fp32 compute) | **0.758** [0.67, 0.84] | 0.683 | 0.733 | 0.917 | 0.114 |
| Jais-3-3.7B (fp32) | 0.742 [0.65, 0.83] | 0.667 | **0.758** | 0.908 | **0.030** |
| Nile-Chat 4B (fp32) | 0.742 [0.65, 0.83] | 0.575 | 0.725 | 0.925 | 0.060 |
| Qwen3 8B (fp16) | 0.700 [0.61, 0.79] | 0.642 | 0.708 | 0.883 | 0.108 |
| Jais-3-7B (fp16) | 0.550 [0.46, 0.65] | 0.642 | 0.733 | 0.683 | 0.226 |

- **Jais 2, Jais-3-3.7B and Nile-Chat show no detectable difference** on 120 items. No pairwise LLM test
  was pre-registered for test.
- **Few-shot helps on test** (6-shot vs 0-shot `prior`): +0.075 / +0.075 / +0.167 / +0.058. The exception
  is Jais-3-7B in chat-turn format, which declines 34 in-scope requests.
- **Confidence gate (τ from dev).** For Jais 2 (τ = 0.7) it lowers unsafe execution from 0.142 to 0.083 at
  a cost in STS (0.758 → 0.700). For the other models the dev τ is 0, so they are unchanged.

### 7.2 Pre-registered secondary test: inline vs chat-turn exemplars

Declared 2026-10-16 before the test was opened. Family-level paired test, Holm over the two confirmatory
models:

| Model | Inline | Chat turns | Difference [95% CI] | Discordant (inline / turns) | p | Holm p |
|---|---|---|---|---|---|---|
| **Jais-3-7B** (confirmatory) | 0.733 | 0.550 | **+0.183** [0.113, 0.258] | 24 / 2 | 0.0001 | **0.0002** |
| Jais 2 8B (confirmatory) | 0.733 | 0.758 | −0.025 [−0.078, 0.024] | 3 / 6 | 0.516 | 0.516 |
| Jais-3-3.7B (descriptive) | 0.758 | 0.742 | +0.017 [−0.024, 0.059] | 4 / 2 | 0.695 | |
| Nile-Chat 4B (descriptive) | 0.725 | 0.742 | −0.017 [−0.063, 0.027] | 3 / 5 | 0.726 | |
| Qwen3 8B (descriptive) | 0.708 | 0.700 | +0.008 [−0.034, 0.051] | 4 / 3 | 1.000 | |

The format effect is specific to Jais-3-7B and replicates from dev to held-out data. For Jais 2, the dev
advantage (+17 points) did not replicate.

### 7.3 S7: Jais 2 8B + QLoRA (frozen run-6b adapters)

| Scoring | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---|---|---|---|
| **`prior` (headline)** | 0.758 [0.68, 0.84] | 0.800 [0.72, 0.88] | 0.800 [0.72, 0.88] | **0.786 ± 0.024** |
| `raw` | 0.767 | 0.800 | 0.775 | 0.781 ± 0.017 |
| `calibrated` | 0.725 | 0.658 | 0.700 | 0.694 ± 0.034 |

- **Intent accuracy is 0.925 / 0.983 / 0.992.** Fine-tuning a pretrained LLM on the same 298 mostly
  drafted items transfers to the author's phrasing; the lexical classifiers do not (section 5).
  Remaining failures are the shared core's.
- **Fine-tuned vs zero-shot Jais 2** (declared 2026-10-17 as descriptive), against 0-shot `prior` (0.683):
  - seed 0: +0.075 [0.000, 0.153], p = 0.077;
  - seeds 1 and 2: +0.117 [0.063, 0.183], p = 0.0003 each.
- **Reproduction check.** The dev rescoring of the frozen adapters reproduced run 6b in 6 of 9 cells; the
  other 3 differ by one item (GPU nondeterminism).

### 7.4 S8: Jais-3-7B end to end through native tool calling

fp16, reasoning effort high, xml tool calls, sampling T 0.7 / top-p 0.95 / top-k 20, seeds 0-2.

| | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---|---|---|---|
| Safe Task Success | 0.758 [0.69, 0.83] | 0.725 [0.65, 0.80] | 0.692 [0.61, 0.77] | **0.725 ± 0.033** |
| intent macro-F1 | 0.941 | 0.930 | 0.882 | 0.918 ± 0.032 |
| unsafe execution | 0.100 | 0.108 | 0.108 | 0.106 ± 0.005 |
| no tool call (plain-text answer) | 11 | 12 | 16 | |
| success: ACT / CLARIFY / ABSTAIN | 0.92 / 0.56 / 0.85 | 0.90 / 0.52 / 0.80 | 0.90 / 0.50 / 0.65 | |

- **Strength: complete requests.** S8 recovers 12-14 of the 15 items that all five S4 models fail through
  the shared core. It almost never asks unnecessarily (0.00-0.02).
- **Weakness: incomplete requests.** 22-25 of the 50 CLARIFY items fail:
  - 8-12 are plain-text answers instead of `ask_clarification`;
  - 8-9 are executed with a guessed value;
  - 4-7 ask for the wrong slots.
- **On complete requests,** 4-5 executions per seed fail, of which 0-2 are text-only; the rest are a wrong
  instant or a wrong tool. It never executed an out-of-scope request.

### 7.5 Two failure profiles

| | Shared pipeline (oracle) | End-to-end Jais 3 (S8) |
|---|---|---|
| ACT success | 0.68 | **0.90-0.92** |
| CLARIFY success | **0.86** | 0.50-0.56 |
| Incomplete requests executed | 3 | 8-9 |
| Executions at a wrong instant or with the wrong tool | **0** | 2-4 |
| Out-of-scope requests executed | 0 | 0 |

The rule-based core is conservative but brittle on native phrasing. The LLM extracts arguments well but
does not follow the clarification protocol. A hybrid is the natural next step: LLM argument extraction,
with the deterministic policy deciding when to execute and when to ask. That is future work, not a
result here.

---

## 8. Development results (40 dev items, 39 AI-drafted)

Dev drove the development decisions. Section 9 shows it did not predict test.

### 8.1 Core systems

S1 0.800 [0.59, 0.95]; S2 **0.950** [0.88, 1.00]; S3 0.925 [0.84, 1.00]; oracle 1.000. No difference was
significant (S2 vs S1 +0.150, Holm p = 0.80).

### 8.2 S4

| Model | Primary (6-shot `prior`) | Best other setting |
|---|---|---|
| Jais-3-3.7B | 0.950 | 0.950 (several) |
| Nile-Chat 4B | 0.900 | 0.875 |
| Qwen3 8B | 0.850 | 0.875 (6-shot `raw`) |
| Jais-3-7B | 0.825 | 0.975 (0-shot `calibrated`) |
| Jais 2 8B | 0.725 | 0.925 (0-shot `prior`) |

### 8.3 Native Egyptian out-of-scope recall (192 requests, independent native writer)

| System | Recall [95% Wilson CI] |
|---|---|
| Jais-3-7B | **1.000** [0.980, 1.000] |
| S7 (seeds 0/1/2) | 1.000 / 0.990 / 0.990 |
| S1 | 0.990 [0.963, 0.997] |
| Qwen3 8B | 0.974 [0.940, 0.989] |
| Jais 2 8B / Nile-Chat 4B | 0.964 [0.927, 0.982] |
| Jais-3-3.7B | 0.958 [0.920, 0.979] |
| S2 / S3 | 0.870 [0.815, 0.910] / 0.865 [0.809, 0.906] |

- **Paired McNemar.** Jais 2 → Jais-3-7B 0 / 7 discordant (p = 0.016); Nile-Chat → Jais-3-3.7B 4 / 3
  (p = 1.0).
- **No dialect gap.** S1-S3 on the same requests in their MSA originals: 0.979 / 0.865 / 0.859.
- **Caveats.** Jais 2 ran in 4-bit, the native set informed the S4 method choice, and the p-values are not
  multiplicity-adjusted.

### 8.4 Few-shot format, S7 training, S8 (dev)

- **Exemplar format** (run 4, three exemplar sets). Inline exemplars beat chat turns in every set: Jais 2
  0.875 vs 0.708, Jais-3-7B 0.900 vs 0.825. Fixing Jais 2's answer role (`ai`) made no difference (11 vs 12
  discordant, p = 1.0).
- **S7 training stability.**
  - With one optimizer step per item at a constant LR, `raw` STS by seed was 0.925 / 0.325 / 0.650. That
    was label-bias collapse, not lost discrimination: calibration on the same weights recovered 0.875.
  - With the pre-registered S6 recipe it became 0.950 / 0.925 / 0.925. The recipe is batch 16, 10%
    warm-up and linear decay, clipping at 1.0 and a class-balanced loss, with answers ending in the
    template's `<|eot_id|>`.
  - The components changed together, so the fix is not attributed to any single one.
- **S8 on dev.** STS 0.717 ± 0.014 and intent F1 0.924. 18 of 120 answers were plain-text replies instead
  of tool calls.

---

## 9. Dev vs test

| System | Dev STS | Test STS |
|---|---|---|
| S2 | 0.950 | 0.600 |
| S3 | 0.925 | 0.625 |
| S1 | 0.800 | 0.767 |
| oracle | 1.000 | 0.808 |
| S7 (`prior`, 3 seeds) | 0.950 ± 0.000 | 0.786 ± 0.024 |
| S4 Jais 2 | 0.725 | 0.758 |
| S4 Jais-3-3.7B | 0.950 | 0.742 |
| S4 Nile-Chat | 0.900 | 0.742 |
| S8 (3 seeds) | 0.717 ± 0.014 | 0.725 ± 0.033 |
| S4 Qwen3 | 0.850 | 0.700 |
| S4 Jais-3-7B | 0.825 | 0.550 |

The systems that depend most on surface patterns fell most:
- S2 and S3 fell by 0.30-0.35;
- the drafts-developed extractor's oracle fell from 1.000 to 0.808;
- S1 and the S4 Jais 2 model were nearly stable.

Dev had no noun-form reminders (section 3), so a dev-selected system was selected on the wrong
distribution.

---

## 10. Cost

| System | Training | Inference |
|---|---|---|
| S1 | none | 0.011 ms per item (CPU) |
| S2 | 0.08 s CPU (12-fit search: 0.7 s) | 0.054 ms per item (CPU) |
| S3 | 0.30 s CPU (36-fit search: 18 s) | 0.214 ms per item (CPU) |
| S4 (5 models) | none | 40-182 min per model per test commit on 2× T4. Each commit scores 9 settings, plus the 9-perturbation robustness suite, on 120 items |
| S7 | about 60 min per seed on 2× T4 (run 6b) | 4-bit Jais 2 forward passes, as S4 |
| S8 | none | 144 min for 3 seeds × 120 items on 2× T4, about 24 s per request (reasoning + tool call) |

The GPU times are wall-clock for whole notebook commits, not tuned latency benchmarks. At a similar STS,
S1 needs no training and runs in microseconds.

---

## 11. Threats to validity

1. **One test author, who also designed the system and S1's lexicon.** Test measures one native
   speaker's phrasing. That the rules beat the classifiers (H1, H2) may partly reflect the same person
   writing rules and test.
   - Mitigating: S1's rules were last changed on 2026-10-07, before any test item was written (from
     2026-10-10).
   - Mitigating: the independent native set shows S1's strength on out-of-scope requests (section 8.3).
   - Not mitigated for in-scope requests, where the native set has almost no items.
2. **Shared writer in train and test.** 37 of the 298 training items are also by the author. Split by
   family, this is not leakage, but trained systems saw some of the test writer's style.
3. **Synthetic-to-real shift.** Training and dev are 88% and 98% AI-drafted. Sections 3, 5 and 9 show this
   shaped the results.
4. **Small samples.** On 120 test items, STS intervals are about ±0.09, and differences under about 0.1
   between systems are not resolvable. H1 and H2 are clear of this; the LLM rankings are not.
5. **Strict text matching.** STS requires an exact canonical reminder text, which penalises cosmetic
   differences. A post hoc lenient variant leaves H1-H3 unchanged (section 5).
6. **Possible gold errors** in two test items without a second annotation (section 6.3), kept unchanged.
7. **Precision confound.** On T4, Jais 2 (S4, S7) ran in 4-bit; the other models ran in fp16/fp32.
8. **Analytic flexibility in S4 and S7**, disclosed in section 4. H1-H3 are unaffected.
9. **S5/S6** (sentence embeddings, fine-tuned MARBERTv2) are implemented but were not fit on the final
   split, and are not reported.

---

## 12. Notes for Jais practitioners

Each point was verified in this project, against the pinned revisions and the models' own chat
templates.

- **Jais 3 chat template.**
  - Every assistant turn needs a thinking field; the template raises an error without one.
  - The generation prompt opens a reasoning block (`<ifm|think>`).
  - To score a label directly, we send few-shot answers with an empty `reasoning_content` and append
    `</ifm|think>` to the prompt. That reproduces exactly how the template writes an answer with no
    reasoning (checked by `llm.answer_consistent` on the real template, which is a test fixture).
- **Jais 3 loading.** It needs `trust_remote_code` (`K2HorizonForCausalLM`). Pin revisions: a Hub history
  rewrite on 2026-10-16 made earlier commit hashes unresolvable, though every file we checked was
  byte-identical.
- **Jais 2 chat template.** Roles are written literally, and the answer opens under
  `<|start_header_id|>ai<|end_header_id|>`. Few-shot answers should use the role `ai`, though on our data
  this made no measurable difference. The template ends a turn with `<|eot_id|>`; train fine-tuned answers
  to end with it.
- **Tokenizers.** Jais 2 and Jais 3 encode all three Arabic labels in 2 tokens each, so summed verbalizer
  scores carry no length bias for them. Nile-Chat (3/2/4) and Qwen3 (4/2/4) do not have this property.
- **Precision on T4 (no native bf16).** Jais 2 8B overflowed in fp16 and ran in 4-bit NF4 with fp32
  compute. Jais-3-7B ran in fp16 without overflow.
- **Few-shot format.** Jais-3-7B is strongly sensitive to exemplars given as chat turns: it over-declines
  in-scope requests, and inline exemplars add +18 STS points on test. Report the exemplar format with any
  few-shot Jais-3-7B result.
- **Out-of-scope behaviour.** Jais-3-7B declined 192/192 native out-of-scope Egyptian requests.
- **Tool calling (Jais 3).**
  - Its argument extraction from real Egyptian requests beats our rule-based extractor on complete
    requests.
  - In 9-15% of answers it replies in plain text where a tool call is required, and it may execute an
    incomplete request with a guessed value.
  - A strict tool contract needs constrained decoding, or a validator that turns a plain-text question
    into a clarification call.
- **QLoRA on Jais 2.** One optimizer step per example at a constant LR made the label bias depend on the
  seed. Batched, scheduled, clipped, class-balanced updates removed this. With that recipe, 298 mostly
  drafted examples gave test intent accuracy of 0.93-0.99.

---

## 13. Reproducibility

| Item | Value |
|---|---|
| Split manifest | `data/splits/manifest.json`, `940beba352ec…` |
| Freeze used by the locked core run | `d9bc40cf9473…` (`results/test_results.json`). The current `models/freeze.json` gives identical predictions (section 4) |
| Eval config | `configs/eval.yaml`, SHA-256 `dfbd8398c8c3…` (recorded in every report) |
| Model revisions | Jais-3-7B `94d69f49b4af`, Jais-3-3.7B `3d5754d5db87`, Jais-2-8B-Chat `da0e1639cd92`, Nile-Chat-4B `2a365667659b`, Qwen3-8B `b968826d9c46` |
| S7 adapters | SHA-256 in [jais3.md](jais3.md), section 13 |
| GPU notebook | `notebooks/kaggle_s4_llm.ipynb` (S4 test: 2026-10-17a; S7/S8 test: 2026-10-18b); CPU rehearsal: `scripts/smoke_kaggle_notebook.py [--split test]` |
| Environment | Kaggle 2× T4, torch 2.11.0, transformers 5.19.0, peft 0.20.0, bitsandbytes 0.50.2 |
| Test outputs | `results/test_results.json` (S1-S3, oracle, H1-H3); `results/s4/7a-jais3/`, `7b-others/`, `7cd-s7-s8/`; `results/error_analysis/` |
| Dev outputs | `runs/s4/…`, `runs/ood/core_2026-10-18.json` |
| Commands | `athar-eval --split test --unlock-test` (once); `python scripts/error_analysis.py` |
| Details | [jais3.md](jais3.md) (LLM runs), [evaluation_protocol.md](evaluation_protocol.md) (design, deviations), [data_card.md](data_card.md), [agreement.md](agreement.md) |

### References

Zhao, Wallace, Feng, Klein & Singh (2021), *Calibrate Before Use*. Zhou et al. (2024), *Batch Calibration*.
Dettmers, Pagnoni, Holtzman & Zettlemoyer (2023), *QLoRA*. Mosbach, Andriushchenko & Klakow (2021), *On
the Stability of Fine-tuning BERT*. Ribeiro et al. (2020), *CheckList*. Berg-Kirkpatrick et al. (2012) and
Dror et al. (2018) on significance testing in NLP. Holm (1979). FitzGerald et al. (2023), *MASSIVE*.
Abdellatif, *Egyptian NLU* (Hugging Face, CC BY 4.0).
