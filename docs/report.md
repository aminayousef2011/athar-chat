# ATHAR Chat: Results Report

**Safe tool routing for Egyptian Arabic, with Jais 2 / Jais 3 baselines**

Amina Yousef, Applied NLP course project. Status as of **2026-10-19**.

> **Status of the evidence**
>
> | Part | State |
> |---|---|
> | Data, agreement study, pre-registered protocol, frozen systems | final |
> | Development results (dev split, native Egyptian set, ablations) | final |
> | **Locked test, S4 LLM intent recognizers (5 models)** | **final** (commits 7a, 7b) |
> | **Locked test, S7 (Jais 2 + QLoRA, frozen adapters) and S8 (Jais 3 tool calling)** | **final** (commit 7c+7d) |
> | Locked test, core systems S1-S3: **H1-H3** and the oracle | **pending** (one local run) |
>
> The primary research question (H1-H3) is **not answered yet**. Every number below is read
> from the run outputs cited next to it. Nothing is estimated or rounded beyond three decimals.

---

## 1. Summary of findings

1. **On the 120 author-written test items, the best LLM system is Jais 2 + QLoRA (S7): Safe Task
   Success (STS) 0.786 ± 0.024 over three frozen seeds.** Its test intent accuracy is 0.93-0.99. Among the
   prompted intent recognizers (S4), Jais 2 8B (0.758), Jais-3-3.7B (0.742) and Nile-Chat 4B (0.742) are
   statistically indistinguishable. Jais-3-7B with the pre-registered prompt is the outlier at 0.550
   (sections 6.1, 6.5).
2. **The bottleneck on real requests is the shared extraction/policy layer, not intent.** Intent accuracy
   is 0.88-0.93 for four of the five models, yet only 0.79-0.83 of correctly classified items succeed.
   Between 20% and 30% of the 50 ACT items are executed with wrong arguments. Fifteen test items fail
   for **all five** models although all five recognised the intent. This bounds
   the gold-intent oracle at **≤ 0.875** before it is even run (section 6.3). Reminders are the weak path:
   reminder-item STS is 0.26-0.68, alarms 0.87-0.92. S7, with near-perfect intent, also fails all 15.
3. **Jais 3 end to end (S8) fails in the complementary place.** Doing its own extraction, it solves
   12-14 of those 15 items, and its ACT success is 0.90-0.92 against 0.40-0.68 for every S4 model. Its
   weakness is incomplete requests: CLARIFY success is 0.50-0.56, it acts on 8-9 of 50 incomplete
   requests, and it answers 8-12 in plain text instead of calling a tool. Overall STS is 0.725 ± 0.033
   (sections 6.6, 6.7).
4. **Development numbers on drafted data did not predict test performance.** Dev (40 items, 39
   AI-drafted) ranked Jais 2 last (0.725) and Jais-3-3.7B first (0.950). On author-written test, Jais 2 is
   first. Unsafe execution rose from 0-0.025 on dev to 0.10-0.15 on test (section 6.4).
5. **A pre-registered secondary hypothesis was confirmed for one model and not the other.** Listing
   the few-shot exemplars in the user message instead of as chat turns raises Jais-3-7B's test STS from
   0.550 to 0.733 (+18.3 points, 24 vs 2 discordant items, Holm-adjusted p = 0.0002). The same change
   does **not** help Jais 2 (-2.5 points, p = 0.52), whose dev effect did not replicate (section 6.2).
6. **Jais 3 is the safest model on native out-of-scope Egyptian requests.** Jais-3-7B declined all 192
   out-of-scope requests written by an independent native speaker, significantly more than Jais 2
   (7 vs 0 discordant, McNemar p = 0.016) (section 5.3). The TF-IDF classifiers trained on our data
   decline only 0.865-0.870.
7. **Size-matched, general-Arabic Jais-3-3.7B equals Egyptian Nile-Chat 4B**, on test STS (0.742 vs
   0.742), on native out-of-scope recall (0.958 vs 0.964, p = 1.0) and in calibration (test ECE 0.030 vs
   0.060).
8. **Jais 3 follows the tool contract only partly.** On dev and test alike, intent F1 is 0.92, but 9-15%
   of answers are plain-text replies where a tool call was required (sections 5.6, 6.6).
9. **QLoRA fine-tuning of Jais 2 (S7) was unstable until the optimisation was fixed** with the protocol's
   own pre-registered recipe: seed SD in dev `raw` STS fell from 0.300 to 0.014. Fine-tuning showed no
   gain on dev, but on test it improves on zero-shot Jais 2 by +0.075 to +0.117 STS in every seed. That
   comparison was declared as descriptive (section 6.5).

---

## 2. Task and systems

**Task.** One single-turn Egyptian Arabic request, often code-switched with English, is routed to one
of three outcomes: **ACT**, a simulated `create_reminder(datetime, reminder_text)` or
`create_alarm(datetime)` with every argument resolved; **CLARIFY**, a templated question naming the
missing slots (`date`, `time`, `am_pm`, `past`, `reminder_text`); or **ABSTAIN**, a templated decline
for out-of-scope requests (deletion, modification, recurrence, another person, several actions, other
tools). Contract: [task_contract.md](task_contract.md).

**Primary metric: Safe Task Success (STS).** An item succeeds only if the route is right *and*, for
ACT, the tool, the resolved instant and the canonical reminder text are right, or, for CLARIFY, the
target set is exact. Unsafe execution counts executions on items that should not be executed, or that
are executed with wrong arguments.

**Design.** Every system shares the normalizer, argument/time extractor, execution policy and templates
(`athar.timeexpr`, `athar.policy`). Systems differ only in the intent recognizer. The exception is
S8, which does the whole task itself.

| System | Intent recognizer | Trained on | Role |
|---|---|---|---|
| S1 | keyword rules | none | baseline, H1-H3 |
| S2 | word TF-IDF + logistic regression | train | H1, H3 |
| S3 | S2 + character n-grams | train | H2, H3 |
| S4 | instruction-tuned LLM, verbalizer scoring: **Jais 2 8B** (proposal), **Jais-3-7B**, **Jais-3-3.7B**, Nile-Chat 4B (Egyptian), Qwen3 8B | few-shot exemplars from train | extended, reported separately |
| S7 | Jais 2 8B + QLoRA | train | exploratory |
| S8 | Jais-3-7B end to end through native tool calling | none | exploratory |
| oracle | gold intent + shared core | | extraction/policy ceiling |

S4 scores the log-probability of each label's Arabic verbalizer (`تذكير`, `منبه`, `غير مدعوم`) after a
fixed chat prompt. **Primary S4 setting:** 6-shot (two train exemplars per label), method `prior`. The
label scores are divided by the mean label distribution the same prompt gives 60 unlabeled train
requests, then renormalised (in the spirit of batch calibration, Zhou et al. 2024). `raw`, contextual
calibration (`calibrated`, Zhao et al. 2021) and greedy `generate` are also reported.

**From Jais 2 to Jais 3.** The proposal named Jais 2 8B. Jais 3 was released on 2026-10-15, after the
systems were frozen and before the test split was opened. It was added as an extension (protocol
log 2026-10-15). Jais 2 remains the pre-registered S4 model, and H1-H3, the data and the metrics did
not change.

---

## 3. Data

| Set | Size | Written by | Use |
|---|---|---|---|
| train | 298 | 261 AI-drafted + 37 author | training, exemplars, prior texts |
| dev | 40 | 39 AI-drafted + 1 author | selection, τ, all development decisions |
| **test** | **120** (50 ACT / 50 CLARIFY / 20 ABSTAIN) | **the author, a native Egyptian speaker**; no drafts | the locked, one-shot evaluation |
| native Egyptian (`Alhasan/egyptian-nlu`, CC BY 4.0) | 196 (192 out of scope, 4 alarms) | an independent native speaker | external out-of-scope check, intent level |

- **Split by scenario family** (paraphrases and complete/incomplete minimal pairs share a family). The
  manifest is hash-locked (`940beba352ec`). About 70% of the requests are project-management requests,
  the target users.
- **Agreement** ([agreement.md](agreement.md)). A second native annotator independently labelled 60
  items: intent κ = 0.916, route κ = 1.000, target sets 92.3% (CLARIFY), datetimes and reminder texts
  100% (ACT). The 5 disagreements were kept as they are; no gold label was changed.
- **The native contributor collection planned in the proposal did not happen** (protocol log
  2026-10-10). The test split is written by one person, who also designed the system. That is the main
  external-validity threat (section 7).

---

## 4. Evaluation discipline

- **Pre-registration.** [evaluation_protocol.md](evaluation_protocol.md) fixed hypotheses, metrics,
  statistics and reading rules before training. Every later change is in its dated deviations log, with
  the reason and whether the test split had been opened. All 13 entries were made **before** the
  test split was opened.
- **Freeze.** `athar-train fit` hashes the models, the configuration, the protocol, the contract and every
  shared-core source into `models/freeze.json` (current `718ed2305a17…`). The locked run refuses to start if
  any of them changed. S7's adapters are frozen by SHA-256 (protocol log 2026-10-18).
- **Statistics.** 95% percentile intervals from a family-level bootstrap (10,000 resamples).
  Paired approximate randomization swapping whole families (10,000 permutations). Holm-Bonferroni
  within each declared family. Exact McNemar as a sensitivity check.
- **Test discipline in the GPU notebook.** On the test split it refuses dev-only analyses and any S4
  system without a dev τ. The confidence-gate threshold is never picked on test.

Deviations that shaped the LLM results (full log: protocol section 10):

| Date | Change | Basis |
|---|---|---|
| 2026-10-10 | S4 primary `calibrated` → `raw` | A priori: content-free calibration assumes an empty input deserves a uniform label, but here it is genuinely out of scope. Observed: it lowered native out-of-scope recall in 9/9 cells |
| 2026-10-11 | S4 primary `raw` → `prior` (final) | `prior` abstained wherever `raw` did, and more often (significant after Holm for Nile-Chat 3/3 and Jais 2 2/3 exemplar sets). **Disclosed:** chosen with the native set, which is therefore supporting, not untouched, evidence |
| 2026-10-15 | Jais 3 (S4), S8, precision policy, Jais 2 few-shot role fix | Release of Jais 3; prompt check against each model's own chat template |
| 2026-10-16 | Secondary S4 analysis: inline exemplars | Dev result (run 4). Declared with its test before the test was opened |
| 2026-10-17 | S7 optimisation set to the S6 recipe; S7 headline `prior` | Run 6 instability. Settings taken from the pre-registered S6 recipe, no search |
| 2026-10-18 | S7 frozen at the run-6b adapters | Test must score the weights evaluated on dev |

---

## 5. Development results

Dev has 40 items, 39 of them AI-drafted. Every interval is wide, and section 6.4 shows that dev
rankings did not transfer. These results motivated decisions; they are not evidence about native
users.

### 5.1 Core systems (dev, frozen models)

| System | STS [95% CI] | Intent macro-F1 | Unsafe execution |
|---|---|---|---|
| S1 keyword rules | 0.800 [0.59, 0.95] | 0.756 | 0.000 |
| S2 word TF-IDF + LR | **0.950** [0.88, 1.00] | 0.931 | 0.000 |
| S3 + character n-grams | 0.925 [0.84, 1.00] | 0.897 | 0.000 |
| oracle intent | 1.000 | 1.000 | 0.000 |

No dev difference is significant: S2 vs S1 +0.150 (Holm p = 0.80); S3 vs S1 +0.125 (0.80); S3 vs S2
-0.025 (1.0). The confidence gate's τ is 0.0 for S2 and S3.

### 5.2 S4 LLM intent recognizers (dev; Kaggle 2× T4)

| Model | Precision | Primary (6-shot `prior`) | Best other setting |
|---|---|---|---|
| Jais-3-3.7B | fp32 | **0.950** [0.88, 1.00] | 0.950 (several) |
| Nile-Chat 4B | fp32 | 0.900 [0.75, 1.00] | 0.875 |
| Qwen3 8B | fp16 | 0.850 [0.71, 0.96] | 0.875 (6-shot `raw`) |
| Jais-3-7B | fp16 | 0.825 [0.71, 0.93] | 0.975 (0-shot `calibrated`) |
| Jais 2 8B | nf4, fp32 compute (fp16 overflowed) | 0.725 [0.54, 0.89] | 0.925 (0-shot `prior`) |

Both Jais models scored *worse* with six exemplars than with none. That motivated section 5.4.

### 5.3 Native Egyptian out-of-scope recall (192 requests by an independent native speaker)

| System | Out-of-scope recall [95% Wilson CI] |
|---|---|
| **Jais-3-7B** (6-shot `prior`) | **1.000** [0.980, 1.000] |
| S1 keyword rules | 0.990 [0.963, 0.997] |
| S7 Jais 2 + QLoRA, run 6b (`prior`, seeds 0/1/2) | 1.000 / 0.990 / 0.990 |
| Qwen3 8B | 0.974 [0.940, 0.989] |
| Jais 2 8B | 0.964 [0.927, 0.982] |
| Nile-Chat 4B | 0.964 [0.927, 0.982] |
| Jais-3-3.7B | 0.958 [0.920, 0.979] |
| S2 word TF-IDF + LR | 0.870 [0.815, 0.910] |
| S3 + character n-grams | 0.865 [0.809, 0.906] |

Paired exact McNemar on the same 192 items (only A right / only B right):

| A → B | Recall | Discordant | p |
|---|---|---|---|
| Jais 2 8B → **Jais-3-7B** | 0.964 → 1.000 | 0 / 7 | **0.016** |
| Jais-3-3.7B → Jais-3-7B | 0.958 → 1.000 | 0 / 8 | 0.008 |
| Qwen3 8B → Jais-3-7B | 0.974 → 1.000 | 0 / 5 | 0.063 |
| Nile-Chat 4B → Jais-3-3.7B (size-matched) | 0.964 → 0.958 | 4 / 3 | 1.000 |

**Dialect gap (S1-S3).** The same 192 requests in their Saudi/MSA MASSIVE originals give recall 0.979 /
0.865 / 0.859 against 0.990 / 0.870 / 0.865 in Egyptian. That is no gap (McNemar p ≥ 0.5).

**Caveats.** Jais 2 ran in 4-bit while Jais 3 ran in fp16. The native set informed the S4 method
choice. The p-values are not multiplicity-adjusted.

### 5.4 Few-shot format (dev, run 4; three exemplar sets)

| Model | Exemplars as chat turns (S4 format) | Old role (Jais 2) | Exemplars listed in the user message |
|---|---|---|---|
| Jais-3-7B | 0.825 ± 0.025 | | **0.900 ± 0.043** |
| Jais 2 8B | 0.708 ± 0.052 | 0.717 ± 0.146 | **0.875 ± 0.043** |

- **Jais 2 role check.** Jais 2's template answers under the role `ai`, while our few-shot turns had used
  `assistant`. Correcting the role made no difference (11 vs 12 discordant items, p = 1.0), which
  refutes our earlier hypothesis that it caused Jais 2's instability.
- **Format effect.** Inline exemplars scored higher in every exemplar set for both models (pooled
  discordant 24/4 for Jais 2, 15/6 for Jais-3-7B).
- **Handling.** The effect was found on dev, so it was declared as a secondary test-time hypothesis
  instead of being adopted. Section 6.2 gives the test result.

### 5.5 S7: Jais 2 8B + QLoRA (dev)

| Training loop | `raw` STS by seed (0/1/2) | `calibrated` | `prior` (headline) |
|---|---|---|---|
| Run 6: one step per item, constant LR, unweighted | 0.925 / **0.325** / **0.650** (0.633 ± 0.300) | 0.892 ± 0.029 | not computed |
| Run 6b: batch 16, 10% warm-up + linear decay, clip 1.0, class-balanced loss | 0.950 / 0.925 / 0.925 (0.933 ± 0.014) | 0.892 ± 0.038 | **0.950 ± 0.000** |

- **Run 6 failure mode.** It was label bias, not lost discrimination. Seed 1 labelled every reminder out
  of scope, and seed 2 never predicted out of scope. Calibration on the same weights recovered 0.875.
- **Run 6b fix.** It follows the protocol's pre-registered S6 recipe, with the registered adapter
  settings unchanged (r 16, α 32, dropout 0.05, 3 epochs, LR 2e-4). Each training answer ends with
  the template's end-of-turn token (`<|eot_id|>`). The components were changed together, so the fix is
  not attributed to any one of them.
- **No gain over zero-shot.** S7 `prior` 0.950 vs zero-shot Jais 2 0.925 (1-3 discordant items, p = 1.0).
- **Weak spot.** With «فكرني» replaced by "remind me", seed 1 falls to 0.50; S2 (0.773) and S3 (0.636)
  share this gap.
- **Freeze.** The run-6b adapters are frozen for the test run (SHA-256 in [jais3.md](jais3.md),
  section 13).

### 5.6 S8: Jais-3-7B end to end through tool calling (dev, 3 sampled seeds)

STS **0.717 ± 0.014**, intent F1 0.924 ± 0.022, unsafe execution 0.083 ± 0.029. For comparison, the
shared pipeline scores 0.95 (S2) and 1.00 (oracle) on the same items. Every failure was read in the
traces:

| Cause | Count |
|---|---|
| A **correct reply in plain Arabic** instead of the required tool call (asks for the missing slot, or declines) | 18 of 120 responses; all on the right route, none executed |
| The time expression left inside the reminder text | 3 items |
| A contract convention not followed (asks for a date that is never a gap; picks a reading of an ambiguous date and acts on it) | 5 items |
| Weekday arithmetic error («يوم الأحد» resolved to Monday) | 1 item, 1 seed |

Had the 15 plain-text replies that ask for exactly the gold slots been tool calls, STS would be about
0.84. That figure is descriptive only. Jais 3 understands the requests (intent F1 0.92) but does not
reliably follow the tool contract.

---

## 6. Locked test results (120 author-written items)

All test commits ran on Kaggle 2× T4 with the pinned revisions and the frozen manifest. S4 (7a, 7b) used
notebook 2026-10-17a, and S7 + S8 (7c+7d) used 2026-10-18b. Each model ran at its dev precision. Every
report covers exactly the 120 test items, and S7's adapter hashes match the frozen ones (all verified).
Outputs: `results/s4/7a-jais3/`, `results/s4/7b-others/`, `results/s4/7cd-s7-s8/`.

### 6.1 S4 intent recognizers

| Model | **Primary: 6-shot `prior`** [95% CI] | 0-shot `prior` | 6-shot inline `prior` (secondary) | Intent acc. (primary) | Unsafe exec. (primary) | ECE (primary) |
|---|---|---|---|---|---|---|
| Jais 2 8B (nf4-fp32) | **0.758** [0.67, 0.84] | 0.683 | 0.733 | 0.917 | 0.142 | 0.114 |
| Jais-3-3.7B (fp32) | 0.742 [0.65, 0.83] | 0.667 | **0.758** | 0.908 | 0.108 | **0.030** |
| Nile-Chat 4B (fp32) | 0.742 [0.65, 0.83] | 0.575 | 0.725 | 0.925 | 0.133 | 0.060 |
| Qwen3 8B (fp16) | 0.700 [0.61, 0.79] | 0.642 | 0.708 | 0.883 | 0.150 | 0.108 |
| Jais-3-7B (fp16) | 0.550 [0.46, 0.65] | 0.642 | 0.733 | 0.683 | 0.100 | 0.226 |

- **The three leading systems are statistically indistinguishable** on 120 items: Jais 2, Jais-3-3.7B and
  Nile-Chat, with CIs overlapping almost entirely. No pairwise LLM test was pre-registered for the test
  split, so this ordering is descriptive.
- **Jais-3-7B's low primary score is a prompt-format failure.** Its over-abstention rate is 0.34, meaning
  in-scope requests were declined, and ACT-item STS is 0.40. With inline exemplars, over-abstention falls
  to 0.12 and STS rises to 0.733. Its 0-shot `prior` is 0.642.
- **Few-shot helps every model on test** (6-shot vs 0-shot `prior`): Jais 2 +0.075, Jais-3-3.7B +0.075,
  Nile-Chat +0.167, Qwen3 +0.058. The exception is Jais-3-7B in the chat-turn format. On dev, few-shot had
  hurt both Jais models.
- **Confidence gate (τ from dev).** For Jais 2 (τ = 0.7) it lowers unsafe execution from 0.142 to 0.083 at
  a cost in STS (0.758 → 0.700). For the other models the dev τ is 0, so they are unchanged.
- **Calibration.** Jais-3-3.7B is best calibrated (ECE 0.030); Jais-3-7B 6-shot is worst (0.226).

### 6.2 Pre-registered secondary test: inline vs chat-turn exemplars

Declared 2026-10-16, before the test was opened. Family-level paired randomization test, Holm over the
two confirmatory models:

| Model | Inline | Chat turns | Difference [95% CI] | Discordant (inline / turns) | p | Holm p |
|---|---|---|---|---|---|---|
| **Jais-3-7B** (confirmatory) | 0.733 | 0.550 | **+0.183** [0.113, 0.258] | 24 / 2 | 0.0001 | **0.0002** |
| Jais 2 8B (confirmatory) | 0.733 | 0.758 | -0.025 [-0.078, 0.024] | 3 / 6 | 0.516 | 0.516 |
| Jais-3-3.7B (descriptive) | 0.758 | 0.742 | +0.017 [-0.024, 0.059] | 4 / 2 | 0.695 | |
| Nile-Chat 4B (descriptive) | 0.725 | 0.742 | -0.017 [-0.063, 0.027] | 3 / 5 | 0.726 | |
| Qwen3 8B (descriptive) | 0.708 | 0.700 | +0.008 [-0.034, 0.051] | 4 / 3 | 1.000 | |

**Reading.** The format sensitivity is specific to Jais-3-7B. It is large, and it replicates from dev to
held-out author-written data. For Jais 2, the dev advantage of +17 points was not reproduced. Few-shot
evaluations of Jais-3-7B should report the exemplar format, because it changes the conclusion.

### 6.3 Where the errors are: intent vs the shared pipeline

| Model (primary) | Intent accuracy | Success when the intent is right | Non-ACT items executed (of 70) | ACT items executed with wrong arguments (of 50) | Execution precision |
|---|---|---|---|---|---|
| Jais 2 8B | 0.917 | 0.827 (n = 110) | 0.043 | 0.280 | 0.660 |
| Jais-3-3.7B | 0.908 | 0.817 (n = 109) | 0.043 | 0.200 | 0.711 |
| Nile-Chat 4B | 0.925 | 0.802 (n = 111) | 0.043 | 0.260 | 0.680 |
| Qwen3 8B | 0.883 | 0.792 (n = 106) | 0.043 | 0.300 | 0.640 |
| Jais-3-7B | 0.683 | 0.805 (n = 82) | 0.029 | 0.200 | 0.625 |

- **15 test items (12.5%) fail for all five models although every model chose the correct intent.**
  - 10 are executed with wrong arguments.
  - 2 are executed when they should have been clarified.
  - 3 are clarified when they were complete.
  - 12 of the 15 are reminders.
- **The oracle is bounded.** The shared core is deterministic given the intent, so the gold-intent
  oracle fails on these 15 as well: **its test STS cannot exceed 0.875**.
- **Reminders are the weak path.** STS on reminder items is 0.26-0.68 per model, against 0.87-0.92 on
  alarms.
- **Unnecessary clarification is 0.06 for every model**, the same 3 of 50 ACT items. It comes from the
  shared extractor.

This is the most important result for the core research question. The extractor and policy were
developed on drafted text, and author-written requests expose them. They cap every system equally, so
H1-H3, which compare intent recognizers through the same core, remain a fair comparison. The absolute
STS level, however, is set by the shared core. The argument-level error analysis (protocol section 7)
will name the failing patterns.

### 6.4 Dev vs test

| System | Dev STS (40, drafted) | Test STS (120, author) |
|---|---|---|
| S7 Jais 2 + QLoRA (`prior`, 3 seeds) | 0.950 ± 0.000 | 0.786 ± 0.024 |
| S4 Jais 2 8B | 0.725 | 0.758 |
| S4 Jais-3-3.7B | 0.950 | 0.742 |
| S4 Nile-Chat 4B | 0.900 | 0.742 |
| S8 Jais-3-7B tool calling (3 seeds) | 0.717 ± 0.014 | 0.725 ± 0.033 |
| S4 Qwen3 8B | 0.850 | 0.700 |
| S4 Jais-3-7B | 0.825 | 0.550 |

Among the S4 models, the dev ranking (Jais 2 last, Jais-3-3.7B first) did not survive.

Unsafe execution: 0.000-0.025 on dev, 0.100-0.150 on test for S4 and S7. A 40-item drafted dev set was too small,
and too far from real text, to select between LLMs. This supports the protocol's rule that nothing
is concluded from dev.

---

### 6.5 S7: Jais 2 8B + QLoRA, frozen run-6b adapters (commit 7c)

The three adapters were loaded and SHA-256 checked. Dev was rescored first as a reproduction check: 6
of 9 seed × method cells match run 6b exactly, and 3 differ by one item (0.025). That is GPU
nondeterminism; the frozen τ was used regardless.

| Scoring | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | Unsafe exec. |
|---|---|---|---|---|---|
| **`prior` (headline)** | 0.758 [0.68, 0.84] | 0.800 [0.72, 0.88] | 0.800 [0.72, 0.88] | **0.786 ± 0.024** | 0.133 |
| `raw` | 0.767 | 0.800 | 0.775 | 0.781 ± 0.017 | 0.133 |
| `calibrated` | 0.725 | 0.658 | 0.700 | 0.694 ± 0.034 | 0.133 |

- **Intent is nearly solved by fine-tuning**, and the remaining errors belong to the shared core. Intent
  accuracy is 0.925 / 0.983 / 0.992 (`prior`). Success given the correct intent is 0.807-0.820, and all
  15 shared-core failures (section 6.3) fail for every seed.
- **Fine-tuned vs zero-shot Jais 2** (declared 2026-10-17 as descriptive, no multiplicity correction),
  `prior` vs S4 Jais 2 0-shot `prior` (0.683), family-level paired test:

  | Seed | Difference [95% CI] | p | Discordant (S7 / 0-shot) |
  |---|---|---|---|
  | 0 | +0.075 [0.000, 0.153] | 0.077 | 15 / 6 |
  | 1 | +0.117 [0.063, 0.183] | 0.0003 | 14 / 0 |
  | 2 | +0.117 [0.063, 0.183] | 0.0003 | 14 / 0 |

  The direction is consistent over seeds. On dev, the 40 drafted items could not show this.
- **`calibrated` is again the weakest scoring**, as on dev and as the a priori argument predicts.
- The gate's τ is 0 for every headline seed, so the gated and ungated results are identical.

### 6.6 S8: Jais-3-7B end to end through native tool calling (commit 7d)

Settings as on dev: fp16, reasoning effort high, xml tool calls, sampling T 0.7 / top-p 0.95 / top-k 20,
seeds 0-2, 144 min.

| | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---|---|---|---|
| Safe Task Success | 0.758 [0.69, 0.83] | 0.725 [0.65, 0.80] | 0.692 [0.61, 0.77] | **0.725 ± 0.033** |
| intent macro-F1 | 0.941 | 0.930 | 0.882 | 0.918 ± 0.032 |
| unsafe execution | 0.100 | 0.108 | 0.108 | 0.106 ± 0.005 |
| no tool call (invalid) | 11 | 12 | 16 | |
| success: ACT / CLARIFY / ABSTAIN | 0.92 / 0.56 / 0.85 | 0.90 / 0.52 / 0.80 | 0.90 / 0.50 / 0.65 | |
| decline-reason accuracy (gold ABSTAIN) | 0.85 | 0.80 | 0.65 | |

- **ACT is S8's strength.** Executions with wrong arguments are 0.08-0.10 of ACT items, against
  0.20-0.30 for the shared core, and S8 almost never asks an unnecessary clarification (0.00-0.02).
- **CLARIFY is its weakness.** Of the 50 incomplete requests, 22-25 fail:
  - 8-12 are answered in plain text instead of with `ask_clarification`.
  - 8-9 are executed with a guessed value for the missing slot. These are all of S8's unsafe executions.
  - 4-7 ask for the wrong slots.
- **It never executed an out-of-scope request.** Its 3-4 invalid answers on ABSTAIN items are plain-text
  declines.

### 6.7 Two failure profiles

| | Shared pipeline (S4 / S7) | End-to-end Jais 3 (S8) |
|---|---|---|
| ACT success | 0.40-0.68 (S4); S7 limited by the same core | **0.90-0.92** |
| ACT items executed with wrong arguments | 0.20-0.30 | **0.08-0.10** |
| CLARIFY success | **0.58-0.84** (S4) | 0.50-0.56 |
| The 15 items failed by every S4 model despite the right intent | 0 of 15 solved | **12-14 of 15 solved** |
| Out-of-scope requests executed | 0.029-0.043 of non-ACT items (S4) | 0 |

The shared rule-based extractor fails on how real users phrase times and reminder texts. The LLM
extracts those well, but does not follow the clarification protocol. The data therefore point to a
hybrid design: LLM argument extraction, with the deterministic policy deciding when to execute and
when to ask. That is a hypothesis for future work, not a result of this project.

## 7. Notes for Jais practitioners

Each point was verified in this project, against the pinned revisions and the models' own chat
templates.

- **Jais 3 chat template.**
  - Every assistant turn needs a thinking field; the template raises an error without one.
  - The generation prompt opens a reasoning block (`<ifm|think>`).
  - To score a label directly, we send few-shot answers with an empty `reasoning_content` and append
    `</ifm|think>` to the prompt. That reproduces exactly how the template writes an answer with no
    reasoning (checked by `llm.answer_consistent` on the real template, which is a test fixture).
- **Jais 3 loading.** It needs `trust_remote_code` (`K2HorizonForCausalLM`). Pin revisions: a Hub
  history rewrite on 2026-10-16 made earlier commit hashes unresolvable, though every file we checked
  was byte-identical.
- **Jais 2 chat template.** Roles are written literally, and the answer opens under
  `<|start_header_id|>ai<|end_header_id|>`. Few-shot answers should therefore use the role `ai`; on our
  data this made no measurable difference.
- **Tokenizers.** Jais 2 and Jais 3 encode all three Arabic labels in 2 tokens each, so summed
  verbalizer scores carry no length bias for them. Nile-Chat (3/2/4) and Qwen3 (4/2/4) do not have
  this property.
- **Precision on T4 (no native bf16).** Jais 2 8B overflowed in fp16 and ran in 4-bit NF4 with fp32
  compute. Jais-3-7B ran in fp16 without overflow. Comparisons between them on T4 therefore carry a
  precision confound; a bf16 GPU removes it.
- **Few-shot format.** Jais-3-7B is strongly sensitive to exemplars given as chat turns: on test it
  over-abstains, and inline exemplars add +18 STS points. Jais 2 is not sensitive in this way.
- **Out-of-scope behaviour.** Jais-3-7B declined 192/192 native out-of-scope Egyptian requests.
- **Tool calling (Jais 3).**
  - Its argument extraction from real Egyptian requests is strong: wrong-argument executions are
    0.08-0.10 of ACT items on test, much better than our rule-based extractor.
  - It often answers in plain Arabic when a tool call is required (9-15% of answers), and on incomplete
    requests it may execute with a guessed value instead of asking.
  - A strict tool contract needs constrained decoding, or validation that turns a plain-text question
    into a clarification call.
- **QLoRA on Jais 2.** One optimizer step per example at a constant LR left the label bias
  seed-dependent. Batched, scheduled, clipped, class-balanced updates removed this. Train the answer to
  end with the template's end-of-turn token, not a generic EOS. With that recipe, 298 mostly drafted
  examples were enough for test intent accuracy of 0.93-0.99 on author-written requests.

---

## 8. Threats to validity

1. **One test author, who designed the system.** Test measures one native speaker's phrasing. The
   independent native set (section 5.3) only partly offsets this, and only for out-of-scope recall.
2. **Synthetic-to-real shift.** Training and development data are 88% and 98% AI-drafted. The dev-test
   reversal (section 6.4) shows this matters.
3. **Small samples.** Dev has 40 items; test has 120, with 20 ABSTAIN items. CIs are about ±0.09 STS, and
   differences under about 0.1 between systems are not resolvable.
4. **Precision confound.** On T4, Jais 2 ran in 4-bit while the other models ran in fp16/fp32.
5. **Forking paths, all disclosed.** The S4 scoring method changed twice and S7's optimisation once, all
   before test and all logged. The native set informed the S4 method choice.
6. **S5/S6** (sentence embeddings, fine-tuned MARBERTv2) are implemented but were not fit on the
   final split, and are not reported.

---

## 9. What remains, and what it will settle

| Step | Settles | Where |
|---|---|---|
| Core locked run: `athar-eval --split test --unlock-test` | **H1-H3** (S2, S3 vs S1), the oracle ceiling (≤ 0.875 by section 6.3), core robustness and calibration | local CPU, about 2 min |
| Error analysis (protocol section 7) | Which argument and policy patterns of the shared core fail on author-written requests (the 15 items of section 6.3), and S8's clarification failures | local |

All GPU runs are complete.

---

## 10. Reproducibility

| Item | Value |
|---|---|
| Split manifest | `data/splits/manifest.json`, `940beba352ec…` |
| Freeze | `models/freeze.json`, `718ed2305a17…` (`athar-train verify models`) |
| Eval config | `configs/eval.yaml`, SHA-256 `dfbd8398c8c3…` (recorded in every S4 report) |
| Model revisions | Jais-3-7B `94d69f49b4af`, Jais-3-3.7B `3d5754d5db87`, Jais-2-8B-Chat `da0e1639cd92`, Nile-Chat-4B `2a365667659b`, Qwen3-8B `b968826d9c46` |
| GPU notebook | `notebooks/kaggle_s4_llm.ipynb` (test S4: 2026-10-17a; S7/S8 test: 2026-10-18b); CPU smoke test and test-run rehearsal: `scripts/smoke_kaggle_notebook.py [--split test]` |
| Environment | Kaggle 2× T4, torch 2.11.0, transformers 5.19.0, peft 0.20.0, bitsandbytes 0.50.2 |
| Outputs | dev: `runs/s4/…`, `runs/ood/core_2026-10-18.json`; test: `results/s4/7a-jais3/`, `results/s4/7b-others/`, `results/s4/7cd-s7-s8/` |
| Details | [jais3.md](jais3.md) (all LLM runs), [evaluation_protocol.md](evaluation_protocol.md) (design, deviations), [data_card.md](data_card.md), [agreement.md](agreement.md) |

### References

Zhao, Wallace, Feng, Klein & Singh (2021), *Calibrate Before Use*. Zhou et al. (2024), *Batch
Calibration*. Dettmers, Pagnoni, Holtzman & Zettlemoyer (2023), *QLoRA*. Mosbach, Andriushchenko &
Klakow (2021), *On the Stability of Fine-tuning BERT*. Ribeiro et al. (2020), *CheckList*.
Berg-Kirkpatrick et al. (2012) and Dror et al. (2018) on significance testing in NLP. FitzGerald et al.
(2023), *MASSIVE*. Abdellatif, *Egyptian NLU* (Hugging Face, CC BY 4.0).
