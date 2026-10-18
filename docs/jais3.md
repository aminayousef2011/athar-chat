# Jais 3 in ATHAR Chat

**Status (2026-10-15).** Code, tests, notebook and protocol entry done; GPU runs pending (section 7).
Protocol entry: [evaluation_protocol.md](evaluation_protocol.md), section 10, 2026-10-15.

## 1. What changed, and why

The proposal named **Jais 2 8B** (`inception42/Jais-2-8B-Chat`) as the Arabic LLM baseline, and the
pre-registered protocol keeps it as the primary S4 model. On **2026-10-15** MBZUAI's Institute of
Foundation Models, Inception42 and Cerebras released **Jais 3**: six open models from 0.9B to 375B
parameters (Apache-2.0), built on the K2 Horizon base, with native tool calling, selectable reasoning
effort and a 512K context. The developers report Arabic and agentic gains. Their dialect list names
Emirati, Gulf and North African Arabic, **not Egyptian**.

The release came after the models were frozen (2026-10-12) and **before the test split was opened**.
Jais 3 is therefore **added beside Jais 2, not substituted for it**. Nothing in the core comparison
(H1-H3: S1 vs S2 vs S3) changes. Jais 3 lets the project ask two questions that the proposal could not:

| | Question | System | Evidence |
|---|---|---|---|
| Q1 | Does Jais 3 improve on Jais 2 at the same task, prompt and scoring? Does a general Arabic 3.7B model match an Egyptian-specialised 4B one (Nile-Chat)? | S4 (intent only), Jais-3-7B and Jais-3-3.7B | locked test split (STS, CIs); paired McNemar on 192 native out-of-scope Egyptian requests |
| Q2 | Can a tool-calling LLM do the **whole** safe-routing task (tool, moment, reminder text, clarification targets, abstention) without the deterministic extractor? | S8 (new), Jais-3-7B end to end | locked test split (STS over 3 seeds), failure breakdown |

Q2 is the question an expert reviewer asks of an intent-only LLM baseline. Jais 3's native tool
calling makes it a direct test of the developers' agentic claims on Egyptian Arabic.

## 2. Models

| Key | Model | Revision | Params | Role | Precision: bf16 GPU / 2x T4 |
|---|---|---|---|---|---|
| `jais3-7b` | `inception42/Jais-3-7B` | `94d69f49b4af` | 9.0B (7B class) | S4, S8 | bf16 / fp16 (nf4-fp32 on overflow) |
| `jais3-3.7b` | `inception42/Jais-3-3.7B` | `3d5754d5db87` | 5.1B | S4 | bf16 / fp32 |
| `jais2-8b` | `inception42/Jais-2-8B-Chat` | `da0e1639cd92` | 8B | S4 (pre-registered), S7 | bf16 / fp16 (nf4-fp32 on overflow) |
| `nile-4b` | `MBZUAI-Paris/Nile-Chat-4B` | `2a3656676591` | 4B | S4 (Egyptian) | bf16 / fp32 |
| `qwen3-8b` | `Qwen/Qwen3-8B` | `b968826d9c46` | 8B | S4 (multilingual reference) | bf16 / fp16 |

Revisions are pinned in the notebook. For Jais 3 the pin also fixes the model code it runs
(`trust_remote_code=True`).

**Hub history rewrite (2026-10-16).** The Jais 3 repositories' git history was rewritten after
release: the same commits (same titles and dates) now have new hashes, and the first pins
(`a9097a3e2318`, `e664b9e1c555`) no longer resolve. At the new pins, every file checked is
byte-identical to the files downloaded at the old ones on 2026-10-15: chat template, config,
generation config, tokenizer and tokenizer config, and the model code `modeling_k2_horizon.py`.
The weights date from the 2026-09-29 upload commit. Their SHA-256 (Git LFS) at the new pins,
recorded so that any later change is detectable:

| Model | File | SHA-256 (first 16) | Bytes |
|---|---|---|---|
| Jais-3-7B | model-00000-of-00004.safetensors | `0adda0c43ca49c64` | 5,314,184,744 |
| Jais-3-7B | model-00001-of-00004.safetensors | `f3260a4370d82867` | 5,335,160,760 |
| Jais-3-7B | model-00002-of-00004.safetensors | `62b52f3a705206de` | 5,335,764,816 |
| Jais-3-7B | model-00003-of-00004.safetensors | `62a1b30ea404f525` | 2,013,284,024 |
| Jais-3-3.7B | model-00000-of-00002.safetensors | `acd8f4b0100d53c2` | 5,345,122,440 |
| Jais-3-3.7B | model-00001-of-00002.safetensors | `6a885c9d010571c1` | 4,771,426,176 |
| both | tokenizer.json | `838d767b7c9925ff` | 20,637,260 |

**Precision parity** (review item). On the drafts, Jais 2 ran 4-bit (`nf4-fp32`) because it overflows
in fp16 on T4s, while Qwen3 ran in fp16 and Nile-Chat in fp32. A Jais 2 vs Jais 3 difference could then
partly be a quantization effect. The notebook now takes `auto:<fallback>` precisions:
- **bf16 on every GPU with native bf16** (compute capability 8.0+: A100, L4, H100). That is the
  precision all five models were released in.
- The fallback on T4s.

Every result records the precision it used. The Jais 2 vs Jais 3 comparison should come from a bf16
run. If only T4s are available, it is reported with its precisions and the confound is stated.

## 3. Integration details, verified against the real template

The Jais 3 chat template is vendored byte for byte as a test fixture
(`tests/fixtures/jais3_chat_template.jinja`, SHA-256 checked). The offline tests render prompts exactly
as Jais 3 does.

1. **Every assistant turn needs a thinking field.** Few-shot answers without one make the template
   raise an error (tested). The Jais 3 chat profile (`llm.chat_profile`) gives each exemplar answer an
   empty `reasoning_content`, which the template renders as `<ifm|think>\n</ifm|think>answer`.
2. **The generation prompt opens a thinking block.** With `reasoning_effort="high"` it ends in
   `<ifm|think>\n`. S4 scores the label after an appended `</ifm|think>`. The test
   `test_jais3_scores_the_label_exactly_where_its_template_writes_an_answer` shows that "prompt + label"
   equals the template's own rendering of a finished answer with no reasoning. The label is therefore
   scored in-distribution, as with Qwen3's `enable_thinking=False`.
3. **Tool calls.** S8 uses the template's default `xml` call format (the vendor's example uses it too).
   The parser also reads `json`, `xml_typed` and Qwen-style `<tool_call>{...}</tool_call>`.
4. **Runtime.** The model code supports SDPA attention, so it runs under Transformers on T4s.
   `transformers` is pinned to the lockfile version (5.19.0) on Kaggle as well. The card validated
   5.15 or later.
5. **Tokenizer.** Jais 3 has a new tokenizer (250,624 entries). The prompt check records how many
   tokens each label answer takes, because the summed score favours short answers (protocol log
   2026-10-09).

The other models' prompts are **byte-identical** to the earlier runs (tested).

## 4. S8: end-to-end tool calling (`src/athar/agent.py`)

| Tool | Arguments | Response |
|---|---|---|
| `create_reminder` | `datetime`, `reminder_text` | ACT, `create_reminder` |
| `create_alarm` | `datetime` | ACT, `create_alarm` |
| `ask_clarification` | `tool`, `missing` ⊆ {date, time, am_pm, past, reminder_text} | CLARIFY with those targets |
| `decline` | `reason` ∈ the six contract reasons (+ `other`) | ABSTAIN (reason kept as a diagnostic) |

- **Input.** The model gets the raw request and the moment it was sent (weekday, date, time, IANA
  zone). It also gets an English summary of the task contract: scope, the clarification targets, the
  date and period conventions. This is the knowledge S1's rules were written from, so S8's failures
  measure understanding and compliance, not ignorance of the project's conventions.
- **Output.** Exactly one valid call becomes the same `Response` every system emits and is scored with
  the same Safe Task Success. A datetime without an offset is read as wall-clock time in the request's
  zone, so DST arithmetic is not part of the task.
- **Failures.** No call, several calls, an unknown tool, invalid arguments or an answer cut off at the
  token limit score as failures (protocol section 3). Nothing is repaired or retried. The cause is kept
  per item.
- **Decoding** follows the model card: reasoning effort `high`; temperature 0.7, top-p 0.95, top-k 20
  for Arabic. Each request is answered once per seed (0, 1, 2), and S8 is reported as mean ± SD over
  seeds. The output limit is 8,192 new tokens, and the truncation rate is reported.
- **Not run for S8:**
  - **Robustness perturbations**, which would multiply the generation cost by ten.
  - **The confidence gate**, because S8 produces no intent probabilities.
- **Diagnostics** (`agent.diagnose`): invalid outputs by cause, truncation, mean output length,
  decline-reason accuracy on gold ABSTAIN items, and executions with the right tool and moment but a
  reminder text that differs from gold. The strict text match is part of Safe Task Success, so this
  separates wording from understanding.
- **Backends.**
  - `HFToolBackend`: Transformers `generate`. Works on any GPU, including Kaggle.
  - `OpenAIToolBackend`: a vLLM/SGLang server with the model's own `k2_horizon` reasoning and tool
    parsers. This is the serving setup the card recommends and is much faster on an H100.
- **How to read it.** S8 vs the oracle-intent row compares an LLM doing everything against the
  rule-based extractor and policy given the gold intent. S8 vs S1-S3 compares end-to-end LLM routing
  against the shared pipeline.

## 5. Other review items addressed

| Item | Change |
|---|---|
| Jais 2 few-shot instability (dev STS 0.375-0.85 across exemplar sets) | The prompt check (Kaggle, 2026-10-15) found a format inconsistency. Jais 2's template opens the answer under the role `ai`, but the few-shot answers were sent as `assistant`. Its chat profile now sends them as `ai`, and `llm.answer_consistent` verifies every model's real template. **The format check (section 10) shows the role was not the cause:** fixed and old role are indistinguishable (11/12 discordant items, p = 1.0). The few-shot penalty comes from giving the exemplars as chat turns at all. |
| Precision confound | bf16 policy (section 2). |
| Robustness switch | `perturbed_copies(items, [])` meant *all* perturbations. Cells that switched robustness off still ran it. No reported number changed, only run time. Fixed, with a regression test. |
| Native comparison Jais 2 vs Jais 3 | Paired exact McNemar on the 192 native out-of-scope Egyptian requests (`PAIRED` in the notebook). |

## 6. What can and cannot be claimed

- Jais 3 (S4) and S8 are **extended / exploratory**. They are outside H1-H3 and get no significance
  tests against the core systems (protocol section 2). The Jais 2 vs Jais 3 McNemar test is descriptive.
- The native Egyptian set informed the S4 scoring choice (protocol log 2026-10-10/11). Results on it
  are supporting evidence; the locked, author-written test split is the unbiased evidence.
- The developers' benchmark numbers are not used or restated as findings.
- The threats in protocol section 4.2 apply unchanged: one writer, who knows the contract;
  training mostly on drafts.

## 7. Runs

Notebook `notebooks/kaggle_s4_llm.ipynb` (version 2026-10-18a). Each run is one commit
(*Save & Run All*). Bring the outputs back under `runs/s4/<run name>/`.

| # | Run | Flags | Where |
|---|---|---|---|
| 1 | Prompt check | `RUN_PROMPT_CHECK` only | Kaggle (CPU is enough) |
| 2 | S4 Jais 3 on dev | `RUN_S4`, `RUN_EGYPTIAN_OOD`, `RUN_MODELS=["jais3-7b","jais3-3.7b"]` | Kaggle T4 x2, or bf16 GPU |
| 3 | S4 other models, real split | `RUN_S4`, `RUN_EGYPTIAN_OOD`, `RUN_MODELS=["jais2-8b","nile-4b","qwen3-8b"]` | bf16 GPU preferred (Jais 2 parity) |
| 4 | Prompt-format check | `RUN_SHOT_STYLE` | Kaggle |
| 5 | S8 on dev | `RUN_S8` (+ `RUN_S8_EGYPTIAN`) | Kaggle (slow), or an H100 with `S8_SERVER` |
| 6 | S7 on the real split (done; old training loop, section 12) | `RUN_S7_QLORA` | Kaggle |
| 6b | S7 with the S6-recipe training, dev only (done 2026-10-18, section 13; adapters frozen) | `RUN_S7_QLORA`, `S7_FROZEN=False` | Kaggle |
| 7 | Locked test run: each system once, `SPLIT="test"`, `UNLOCK_TEST=True`, in four commits (12-hour limit): 7a `RUN_S4` Jais 3 models and 7b `RUN_S4` Jais 2, Nile-Chat, Qwen3 (both with `GATE_TAU` from dev); 7c `RUN_S7_QLORA` with `S7_FROZEN` (loads the run-6b adapters from a second dataset, hash-checked; τ from run 6b); 7d `RUN_S8` | Each S4 commit also scores the declared `inline` secondary (`S4_INLINE`) | after run 6 |

## 8. Prompt check results (Kaggle, notebook 2026-10-15b, 2× T4)

Outputs: `runs/results/s4_results/` (not committed). Manifest `940beba352ec`, transformers 5.19.0, all
revisions as pinned.

| Model | Answer role | Answer consistent | Tokens: تذكير / منبه / غير مدعوم | 6-shot prompt tokens |
|---|---|---|---|---|
| Jais-3-7B | assistant (+ empty thinking) | ✓ | 2 / 2 / 2 | 284 |
| Jais-3-3.7B | assistant (+ empty thinking) | ✓ | 2 / 2 / 2 | 284 |
| Jais 2 8B | **ai** (fixed) | ✓ | 2 / 2 / 2 | 263 |
| Nile-Chat 4B | assistant | ✓ | 3 / 2 / 4 | 339 |
| Qwen3 8B | assistant | ✓ | 4 / 2 / 4 | 384 |

- **Every model now scores its label exactly where its own template writes an answer.** The first
  check (2026-10-15a) found the Jais 2 role mismatch. Its real template, printed in this run,
  confirms the diagnosis: roles are written literally, and the answer opens under
  `<|start_header_id|>ai<|end_header_id|>`. The template is now a regression test (`tests/test_llm.py`).
- **Both Jais tokenizers encode the three Arabic labels in the same number of tokens (2/2/2).** So the
  summed-score length bias examined in the 2026-10-09 ablation cannot affect Jais 2 or Jais 3. It can
  still affect Nile-Chat and Qwen3, where `غير مدعوم` (out of scope) is 4 tokens against 2 for `منبه`.

## 9. Dev results, real split (Kaggle 2× T4, notebook 2026-10-16a)

Runs 2 and 3, outputs in `runs/s4/2026-10-16-run2-jais3/` and `runs/s4/2026-10-16-run3-others/` (not
committed). Dev has 40 items (39 assistant drafts, 1 author-written), so every CI is wide and **nothing is
decided on these numbers**. The locked author-written test split decides. Core systems are from
`athar-eval --split dev` on the frozen models.

**Safe Task Success on dev** (95% family-bootstrap CI). S4 rows use the primary method (6-shot `prior`);
the best other S4 setting is shown for reference.

| System | Precision on T4 | STS, primary | Best other S4 setting |
|---|---|---|---|
| S2 word TF-IDF + LR | CPU | **0.950** [0.88, 1.00] | |
| S3 + char n-grams | CPU | 0.925 [0.84, 1.00] | |
| S1 keyword rules | CPU | 0.800 [0.59, 0.95] | |
| S4 Jais-3-3.7B | fp32 | **0.950** [0.88, 1.00] | 0.950 (several) |
| S4 Nile-Chat 4B | fp32 | 0.900 [0.75, 1.00] | 0.875 |
| S4 Qwen3 8B | fp16 | 0.850 [0.71, 0.96] | 0.875 (6-shot raw) |
| S4 Jais-3-7B | fp16 | 0.825 [0.71, 0.93] | **0.975** (0-shot calibrated) |
| S4 Jais 2 8B | nf4-fp32 (fp16 overflowed) | 0.725 [0.54, 0.89] | 0.925 (0-shot prior) |
| oracle intent | | 1.000 | |

Unsafe execution was 0.000-0.025 for every primary setting.

**Native Egyptian out-of-scope recall** (192 requests written by a native speaker, `Alhasan/egyptian-nlu`;
6-shot `prior`). Paired exact McNemar on the same items (`runs/s4/egyptian_paired_2026-10-16.csv`):

| Comparison | Recall A → B | Only A right / only B right | McNemar p |
|---|---|---|---|
| Jais 2 8B → **Jais-3-7B** | 0.964 → **1.000** | 0 / 7 | **0.016** |
| Jais-3-3.7B → Jais-3-7B | 0.958 → 1.000 | 0 / 8 | 0.008 |
| Qwen3 8B → Jais-3-7B | 0.974 → 1.000 | 0 / 5 | 0.063 |
| Nile-Chat 4B → Jais-3-3.7B (size-matched) | 0.964 → 0.958 | 4 / 3 | 1.000 |

S1 rules score 0.990 here.

What the dev runs show (descriptive; the protocol's caveats apply):
- **Jais-3-7B refused all 192 real out-of-scope Egyptian requests**, 7 more than Jais 2 with no reverse
  case (p = 0.016, not multiplicity-adjusted). Two caveats: Jais 2 ran 4-bit while Jais 3 ran fp16, and
  these items informed the S4 method choice (protocol 4.1), so this is supporting evidence, not a test
  result.
- **Few-shot prompts hurt both 7-8B Jais models on dev.** This holds even with Jais 2's role fixed:
  Jais-3-7B goes from 0.925-0.975 at 0-shot to 0.825 at 6-shot, and Jais 2 from 0.925 to 0.725. The
  role mismatch is therefore not the whole story. Run 4 (three exemplar sets, three formats) measures
  how much of this is exemplar choice and how much is format.
- **Size-matched, general Arabic Jais-3-3.7B equals or beats Egyptian Nile-Chat 4B**: 0.950 vs 0.900 on
  dev, the same on native out-of-scope recall (p = 1.0), and the best LLM on dev overall.
- **Jais-3-7B ran in fp16 on T4s without overflow; Jais 2 did not.** On 2× T4, Jais 3 is therefore
  evaluated at higher precision than Jais 2. A bf16 GPU would remove this difference.
- **No LLM beats S2 on dev** (0.950), and every interval overlaps.

## 10. Few-shot format check (run 4; Kaggle 2× T4, notebook 2026-10-16a)

`runs/s4/2026-10-16-run4-shotstyle/` (not committed). Primary method (`prior`), the 40 dev items, three
exemplar sets (shot seeds 0-2, train only).

| Model (precision) | Exemplars as chat turns (S4 format) | Old role (Jais 2 only) | Exemplars listed in the user message |
|---|---|---|---|
| Jais-3-7B (fp16) | 0.825 ± 0.025 | | **0.900 ± 0.043** |
| Jais 2 8B (nf4-fp32) | 0.708 ± 0.052 | 0.717 ± 0.146 | **0.875 ± 0.043** |

Paired exact McNemar per exemplar set (only A right / only B right):

| Comparison | Set 0 | Set 1 | Set 2 |
|---|---|---|---|
| Jais-3-7B, inline vs turns | 5/1 (p 0.22) | 5/2 (p 0.45) | 5/3 (p 0.73) |
| Jais 2, inline vs turns | 7/0 (p 0.016) | 8/1 (p 0.039) | 9/3 (p 0.15) |
| Jais 2, fixed vs old role | 1/5 (p 0.22) | 6/2 (p 0.29) | 4/5 (p 1.00) |

- **The role mismatch did not matter.** Fixed and old role give the same mean STS, and their discordant
  items split evenly (11 vs 12 over the three sets). The fix stays, because it makes the prompt match the
  model's own template, but it explains nothing. The earlier hypothesis that it caused Jais 2's
  instability is not supported.
- **Exemplars as chat turns hurt both Jais models.** Listing the same exemplars in the user message
  scored higher in every exemplar set for both models: +17 points on average for Jais 2 (individually
  significant in 2 of 3 sets) and +8 for Jais-3-7B (no single set significant). With inline exemplars,
  6-shot is about as good as 0-shot (Jais 2 0.875 vs 0.925; Jais-3-7B 0.900 vs 0.925-0.975), so the
  "few-shot hurts" pattern of section 9 is a prompt-format effect, not a lack of in-context learning.
- **This does not change the S4 primary.** The S4 format (chat turns) was fixed before this check, and
  choosing a format on these 40 dev items would be selection on dev. Instead, `inline` was **declared as a
  secondary S4 analysis on 2026-10-16, before the test split was opened** (protocol section 6 and log). It
  is confirmatory for Jais 2 and Jais-3-7B, with the family-level paired test and Holm over the two, and
  descriptive for the other three models. The test run writes `s4_inline_vs_turns.csv`.
- The analysis is limited: 40 dev items, three exemplar sets, and the same items reused across sets,
  so the per-set tests are not independent.

## 11. S8: Jais-3-7B end to end through tool calling (run 5; Kaggle 2× T4)

`runs/s4/2026-10-16-run5-s8/` (not committed). The two downloaded copies (`results_6.zip`,
`results_7.zip`) are byte-identical, so this is one run. Settings: fp16, reasoning effort high, xml
tool calls, sampling T 0.7 / top-p 0.95 / top-k 20, seeds 0-2, 49.6 min. Same 40 dev items.

| | Seed 0 | Seed 1 | Seed 2 | Mean ± SD |
|---|---|---|---|---|
| Safe Task Success | 0.700 | 0.725 | 0.725 | **0.717 ± 0.014** |
| intent macro-F1 | 0.939 | 0.934 | 0.899 | 0.924 ± 0.022 |
| unsafe execution | 0.100 | 0.100 | 0.050 | 0.083 ± 0.029 |
| no tool call (invalid) | 5 | 5 | 8 | 6 of 40 |
| truncated | 0 | 0 | 0 | 0 |
| mean new tokens | 293 | 285 | 382 | |

For comparison: the shared pipeline with an intent recognizer scores 0.95 (S2), 0.925 (S3), 1.00
(oracle intent), and 0.825-0.975 for Jais-3-7B as an S4 intent recognizer.

**Failure analysis.** Every failure was read in the traces (`s8_trace_*.jsonl`).

| Cause | Items (seeds affected) | What happened |
|---|---|---|
| **No tool call: a correct reply in plain Arabic** | 18 replies over 3 seeds | The model asked for the missing time / AM-PM / date, or declined, in natural language instead of calling `ask_clarification` / `decline`, despite "call exactly one tool". All 18 take the right route, and none executes; 15 of 18 ask for exactly the gold targets (by reading the replies). |
| Asks for the date as well as the time | pm-039, pm-099, pm-112 (1-3 seeds) | The contract says a missing date is never a gap (next occurrence). The model asks for `date` + `time`. Harmless, but the target set misses. |
| Time expression left in the reminder text | pm-076, pm-127, pm-128 | e.g. «أكتب التقرير الأسبوعي للإدارة **يوم الأحد الساعة 9 الصبح**». Executed with the wrong text: an unsafe execution by the metric. |
| A contract convention not followed | gen-074 (3/3), pm-128 (3/3) | «النهارده الساعة 12 بالليل» and «أول الشهر» are ambiguous by contract (`date`). The model picks a plausible reading and executes. |
| **Weekday arithmetic error** | pm-098 (seed 0) | «يوم الأحد» sent on Tuesday 3 Nov: the model sets Monday 9 Nov instead of Sunday 8 Nov. The one genuine reasoning error. |
| Text variant | pm-037 (seed 1) | «كلم» for «أكلم»: the canonical text differs by one letter. |

Reading:
- **Understanding is strong; following the tool contract is the weakness.** Intent macro-F1 is 0.92 and
  nothing was executed on out-of-scope requests. Most failures come from conversational replies where a
  tool call was required, and from the project's conventions (reminder text without the time; no
  `date` gap without a date; ambiguous midnight/"first of the month").
- **Descriptive upper bound, not a pre-registered number:** had the 15 natural-language replies that ask
  for exactly the gold targets been tool calls, S8 would score about 0.84. The pre-registered number
  is 0.717.
- **S8 is less safe than the shared pipeline:** unsafe execution is 0.083 against 0.000 for S2/S3 on the
  same items. Every unsafe execution was a convention or argument error, never an out-of-scope request.
- Not run: S8 on the 192 native Egyptian out-of-scope requests (`RUN_S8_EGYPTIAN`).

## 12. S7: Jais 2 8B + QLoRA on the real split (run 6; Kaggle 2× T4, notebook 2026-10-16a)

`runs/s4/2026-10-17-run6-s7/` (not committed). Recipe as pre-registered: r 16, α 32, dropout 0.05,
3 epochs, LR 2e-4, nf4-fp32, trained on the 298 train items (zero-shot prompt → verbalizer, loss on the
answer tokens). About 79 min per seed. Same 40 dev items.

| Scoring | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | Intent F1 (mean) |
|---|---|---|---|---|---|
| `raw` | 0.925 | **0.325** | **0.650** | 0.633 ± 0.300 | 0.556 |
| `calibrated` | 0.925 | 0.875 | 0.875 | **0.892 ± 0.029** | 0.838 |

For comparison on the same items: S4 Jais 2 zero-shot `prior` 0.925 and `raw` 0.875 (no training); S2
0.950.

What the run shows:
- **`raw` collapses in two of three seeds, and the collapse is a label bias, not a loss of
  discrimination.** Seed 1 labels all 20 reminders and 7 of 13 alarms as out of scope. Seed 2 never
  predicts out of scope and labels 7 alarms as reminders. On the same weights, contextual calibration
  (dividing by the label distribution of content-free inputs) recovers 0.875 for both. The fine-tuned
  models separate the classes, but where the decision boundary sits depends on the seed.
- **A likely cause, not verified:** `lora_finetune` takes one optimizer step per example (batch size
  1) at a constant LR of 2e-4, with no warm-up, decay or gradient clipping. The final label bias then
  depends heavily on the last updates of a shuffled epoch. The per-epoch losses printed in the Kaggle log
  would show whether a seed diverged; they are not in the downloaded outputs.
- **Fine-tuning brought no gain over zero-shot Jais 2.** The best S7 setting (0.892 ± 0.029) is at or
  below zero-shot `prior` (0.925). The same four out-of-scope items fail with and without training:
  pm-193, pm-194 and pm-195 (recurring reminders) and pm-206 (two actions in one request). These are
  out of scope only by the task contract, and the training set has 5 recurring and 9 multi-action
  examples. S2 gets all four right from lexical cues (`كل`, `يومي`), but sends an alarm (pm-156) to
  out of scope.
- **Safety held:** unsafe execution is 0.000-0.025 for `calibrated`. The missed out-of-scope requests
  mostly lacked a time and became clarification questions.
- **Not comparable to the draft runs:** S7 `raw` scored 0.825/0.950/0.900 on the drafts, but with a
  different train set.

**What changed for the test run** (protocol log 2026-10-17, notebook 2026-10-17a), each from a stated
principle rather than a dev search:

| Issue found in run 6 | Change |
|---|---|
| One optimizer step per item at a constant LR; label bias depends on the seed | The protocol's pre-registered S6 recipe: batch 16 (gradient accumulation), 10% warm-up then linear decay, clipping at 1.0, weight decay 0.01. Adapter settings unchanged |
| Training data 68% reminders; out-of-scope requests under-weighted | Class-balanced loss, as S2/S3 (`class_weight="balanced"`) and S6 |
| Answers trained to end with the tokenizer's EOS, not the template's `<|eot_id|>` | `llm.answer_suffix`: the answer ends as the template ends it (tested on the real Jais 2 template) |
| No S7 headline in the protocol | `prior`, the S4 primary rule; it had not been computed for S7 when chosen |
| Run 7c would retrain, and GPU QLoRA is not bit-exact | Each seed is trained once and saved; the same weights score dev, where τ is picked, and then test |
| Loss curve not saved | `s7_training.csv`: loss, LR and gradient norms per epoch |

Run 6 stays in the record. Run 6b checks the new loop on dev only, and it can stop the test run only for a
defect (non-finite loss, collapse onto one label), not for a lower score.

## 13. S7 with the S6-recipe training (run 6b; Kaggle 2× T4, notebook 2026-10-17a)

`runs/s4/2026-10-18-run6b-s7/` (not committed). Same 298 train items, 40 dev items and seeds 0-2 as run 6.
Training as in the protocol log of 2026-10-17: batch 16, 19 steps per epoch, LR warmed up from 4e-5 and
decayed to 4e-6, clipping at 1.0, class-balanced loss, answers ending with `<|eot_id|>`. About 60 min per
seed.

| Scoring | Seed 0 | Seed 1 | Seed 2 | Mean ± SD | Run 6 (old loop) |
|---|---|---|---|---|---|
| **`prior` (headline)** | 0.950 | 0.950 | 0.950 | **0.950 ± 0.000** | not computed |
| `raw` | 0.950 | 0.925 | 0.925 | 0.933 ± 0.014 | 0.633 ± 0.300 |
| `calibrated` | 0.925 | 0.850 | 0.900 | 0.892 ± 0.038 | 0.892 ± 0.029 |

Intent macro-F1 for `prior`: 0.933 ± 0.004. Unsafe execution: 0.000 for `prior` and `raw` in every seed.

**Defect check (the rule fixed in advance): passed.** Every loss is finite, and every seed and method
predicts all three labels (no collapse). Run 6b does not stop the test run.

What the run shows:
- **The seed instability is gone.** `raw` went from 0.325-0.925 to 0.925-0.950. The training loss falls
  smoothly in every seed (epoch 1 → 3: 0.13-0.30 → 0.002-0.007). Clipping was active in epoch 1
  (gradient norm up to 21, clipped to 1.0).
- **Which change fixed it is not identified.** Batching, schedule, clipping, class weights and the
  end-of-turn token changed together. Run 6 vs 6b is a before/after comparison, not an ablation.
- **`calibrated` is now the weakest scoring.** Content-free calibration pushes the out-of-scope requests
  toward in scope (out-of-scope success 0.29-0.57 vs 0.71-0.86 for `prior`). That matches the a priori
  reason it was dropped as the S4 primary (protocol log 2026-10-10): an empty input really is out of
  scope.
- **Fine-tuning vs zero-shot, dev, descriptive:** S7 `prior` 0.950 vs Jais 2 0-shot `prior` 0.925. Each
  seed differs from zero-shot on 1-3 items (paired McNemar p = 1.0), so **no measurable gain on 40
  items**. S2 also scores 0.950.
- **Out-of-scope recall on the 192 native Egyptian requests:** 1.000 / 0.990 / 0.990. For comparison:
  Jais 2 6-shot `prior` 0.964, Jais-3-7B 1.000, S1 0.990. The 4 alarms are too few to read (0.75-1.00).
- **Code-switched triggers are the weak spot, and vary by seed.** With «فكرني» replaced by "remind me"
  (22 items), STS is 0.95 / **0.50** / 0.91. The one dev item phrased in English
  (pm-111, «set a reminder بكرة 9 الصبح …») is read as out of scope by seeds 0 and 1. The trained
  classical systems share the gap: S2 0.773, S3 0.636 on the same perturbation. The training data has
  few English triggers.
- **Remaining errors:** recurring reminders (pm-193 in every seed, pm-194 in seed 2), which are out of scope
  only by the task contract, and pm-111.
- **Caveats:** 39 of 40 dev items are assistant drafts, as are 261 of the 298 train items. The training
  loss near zero shows the train set is almost memorised. Whether this transfers to author-written
  requests is exactly what the test split measures.

Saved adapters (`s7_adapters/s<seed>/`, not committed, 177 MB each; same `adapter_config.json`, SHA-256
`abdb6553a493…`):

| Seed | `adapter_model.safetensors` SHA-256 |
|---|---|
| 0 | `64e87a48a6c24f6ff73cd4cfc62946ba28104f218aa9a63a74d196224bb6c62c` |
| 1 | `30abde5994ea9b010f017b8c802a4c1fa19e1275456021d16de63c2d2a86735d` |
| 2 | `8ada29076edce065ac0c6ac38c8d8d0ad9075ac447d740a6e9724b443d16c9ab` |

**Frozen for the test run (protocol log 2026-10-18).** These three adapters are the S7 models. Commit 7c loads
them from a second Kaggle dataset, refuses any file whose hash differs from the table above, rescores dev as a
reproduction check, and scores test with the τ that run 6b picked on dev (0.0, except 0.9 for s1 `calibrated`).
