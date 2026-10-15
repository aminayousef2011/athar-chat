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
| `jais3-7b` | `inception42/Jais-3-7B` | `a9097a3e2318` | 9.0B (7B class) | S4, S8 | bf16 / fp16 (nf4-fp32 on overflow) |
| `jais3-3.7b` | `inception42/Jais-3-3.7B` | `e664b9e1c555` | 5.1B | S4 | bf16 / fp32 |
| `jais2-8b` | `inception42/Jais-2-8B-Chat` | `da0e1639cd92` | 8B | S4 (pre-registered), S7 | bf16 / fp16 (nf4-fp32 on overflow) |
| `nile-4b` | `MBZUAI-Paris/Nile-Chat-4B` | `2a3656676591` | 4B | S4 (Egyptian) | bf16 / fp32 |
| `qwen3-8b` | `Qwen/Qwen3-8B` | `b968826d9c46` | 8B | S4 (multilingual reference) | bf16 / fp16 |

Revisions are pinned in the notebook. For Jais 3 the pin also fixes the model code it runs
(`trust_remote_code=True`).

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
| Jais 2 few-shot instability (dev STS 0.375-0.85 across exemplar sets) | **Cause found by the prompt check** (Kaggle, 2026-10-15). Jais 2's template opens the answer under the role `ai`, but the few-shot answers were sent as `assistant`, so the model saw six answers under a header it is never asked to answer under. Its chat profile now sends them as `ai`, and `llm.answer_consistent` verifies every model's real template. The dev-only format check scores `turns` (fixed), `turns-old-role` (as before) and `inline` on three exemplar sets, to measure how much the mismatch mattered. |
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

Notebook `notebooks/kaggle_s4_llm.ipynb` (version 2026-10-15a). Each run is one commit
(*Save & Run All*). Bring the outputs back under `runs/s4/<run name>/`.

| # | Run | Flags | Where |
|---|---|---|---|
| 1 | Prompt check | `RUN_PROMPT_CHECK` only | Kaggle (CPU is enough) |
| 2 | S4 Jais 3 on dev | `RUN_S4`, `RUN_EGYPTIAN_OOD`, `RUN_MODELS=["jais3-7b","jais3-3.7b"]` | Kaggle T4 x2, or bf16 GPU |
| 3 | S4 other models, real split | `RUN_S4`, `RUN_EGYPTIAN_OOD`, `RUN_MODELS=["jais2-8b","nile-4b","qwen3-8b"]` | bf16 GPU preferred (Jais 2 parity) |
| 4 | Prompt-format check | `RUN_SHOT_STYLE` | Kaggle |
| 5 | S8 on dev | `RUN_S8` (+ `RUN_S8_EGYPTIAN`) | Kaggle (slow), or an H100 with `S8_SERVER` |
| 6 | S7 on the real split | `RUN_S7_QLORA` | Kaggle |
| 7 | Locked test run, once | `RUN_S4`, `RUN_S7_QLORA`, `RUN_S8`, `SPLIT="test"`, `UNLOCK_TEST=True`, `GATE_TAU` from dev | after the agreement check |
