# ATHAR Chat: methodology brief for LLM reviewers

Two pages for reviewers who build or evaluate Arabic LLMs (Jais and others): what is compared,
how the LLMs are treated, and what the results can and cannot show. Full details are in
[evaluation_protocol.md](evaluation_protocol.md) and [task_contract.md](task_contract.md).

## The task

Egyptian project managers, and some general users, send one-turn requests in Egyptian Arabic,
usually code-switched (`فكرني أبعت الـ status report للعميل يوم الخميس الساعة 3 العصر`). The
assistant must do one of three things:
- **ACT**: create the reminder or alarm, with the exact time and text;
- **CLARIFY**: ask about exactly what is missing (day, time, AM/PM, a time already past, or the content);
- **ABSTAIN**: decline anything out of scope, such as a recurring request, a request for someone
  else, deleting or editing an item, several actions at once, or another tool.

The primary metric is **Safe Task Success**: an item counts only when the route, the intent, the
exact clarification set and, for ACT, the tool, the instant and the reminder text are all correct.
Unsafe executions are reported separately.

## What is compared, and what is held constant

Every system predicts **only the intent label** (reminder / alarm / unsupported). The normalizer,
the Egyptian date/time extractor, the routing policy and the replies are shared and deterministic.
So any difference in Safe Task Success is caused by intent recognition and nothing else. An
oracle run (gold intent) gives the ceiling that extraction and policy allow.

| | System | Notes |
|---|---|---|
| S1 | keyword rules | written from the contract scope |
| S2 / S3 | TF-IDF + logistic regression (word; + char n-grams) | the core research question: H1–H3 |
| S5 | frozen multilingual-e5 embeddings + LR | pretrained representations, no fine-tuning |
| S6 | MARBERTv2 fine-tuned, 5 seeds | strongest dialectal encoder baseline; mean ± SD |
| **S4** | **Jais 2 8B Chat**, Nile-Chat 4B, Qwen3 8B: 0-shot and 6-shot | instruction-tuned LLMs |
| **S7** | **Jais 2 8B + QLoRA** on the train split | is a tuned Jais competitive with a tuned encoder? |

## How the LLMs are treated (designed to be fair to them)

- **No parsing penalty in the primary number.** Labels are scored by the log-likelihood of their
  Arabic verbalizers (`تذكير` / `منبه` / `غير مدعوم`) after the model's own chat template, as
  lm-evaluation-harness `multiple_choice` does. Free generation is reported as well.
- **Contextual calibration** (Zhao et al. 2021) removes label-prior bias. This is the primary
  S4 method, fixed in advance, so the best of several variants is not cherry-picked after the fact.
- **Raw and generate-then-parse are reported too.** Chat models' first-token scores and written
  answers can disagree (Wang et al. 2024).
- **Few-shot exemplars come from train only.** They use a fixed seed, are balanced over labels,
  and come from distinct scenario families, so no paraphrase of a test item can appear in a prompt.
- **An Arabic system prompt** follows the Jais 2 model card. The prompt is fixed before any test run.
- **Precision is checked, not assumed.** The models run in fp16 on T4 GPUs and every forward pass
  is checked for non-finite logits. If fp16 overflows, the run falls back to 4-bit NF4 weights with fp32
  compute (Gemma-based models run in fp32 from the start), and the precision and the model revision (commit hash) are saved with
  every result.
- **Same inputs, same scorer.** The LLMs receive exactly the normalized text that every other
  system receives, and their outputs go through the same pipeline and scoring code (the `athar`
  package, unit-tested). The LLM code is tested offline with a tiny local model, so the notebook
  logic is covered by CI.

## Statistics and test-set discipline

- **Pre-registered.** Hypotheses, metrics, tests and reading rules are fixed in
  `evaluation_protocol.md` before training. Its hash, the scoring code's hash and the models'
  hashes are stored in a freeze file. The one locked test run refuses to start if anything
  changed, and it refuses to run twice.
- **Family-aware uncertainty.** Paraphrases and minimal pairs share a scenario family, so CIs
  use a family bootstrap (10k resamples). Core comparisons use paired permutation tests at the
  family level with Holm correction.
- **Honest power.** With 120 test items, simulated power is high for differences of about 15
  points of Safe Task Success and low for 5 points. Non-significant results are reported as
  inconclusive, with the effect sizes the CI rules out.
- **Robustness.** Nine label-preserving perturbations are tested: taa marbuta, hamza, alef
  maqsura, digit scripts, diacritics, elongation, punctuation, politeness, and an English
  trigger. Calibration (ECE, Brier, AURC) is reported for every probabilistic system.
- **Out of distribution.** Intent-only results on MASSIVE ar-SA (alarm and calendar-removal
  intents, plus out-of-scope domains).

## What the results can and cannot claim

- **Can:** in this pipeline, on this task, how much does the choice of intent recognizer change
  Safe Task Success? That covers rules, light classifiers, a fine-tuned dialectal encoder, and
  prompted or adapted Jais 2. It also gives the cost of each: CPU milliseconds vs GPU, measured.
- **Cannot:** general claims about Jais or about Egyptian Arabic NLP. The data are 300 written
  (not logged) requests from a few native speakers, and the test set is 120 items.
- **Not tested, by design:** end-to-end LLM tool calling, where the LLM also produces the
  arguments. Here the LLM only chooses the intent. The point is to isolate intent recognition
  and to keep the guarantee that no argument value is invented. An end-to-end Jais
  function-calling run is a natural exploratory follow-up, scored with the same harness.

## Status (October 2026)

- **Done:** contract v1.0 is frozen; tooling, systems, statistics and quality gates are in place
  and tested (CI on Linux and Windows); an end-to-end dry run on assistant-drafted data has
  passed; the Kaggle notebook for S4/S7 is ready.
- **Pending:** the 300 native-speaker items, then the single locked test run.
