# Post hoc analyses (after the locked test run)

The pre-registered protocol ([evaluation_protocol.md](evaluation_protocol.md)) and its deviations
log are part of the freeze: `models/freeze.json` hashes them, and `athar-train verify` fails if they
change. Analyses made **after** the locked test run are therefore recorded here, not in the protocol.

Rules for everything below:
- no frozen system, model, threshold, gold label or locked output is changed;
- nothing here replaces a pre-registered result; the report labels each item *post hoc*;
- every number is produced by a committed script from committed inputs.

## 2026-10-20: review-driven analyses

**Trigger.** An internal review of the results report raised five questions: the test texts look
templated; the family-level statistics may overstate independence; S2/S3 may be over-regularized;
the error analysis' account of the 13 text failures; and how close S1's lexicon is to the test.

**Timing.** All locked runs finished by 2026-10-18 (core) and the 7a/7b/7cd GPU commits. These
analyses read only their stored outputs, the frozen models and the data.

**Script.** `scripts/posthoc_analysis.py` → `results/posthoc/`. Figures: `scripts/make_figures.py`
→ `docs/figures/`.

| Analysis | Output | Finding (details in the report) |
|---|---|---|
| H1-H3 with test items that share a sentence frame treated as dependent (two frame definitions, and frames joined with families) | `template_sensitivity.csv` | Same differences; Holm p rises from 0.002 / 0.005 (families) to 0.05-0.11 (frames) and 0.53 (families + frames joined, 19 clusters) |
| Frame overlap between test and train | `frame_overlap.csv`, `frame_overlap_summary.json` | 59/120 test items reuse a train frame (51 via the author's 37 train items). The oracle shows the same seen/unseen gap as the trained systems, so the gap is the shared extractor, not training exposure |
| S2/S3 regularization (C) sweep; family-grouped 5-fold CV on train+dev as a test-blind selection rule | `regularization_sweep.csv` | Dev selected the most regularized model. Grouped CV would choose S2 C=1 (test STS 0.650) and S3 C=0.3 (0.675): better than frozen, still below S1 (0.767) |
| Shared-core framing fix («عايز / محتاج + تذكير», the «اعمللي» spelling), applied in memory | `extractor_fix.csv`, `extractor_fix_items.csv` | Oracle 0.808 → 0.925; S1 0.767 → 0.883; no item gets worse on train, dev or test |
| Share of in-scope requests containing S1's gold-intent cue | `lexicon_coverage.csv` | train 0.992, dev 0.939, test 1.000 |

**Housekeeping in the same change.** The second annotator's first-pass sheet (the κ source) is
restored to `data/agreement/` byte-identical from commit `5fba0c9`; later revisions and the
discarded first author attempt are moved out of the repository root with READMEs; stale statements
in `data/draft/README.md` and `docs/data_card.md` are corrected.

## 2026-11-05: review-driven analyses, after the natural-phrasing evaluation

**Trigger.** A review of the report asked whether the classifier arm was a fair test of supervised
learning (the S5/S6 encoders were implemented but never reported), why the classifiers were trained
on AI-drafted data when human-written data existed, and which properties of Egyptian Arabic make
the task hard.

**Timing.** After the locked test (2026-10-18) and the locked natural evaluation (2026-11-02). These
analyses read the frozen models and data, fit new models only in memory, and change no frozen
system, split, label or locked output. Report section 9.

| Analysis | Script → output | Finding |
|---|---|---|
| S5 (e5-small + LR) and S6 (MARBERTv2, 5 seeds), fit exactly as `configs/train.yaml` specified on the frozen train split; scored on the locked test and the 238 confirmatory natural items | `scripts/posthoc_encoders_human.py encoders` → `encoders.json`, kept as `encoders_run1.json` (the run the report cites) and `encoders_run2.json` (a full rerun: S5 and S6 seeds 1-4 identical, seed 0 one item different) | S6 ties S1 on the locked test (0.775 ± 0.019 vs 0.767, p = 0.79) but is below it on new writers (0.614 vs 0.718, p = 0.0009), mostly on out-of-scope requests; S5 0.683 / 0.483 |
| Leave-one-writer-out over the 17 contributors: train+dev vs train+dev + other writers' messages vs other writers only, with the held-out writer's scenario families removed from training; hyper-parameters fixed (S2-gcv, S3-gcv, S5 C = 1) | `scripts/posthoc_encoders_human.py human` → `human_training.json` | Human messages add +0.12 to +0.14 STS (Holm p ≤ 0.002); S3 + contributors 0.697 vs S1 v1.1 0.723 (p = 0.49) |
| STS by surface phenomenon (regex tags) from the stored natural outputs | `scripts/posthoc_phenomena.py` → `phenomena.{md,json}` | Verb triggers (oracle 0.684) far below tool nouns (0.915); day-scale expressions hardest (0.45-0.58) |

**Consequence for the data.** Section 9.2 trains on folds of the natural set, so the natural set is
no longer untouched evidence for any system built after 2026-11-05.

## 2026-11-06: label tokenization check (plan step 1.4)

**Question.** S4 (`athar.intent.llm.LLMIntent.token_logprobs`) tokenizes the prompt and each label
verbalizer separately and concatenates the ids. Does a tokenizer split the joined text differently
at the boundary, so that the scored ids are not the ones the model would see?

**Method.** `scripts/check_label_tokenization.py` (tokenizers only, pinned revisions). It uses the
prompts S4 used: normalized text, the notebook's train exemplars, and 0-shot, 6-shot turns and
6-shot inline. For 40 requests × 3 labels per setting, it compares `tok(prompt) + tok(label)` with
`tok(prompt + label)` → `results/posthoc/label_tokenization.json`.

**Result.** No difference for any of the five S4 models (Jais-3-7B, Jais-3-3.7B, Jais 2 8B,
Nile-Chat 4B, Qwen3 8B): 0 of 120 pairs in every setting, and `answer_consistent` holds for all five.
The S4 and S7 results of report sections 6.3, 6.4 and 8.4 need no correction. Jais 2, which is gated,
was checked on 2026-11-06 with the author's token.

## 2026-11-18: LLM intent recognizers vs the rules (final-report review)

**Question.** The final report compared the LLM recognizers (S4, S7) with the rules only
descriptively (report Table 4). Does any of them differ from S1 under the pre-registered test?

**Method.** `scripts/posthoc_llm_vs_rules.py` → `results/posthoc/llm_vs_rules.json`. Nothing is
rerun: it reads the stored per-item outputs of the GPU runs (S4 6-shot with prior calibration, the
primary setting; S7 as the per-item mean over three seeds) and S1's stored outputs, checks that the
gold labels agree item by item, and applies the family-level paired randomization test and family
bootstrap of `configs/eval.yaml` (same seed), Holm over the six comparisons within each set. All
systems use core v1, so differences come from intent recognition only.

**Result.** Natural set (238): only Jais 2 8B differs from S1 (−0.101 [−0.170, −0.032], Holm p
0.040, worse). Jais-3-3.7B (+0.025 [−0.016, 0.070]) and S7 (+0.011 [−0.032, 0.057]) are level with
the rules; Jais-3-3.7B's unsafe execution rate is lower (0.134 vs 0.151, not tested). Locked test
(120): only Jais-3-7B differs (−0.217, Holm p 0.001, worse). No LLM recognizer beats the rules
significantly on either set.

**Consequence for the report.** The conclusion no longer calls the rules the *safer* choice; it
calls them the practical one at low compute (level with the best LLMs, no GPU, no training data).
This analysis is post hoc and was chosen after the results were known; the report labels it so.

## 2026-11-28: a learned reminder-text extractor (final-report review)

**Question.** Finding 3 shows that the hand-written extractor limits every system, but no learned
alternative was tried. Can a reminder-text extractor learned from labelled messages replace the
hand-written framing and filler lists?

**Method.** `scripts/posthoc_learned_text.py` → `results/posthoc/learned_text.json` (tests:
`tests/test_learned_text.py`). Each word of the normalized message is labelled *content* when it
belongs to the gold reminder text; a per-word logistic regression (C = 1, threshold 0.5, fixed in
advance, not tuned) over the word, its neighbours, affixes, position and core v1.1's time spans
predicts the label. Times are still resolved by core v1.1, and the policy and scoring are unchanged
(gold and predicted texts are both canonicalized, as for every system). Trained on the reminder
messages of train + dev (215, mostly AI-drafted) or of the natural set (61 human messages);
evaluated with gold intents against oracle v1.1, family-level paired test (unadjusted p).

**Result.** Drafted + author → natural: text right on 32 vs 25 of 49 complete reminders, oracle
0.790 → 0.824 (+0.034 [0.012, 0.061], p 0.015). Drafted + author → v3: 32 vs 29 of 52, +0.004
(n.s.). Natural (human) → v3: 40 vs 29 of 52, oracle 0.669 → 0.708 (+0.038 [0.013, 0.067], p 0.023).
Adding core v1.1's word lists as a feature and output filter changed no score. All v3 scenario cards
also occur in the natural set, so the human-trained gain is in-distribution, as in Finding 2.

**Consequence for the report.** Finding 3, the abstract and the conclusion report the result as
exploratory (Table 13); §3.3 states why time resolution stays rule-based. Only aggregates of v3
were read.
