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
| S5 (e5-small + LR) and S6 (MARBERTv2, 5 seeds), fit exactly as `configs/train.yaml` specified on the frozen train split; scored on the locked test and the 238 confirmatory natural items | `scripts/posthoc_encoders_human.py encoders` → `encoders.json` | report section 9.1 |
| Leave-one-writer-out over the 17 contributors: train+dev vs train+dev + other writers' messages vs other writers only, with the held-out writer's scenario families removed from training; hyper-parameters fixed (S2-gcv, S3-gcv, S5 C = 1) | `scripts/posthoc_encoders_human.py human` → `human_training.json` | Human messages add +0.12 to +0.14 STS (Holm p ≤ 0.002); S3 + contributors 0.697 vs S1 v1.1 0.723 (p = 0.49) |
| STS by surface phenomenon (regex tags) from the stored natural outputs | `scripts/posthoc_phenomena.py` → `phenomena.{md,json}` | Verb triggers (oracle 0.684) far below tool nouns (0.915); day-scale expressions hardest (0.45-0.58) |

**Consequence for the data.** Section 9.2 trains on folds of the natural set, so the natural set is
no longer untouched evidence for any system built after 2026-11-05.
