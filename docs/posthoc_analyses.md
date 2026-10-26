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
