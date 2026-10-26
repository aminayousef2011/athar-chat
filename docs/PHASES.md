# Delivery phases

HLD: [hld.html](hld.html) (published at https://claude.ai/artifact/RXeHV9e9ctgryPsCBQ8AEW)

| Phase | Scope | Weeks | Exit criterion | Status |
|---|---|---|---|---|
| P0 | Repo, uv lockfile (Python 3.11), HLD, test runner, lint | 1 | `uv sync` + `uv run pytest` green | **done** |
| P1 | Task contract + shared core: normalizer, time/argument extractor, policy, templates, schemas | 1-2 | contract tests green; contract v1.0 frozen after review | **done**: contract **v1.0 frozen 2026-10-07** (all [DECIDE] items confirmed; gender-neutral replies) |
| P2 | Data tooling: JSONL validator, family splitter + manifest, kappa script, annotation guide, data card, 20-item pilot | 2-4 | every item validates; split manifest hashed | **done**: 158 author-written items, split frozen 2026-10-12 (test = 120 author items); second annotator (Ibrahim) on 60 items, independent: intent κ 0.92, route κ 1.00 ([agreement.md](agreement.md)), no gold change; external native check: `Alhasan/egyptian-nlu` |
| P2b | Expert-review hardening: pre-registered evaluation protocol, scoring + statistics code (family bootstrap, paired permutation + Holm, McNemar, calibration, power), CheckList-style robustness suite, Hypothesis property tests, mypy, coverage gate, pre-commit, CI | 4 | protocol + gates in place, all green | **done** |
| P3 | S1 keyword rules, S2 word TF-IDF+LR, S3 +char n-grams; dev-only search; freeze; CPU/RAM log. Extended baselines (reported separately): S5 frozen Arabic sentence embeddings + LR, S6 fine-tuned MARBERTv2 (5 seeds) | 5 | models + config hash saved | S1-S3 **done**: fitted on the frozen split (train 298, dev 40) and frozen. S5/S6: code done and dry-run on the drafts, **not fitted on the final split and not reported** |
| P4 | Locked-test evaluation: one run of `docs/evaluation_protocol.md` (`athar-eval --split test --unlock-test`) | 6 | one run, results JSON + tables | **done** 2026-10-18: one locked run, `results/test_results.json`. H1 S2 vs S1 −0.167 (Holm p 0.002), H2 S3 vs S1 −0.142 (Holm p 0.005), H3 n.s.; oracle 0.808 ([report.md](report.md)) |
| P5 | Error analysis (<=30 coded failures) + Streamlit demo | 7-8 | tables + working demo | error analysis **done** (`scripts/error_analysis.py`, `results/error_analysis/`; report section 6); demo **done** (`app/streamlit_app.py`, logic in `athar.demo`, tested headless in `tests/test_demo.py`) |
| P6 | Report + presentation | 8 | submitted | report **written** ([report.md](report.md)), **revised 2026-10-20** with post hoc analyses (template sensitivity, regularization, extractor diagnosis, frame overlap; [posthoc_analyses.md](posthoc_analyses.md)) and figures; presentation **drafted** (15-slide deck from the original report; to update with the revision) |
| P7 | LLM baselines on GPUs: S4 Jais 2 8B (proposal) / **Jais-3-7B, Jais-3-3.7B** (added 2026-10-15) / Nile-Chat 4B / Qwen3 8B (0- and 6-shot; raw, calibrated, train-prior (primary), generate; 3 exemplar sets), S7 Jais 2 + QLoRA, **S8 Jais-3-7B end-to-end tool calling**, MASSIVE and native Egyptian out-of-distribution checks; reported separately | 5-6 | results JSON per model, with model revision and precision | **dry run on the draft data done** (S4 x3, S7; `docs/dry_run_draft.md`). **Jais 3 + S8 code, tests and notebook 2026-10-15a done** (`docs/jais3.md`); GPU runs on the real split next (run plan in `docs/jais3.md` section 7) |

## Notes
- **Jais 3 (2026-10-15).** The proposal's LLM baseline is Jais 2 8B. Jais 3 was released after the
  freeze and before the test split was opened. It is added beside Jais 2 (S4) and as an end-to-end
  tool-calling system (S8), logged in the protocol (section 10, 2026-10-15), followed by a re-freeze
  that leaves S1-S3 unchanged. Design and claims: [jais3.md](jais3.md).
- **Data as built (differs from the proposal).** The native contributor collection did not happen
  (protocol log 2026-10-10). The test split is 120 requests written by the author, a native speaker;
  train and dev are mostly the 300 assistant-drafted items (`source: assistant_draft`), which never
  enter test. The consequences (a measurable synthetic-to-real shift) are analysed in
  [report.md](report.md), sections 3, 5 and 9.
- P2 tooling is `athar-data` (`src/athar/data/`). Order of work for the dataset: freeze
  contract v1.0 -> write `data/athar.jsonl` -> `validate --strict` -> 60-item agreement sheet
  + `kappa` -> adjudicate -> `split` -> `verify` -> commit dataset + manifest. See
  [annotation_guide.md](annotation_guide.md) section 7.
- P3 is `athar-train` (`src/athar/intent/`). Selection: dev intent macro-F1, ties to the simpler
  model (smaller C, shorter n-grams); the winner is refitted on train only. `models/freeze.json`
  hashes the manifest, config, model files and the shared-core + S1 sources; `load_recognizer`
  refuses anything that changed. S1's lists come from the contract's scope; they may be extended
  after reading train only, before `athar-train fit`.
- Target users: **Egyptian project managers** about their work (~70% of the data, `domain: project_management`) plus general requests; results are reported per domain.
- `data/draft/` holds a 300-item **assistant-drafted** dataset (v2: 210 project-manager + 90 general) for product review and tooling
  dry runs (`source: assistant_draft`, never in test). It does not replace the native-speaker data.
- P2b: `docs/evaluation_protocol.md` pre-registers H1-H3, the metrics, the statistics and a
  power analysis (120 test items detect ~15-point STS differences, not 5-point ones). The
  protocol, `configs/eval.yaml`, the contract and the scoring code are hashed into
  `models/freeze.json`. The test run refuses a changed copy and refuses to run twice.
- **Confidence gate** (`athar.gate`, contract v1.1): below a threshold chosen on dev at `athar-train fit` and frozen,
  an execution becomes a confirmation question. Reported beside the ungated, pre-registered metrics (protocol 6).
- **Native Egyptian OOD check** (`athar-ood`): 196 human-written items of `Alhasan/egyptian-nlu` (CC BY 4.0, pinned).
  On the draft-trained models S2/S3 take ~20% of real out-of-scope Egyptian requests as in scope (S1: 1%): the
  draft has too few out-of-scope negatives, which the native-speaker data must fix.
- P4 adds an **oracle-intent** run (gold intent + shared extractor). It is an upper bound that
  separates extractor errors from intent errors in the error analysis.
