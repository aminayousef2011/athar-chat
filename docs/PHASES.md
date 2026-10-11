# Delivery phases

HLD: [hld.html](hld.html) (published at https://claude.ai/artifact/RXeHV9e9ctgryPsCBQ8AEW)

| Phase | Scope | Weeks | Exit criterion | Status |
|---|---|---|---|---|
| P0 | Repo, uv lockfile (Python 3.11), HLD, test runner, lint | 1 | `uv sync` + `uv run pytest` green | **done** |
| P1 | Task contract + shared core: normalizer, time/argument extractor, policy, templates, schemas | 1-2 | contract tests green; contract v1.0 frozen after review | **done**: contract **v1.0 frozen 2026-10-07** (all [DECIDE] items confirmed; gender-neutral replies) |
| P2 | Data tooling: JSONL validator, family splitter + manifest, kappa script, annotation guide, data card, 20-item pilot | 2-4 | every item validates; split manifest hashed | **native test data being written by the author** (`contrib/author_sheet_wave1.xlsx`, 120 core requests; protocol 4.2); no contributor data; external native check: `Alhasan/egyptian-nlu` (4.1) |
| P2b | Expert-review hardening: pre-registered evaluation protocol, scoring + statistics code (family bootstrap, paired permutation + Holm, McNemar, calibration, power), CheckList-style robustness suite, Hypothesis property tests, mypy, coverage gate, pre-commit, CI | 4 | protocol + gates in place, all green | **done** |
| P3 | S1 keyword rules, S2 word TF-IDF+LR, S3 +char n-grams; dev-only search; freeze; CPU/RAM log. Extended baselines (reported separately): S5 frozen Arabic sentence embeddings + LR, S6 fine-tuned MARBERTv2 (5 seeds) | 5 | models + config hash saved | S1-S3, S5, S6 **code done**; end-to-end **dry run on the draft data passed** (train/dev only); real fit waits for the native-speaker split |
| P4 | Locked-test evaluation: one run of `docs/evaluation_protocol.md` (`athar-eval --split test --unlock-test`) | 6 | one run, results JSON + tables | harness **done**; one locked run once the author-written test split exists |
| P5 | Error analysis (<=30 coded failures) + Streamlit demo | 7-8 | tables + working demo | planned |
| P6 | Report + presentation | 8 | submitted | planned |
| P7 | LLM baselines on Kaggle GPUs: S4 Jais 2 8B / Nile-Chat 4B / Qwen3 8B (0- and 6-shot; raw label scoring (primary), calibrated, train-prior, generate; 3 exemplar sets), S7 Jais 2 + QLoRA, MASSIVE ar out-of-distribution check; reported separately | 5-6 | results JSON per model, with model revision | **dry run on the draft data done** (S4 x3, S7; see `docs/dry_run_draft.md`); MASSIVE check to rerun; real run waits for the native-speaker split |

## Notes
- The 300 dataset items are written and reviewed by native Egyptian Arabic speakers, as the
  proposal states. Tooling never generates them. Any assistant-drafted pilot item is marked
  `source: assistant_draft` and stays out of the test split.
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
