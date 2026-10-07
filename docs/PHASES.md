# Delivery phases

HLD: [hld.html](hld.html) (published at https://claude.ai/artifact/RXeHV9e9ctgryPsCBQ8AEW)

| Phase | Scope | Weeks | Exit criterion | Status |
|---|---|---|---|---|
| P0 | Repo, uv lockfile (Python 3.11), HLD, test runner, lint | 1 | `uv sync` + `uv run pytest` green | **done** |
| P1 | Task contract + shared core: normalizer, time/argument extractor, policy, templates, schemas | 1-2 | contract tests green; contract v1.0 frozen after review | **done (v0.1, awaiting review of [DECIDE] items)** |
| P2 | Data tooling: JSONL validator, family splitter + manifest, kappa script, annotation guide, data card, 20-item pilot | 2-4 | every item validates; split manifest hashed | **tooling done** (pilot validates, pilot manifest hashed); the 300 items are pending |
| P2b | Expert-review hardening: pre-registered evaluation protocol, scoring + statistics code (family bootstrap, paired permutation + Holm, McNemar, calibration, power), CheckList-style robustness suite, Hypothesis property tests, mypy, coverage gate, pre-commit, CI | 4 | protocol + gates in place, all green | **done** |
| P3 | S1 keyword rules, S2 word TF-IDF+LR, S3 +char n-grams; dev-only search; freeze; CPU/RAM log. Extended baselines (reported separately): S5 frozen Arabic sentence embeddings + LR, S6 fine-tuned MARBERTv2 (5 seeds) | 5 | models + config hash saved | S1-S3 **code done** (exercised on the pilot); S5/S6 next; real fit waits for the dataset split |
| P4 | Locked-test evaluation: one run of `docs/evaluation_protocol.md` (`athar-eval --split test --unlock-test`) | 6 | one run, results JSON + tables | harness **done**; run after freeze |
| P5 | Error analysis (<=30 coded failures) + Streamlit demo | 7-8 | tables + working demo | planned |
| P6 | Report + presentation | 8 | submitted | planned |
| P7 | Optional: LLM zero-shot intent (S4: Jais 2 or Nile-Chat), reported separately | after 6 | exploratory section | optional |

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
- `data/draft/` holds a 300-item **assistant-drafted** dataset for product review and tooling
  dry runs (`source: assistant_draft`, never in test). It does not replace the native-speaker data.
- P2b: `docs/evaluation_protocol.md` pre-registers H1-H3, the metrics, the statistics and a
  power analysis (120 test items detect ~15-point STS differences, not 5-point ones). The
  protocol, `configs/eval.yaml`, the contract and the scoring code are hashed into
  `models/freeze.json`. The test run refuses a changed copy and refuses to run twice.
- P4 adds an **oracle-intent** run (gold intent + shared extractor). It is an upper bound that
  separates extractor errors from intent errors in the error analysis.
