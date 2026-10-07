# Delivery phases

HLD: [hld.html](hld.html) (published at https://claude.ai/artifact/RXeHV9e9ctgryPsCBQ8AEW)

| Phase | Scope | Weeks | Exit criterion | Status |
|---|---|---|---|---|
| P0 | Repo, uv lockfile (Python 3.11), HLD, test runner, lint | 1 | `uv sync` + `uv run pytest` green | **done** |
| P1 | Task contract + shared core: normalizer, time/argument extractor, policy, templates, schemas | 1-2 | contract tests green; contract v1.0 frozen after review | **done (v0.1, awaiting review of [DECIDE] items)** |
| P2 | Data tooling: JSONL validator, family splitter + manifest, kappa script, annotation guide, data card, 20-item pilot | 2-4 | every item validates; split manifest hashed | **tooling done** (pilot validates, pilot manifest hashed); the 300 items are pending |
| P3 | S1 keyword rules, S2 word TF-IDF+LR, S3 +char n-grams; dev-only search; freeze; CPU/RAM log | 5 | models + config hash saved | planned |
| P4 | Locked-test evaluation: Safe Task Success, F1s, unsafe/unnecessary rates, pairs, family bootstrap; oracle-intent ceiling | 6 | one run, results JSON + tables | planned |
| P5 | Error analysis (<=30 coded failures) + Streamlit demo | 7-8 | tables + working demo | planned |
| P6 | Report + presentation | 8 | submitted | planned |
| P7 | Optional: Jais zero-shot intent (S4), reported separately | after 6 | exploratory section | optional |

## Notes
- The 300 dataset items are written and reviewed by native Egyptian Arabic speakers, as the
  proposal states. Tooling never generates them. Any assistant-drafted pilot item is marked
  `source: assistant_draft` and stays out of the test split.
- P2 tooling is `athar-data` (`src/athar/data/`). Order of work for the dataset: freeze
  contract v1.0 -> write `data/athar.jsonl` -> `validate --strict` -> 60-item agreement sheet
  + `kappa` -> adjudicate -> `split` -> `verify` -> commit dataset + manifest. See
  [annotation_guide.md](annotation_guide.md) section 7.
- P4 adds an **oracle-intent** run (gold intent + shared extractor). It is an upper bound that
  separates extractor errors from intent errors in the error analysis.
