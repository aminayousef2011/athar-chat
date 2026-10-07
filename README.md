# ATHAR Chat: Safe Tool Routing for Egyptian Arabic

Course project (Applied NLP), Amina Yousef.

Single-turn Egyptian Arabic requests (with Arabic-English code-switching) are routed to one of
three outcomes:

| Route     | Meaning                                                              |
|-----------|----------------------------------------------------------------------|
| `ACT`     | Supported request, every required argument resolved -> simulated tool call |
| `CLARIFY` | Supported request with missing or ambiguous arguments -> template question |
| `ABSTAIN` | Request outside scope -> templated decline                           |

Two simulated tools: `create_reminder(datetime, reminder_text)` and `create_alarm(datetime)`.
No real reminders or alarms are created.

**Research question:** does lightweight text classification (S2: word TF-IDF + logistic
regression; S3: S2 + character n-grams) improve *Safe Task Success* over keyword rules (S1) when
all systems share the same normalizer, argument extractor, execution policy and templates?

## Layout

```
docs/            HLD, task contract, annotation guide, data card
src/athar/       shared core: normalizer, time/argument extractor, policy, responses, schema
src/athar/data/  dataset tooling (`athar-data` CLI): validator, splitter, agreement, stats
src/athar/intent/ S1 keyword rules, S2/S3 classifiers, search + freeze (`athar-train` CLI)
src/athar/eval/  scoring, family-level statistics, robustness suite, harness (`athar-eval` CLI)
configs/         training/search config
data/            dataset, split config and manifest; data/pilot/ = 20 assistant-drafted pilot items
tests/           unit tests for the contract and the tooling (pytest)
```

Later phases add `models/` (frozen systems), `results/` (the one locked test run) and `app/`.

## Setup (CPU only)

```powershell
# Behind a TLS-intercepting proxy, also set:  $env:UV_NATIVE_TLS = "1"
uv sync                 # creates .venv from uv.lock (frozen dependencies)
uv run pytest           # contract, tooling, statistics and property tests
```

## Quality gates

```powershell
uv run ruff check .                  # lint
uv run mypy                          # types (untyped definitions disallowed)
uv run pytest --cov                  # tests + branch coverage (fails under 85%)
uv run pre-commit install            # same gates + data validation on every commit
```

CI (`.github/workflows/ci.yml`) runs them on Linux and Windows from the frozen lockfile.

## LLM baselines on Kaggle GPUs (S4 / S7)

Jais 2 8B, Nile-Chat 4B and Qwen3 8B (S4), and Jais 2 + QLoRA (S7), run in
[notebooks/kaggle_s4_llm.ipynb](notebooks/kaggle_s4_llm.ipynb) on Kaggle's T4 x2 GPUs, using the
same `athar` scoring code (`athar.intent.llm`, tested offline in `tests/test_llm.py`):

```powershell
uv run python scripts/build_kaggle_kit.py   # -> dist/athar-chat-kit.zip: upload as a Kaggle Dataset
```

Then accept the Jais 2 terms on Hugging Face, add an `HF_TOKEN` Kaggle secret, choose GPU T4 x2
with internet on, attach the dataset and run all. The notebook explains each step.

## Evaluation

The protocol is pre-registered in [docs/evaluation_protocol.md](docs/evaluation_protocol.md):
hypotheses H1-H3 on Safe Task Success, family-bootstrap CIs, paired permutation tests with Holm
correction, robustness, calibration and a power analysis.

```powershell
uv run athar-eval --manifest data/splits/manifest.json --models models --split dev
uv run athar-eval --manifest data/splits/manifest.json --models models --split test --unlock-test  # once
```

## Dataset tooling

```powershell
uv run athar-data validate data/athar.jsonl --strict          # schema + dataset checks
uv run athar-data agreement-sheet data/athar.jsonl --out runs/agreement/sheet.csv
uv run athar-data kappa data/athar.jsonl runs/agreement/sheet.csv --annotator NAME --out docs/agreement
uv run athar-data split data/athar.jsonl --config data/split_config.yaml --out data/splits/manifest.json
uv run athar-data verify data/splits/manifest.json            # fails on any edit after the split
uv run athar-data crosscheck --manifest data/splits/manifest.json --split dev
uv run athar-data stats data/athar.jsonl --manifest data/splits/manifest.json --card docs/data_card.md
```

## Training and freezing (Phase 3)

```powershell
uv run athar-train fit --manifest data/splits/manifest.json --config configs/train.yaml --out models
uv run athar-train verify models     # model files, config and shared-core sources unchanged
```

Extended baselines (reported separately from the core comparison): S5 frozen
`multilingual-e5-small` embeddings + logistic regression, and S6 fine-tuned `MARBERTv2` over 5
seeds. Both use pinned model revisions and need the optional `neural` extra (PyTorch CPU,
transformers):

```powershell
uv sync --extra neural
uv run athar-train fit --manifest data/splits/manifest.json --config configs/train.yaml --out models --extended
uv run athar-eval --manifest data/splits/manifest.json --models models --split dev --systems all
```

`fit` reads train and dev only, searches C and n-gram ranges on dev, refits the winners on train,
and writes `models/S2.joblib`, `models/S3.joblib` and `models/freeze.json` (hashes, dev scores,
CPU time, peak memory, library versions). It refuses to overwrite an existing freeze.

The test split is locked: `load_split(..., "test")` raises unless `unlock_test=True`, which only
the Phase 4 evaluation passes. Procedure: [docs/annotation_guide.md](docs/annotation_guide.md).

## Status

See [docs/PHASES.md](docs/PHASES.md).
