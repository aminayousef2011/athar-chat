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
data/            dataset, split config and manifest; data/pilot/ = 20 assistant-drafted pilot items
tests/           unit tests for the contract and the tooling (pytest)
```

Later phases add `src/athar/intent/`, `src/athar/eval/`, `models/` and `app/`.

## Setup (CPU only)

```powershell
# Behind a TLS-intercepting proxy, also set:  $env:UV_NATIVE_TLS = "1"
uv sync                 # creates .venv from uv.lock (frozen dependencies)
uv run pytest           # contract + tooling tests
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

The test split is locked: `load_split(..., "test")` raises unless `unlock_test=True`, which only
the Phase 4 evaluation passes. Procedure: [docs/annotation_guide.md](docs/annotation_guide.md).

## Status

See [docs/PHASES.md](docs/PHASES.md).
