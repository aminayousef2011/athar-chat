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

**Results:** [docs/report.md](docs/report.md). On the locked test (120 author-written requests), the keyword
rules (S1, STS 0.767) beat both classifiers (S2 0.600, S3 0.625). The gap is significant under the
pre-registered family-clustered test (Holm p ≤ 0.005) but only borderline when test items that share a
sentence frame are treated as dependent (post hoc, Holm p 0.05-0.11): the test is one writer's templated
phrasing. The classifiers were trained on mostly AI-drafted text and over-regularized by a 40-item drafted
dev set. The shared extractor caps every pipeline system at 0.808 (oracle); one framing-lexicon gap causes 15
of its 23 failures. Post hoc analyses: [docs/posthoc_analyses.md](docs/posthoc_analyses.md).

**Research question:** does lightweight text classification (S2: word TF-IDF + logistic
regression; S3: S2 + character n-grams) improve *Safe Task Success* over keyword rules (S1) when
all systems share the same normalizer, argument extractor, execution policy and templates?

## From the proposal (Jais 2) to Jais 3

The proposal named **Jais 2 8B** as the Arabic LLM baseline, and it remains the pre-registered
S4 model. On **2026-10-15**, after the models were frozen but before the test split was opened,
MBZUAI's Institute of Foundation Models, Inception42 and Cerebras released **Jais 3**, with native
tool calling and selectable reasoning. The LLM track was extended to use it. The core comparison
(H1-H3), the data and the metrics did not change:

- **Jais 3 as an intent recognizer (S4).** Jais-3-7B and Jais-3-3.7B are scored with exactly the
  prompt, exemplars and calibration used for Jais 2. This allows a like-for-like Jais 2 → Jais 3
  comparison, and a size-matched comparison of general Arabic Jais-3-3.7B with Egyptian Nile-Chat-4B.
  Jais 3's chat template needs a thinking field on every assistant turn and opens a reasoning block
  in its generation prompt. The integration handles both and is tested against the real (pinned)
  template.
- **Jais 3 doing the whole task (S8, new).** Jais-3-7B routes each request on its own through native
  tool calling: it decides whether to `create_reminder`, `create_alarm`, `ask_clarification` (and
  what is missing) or `decline` (and why), and fills in the arguments. Its output is scored with the
  same Safe Task Success as every other system. This tests the developers' agentic claims on
  Egyptian Arabic, a dialect their release does not name.
- **Fairer LLM comparisons.**
  - All models run in bf16 where the GPU supports it, with the precision recorded per result.
  - The prompt each model scores is printed and checked against its own chat template. The check
    found that Jais 2's examples carried a different role from the one it answers with; this was
    fixed before any real-split run. A follow-up experiment showed that the role made no measurable
    difference. What hurts both Jais models is giving the examples as earlier chat turns: listing
    them in the user message instead raised dev Safe Task Success by 8-17 points in every exemplar set
    ([docs/jais3.md](docs/jais3.md), section 10). Because this was found on dev, it is not adopted. It
    was declared as a secondary analysis, with its own paired test, before the test split was opened,
    and the locked test run checks it.
  - Jais 2 and Jais 3 are compared, paired, on 192 native out-of-scope Egyptian requests.

Design, integration details and what can be claimed: [docs/jais3.md](docs/jais3.md). Protocol
entry: [evaluation_protocol.md](docs/evaluation_protocol.md), section 10, 2026-10-15.

## Layout

```
docs/            HLD, task contract, annotation guide, data card
src/athar/       shared core: normalizer, time/argument extractor, policy, responses, schema;
                 agent.py = S8 end-to-end tool calling (Jais 3)
src/athar/data/  dataset tooling (`athar-data` CLI): validator, splitter, agreement, stats
src/athar/intent/ S1 keyword rules, S2/S3 classifiers, search + freeze (`athar-train` CLI)
src/athar/eval/  scoring, family-level statistics, robustness suite, harness (`athar-eval` CLI)
configs/         training/search config
data/            dataset, split config and manifest; data/pilot/ = 20 assistant-drafted pilot items
tests/           unit tests for the contract and the tooling (pytest)
app/             local Streamlit demo (logic in src/athar/demo.py)
scripts/         notebook generator, Kaggle kit, smoke test, error analysis
results/         locked test outputs (core, S4/S7/S8) and the error analysis
```

`models/` holds the frozen systems, `results/` the locked test outputs and the error analysis, and `app/`
the local demo.

## Demo (CPU)

```powershell
uv run --extra ui streamlit run app/streamlit_app.py
```

Type an Egyptian Arabic request and see how S1, S2 and S3 route it (reply, tool call or clarification
question), with examples of the report's findings. A second tab shows the locked test results and a
browser of every test failure. The frozen models are hash-verified before they are loaded; the LLM
systems need GPUs and appear only through their recorded results.

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

## LLM baselines on GPUs (S4 / S7 / S8)

| System | Models | Code |
|---|---|---|
| S4: LLM intent recognizer, shared pipeline | Jais-3-7B, Jais-3-3.7B, Jais 2 8B (proposal), Nile-Chat 4B, Qwen3 8B | `athar.intent.llm` |
| S7: QLoRA fine-tuned | Jais 2 8B | `athar.intent.llm` |
| S8: end-to-end tool calling | Jais-3-7B | `athar.agent` |

All three run in [notebooks/kaggle_s4_llm.ipynb](notebooks/kaggle_s4_llm.ipynb), on Kaggle's T4 x2
GPUs or any bf16 GPU, with the same `athar` scoring code (tested offline in `tests/test_llm.py`
and `tests/test_agent.py`, with the real Jais 3 chat template as a fixture):

```powershell
uv run python scripts/build_kaggle_kit.py   # -> dist/athar-chat-kit.zip: upload as a Kaggle Dataset
uv run python scripts/make_notebook.py notebooks/kaggle_s4_llm.ipynb   # the notebook is generated from this script
uv run --extra neural --with pandas python scripts/smoke_kaggle_notebook.py [--split test]   # whole notebook on CPU, tiny models
```

Then accept the Jais 2 terms on Hugging Face (Jais 3 is not gated) and add an `HF_TOKEN` Kaggle
secret. Choose GPU T4 x2 with internet on, attach the dataset, and set the `RUN_*` flags for the run
(the run plan is in the configuration cell and [docs/jais3.md](docs/jais3.md)).

## Evaluation

The protocol is pre-registered in [docs/evaluation_protocol.md](docs/evaluation_protocol.md):
hypotheses H1-H3 on Safe Task Success, family-bootstrap CIs, paired permutation tests with Holm
correction, robustness, calibration and a power analysis.

```powershell
uv run athar-eval --manifest data/splits/manifest.json --models models --split dev
uv run athar-eval --manifest data/splits/manifest.json --models models --split test --unlock-test  # once
uv run athar-ood --models models --systems S1 S2 S3   # native Egyptian evaluation + paired MSA->Egyptian gap
```

**Confidence gate.** Each prediction carries a confidence (the probability of the chosen
intent). Below a threshold τ, chosen on dev at `athar-train fit` and frozen, the assistant asks
for confirmation (`للتأكيد: ... صح كده؟`) instead of executing. `athar-eval` reports the
pre-registered metrics ungated, plus the metrics at the frozen τ and the full risk-coverage
table. At runtime: `Pipeline(recognizer, gate_threshold=tau)`.

## Dataset tooling

Collecting the 300 items from native speakers: [docs/contributor_guide.md](docs/contributor_guide.md).

```powershell
uv run athar-data cards --out contrib                         # scenario cards + collection sheet
uv run --with openpyxl athar-data author-sheet --cards contrib/scenario_cards.csv   # author workbook (120 core)
uv run athar-data import-contrib filled.csv --cards contrib/scenario_cards.csv --out runs/labelling/sheet.csv
uv run athar-data build runs/labelling/sheet.csv --cards contrib/scenario_cards.csv --out data/athar.jsonl
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

Extended baselines (implemented and dry-run on the drafts; **not fitted on the final split, not reported**): S5 frozen
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
