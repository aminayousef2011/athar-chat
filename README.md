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

**Final report (4 pages + appendices):** built from the committed results by
[final_report/build_report.py](final_report/build_report.py)
(`uv run --with python-docx python final_report/build_report.py --pdf`; the PDF step uses
Microsoft Word on Windows). The full technical report is [docs/report.md](docs/report.md).

**Results:** On the locked test (120 author-written requests), the keyword
rules (S1, STS 0.767) beat both classifiers (S2 0.600, S3 0.625). The gap is significant under the
pre-registered family-clustered test (Holm p ≤ 0.005) but only borderline when test items that share a
sentence frame are treated as dependent (post hoc, Holm p 0.055-0.114, and 0.53 when families and frames
are joined into 19 clusters): the test is one writer's templated phrasing. The classifiers were trained on mostly AI-drafted text and over-regularized by a 40-item drafted
dev set. The shared extractor caps every pipeline system at 0.808 (oracle); fixing one framing-lexicon gap repairs 14
of its 23 failures. Post hoc analyses: [docs/posthoc_analyses.md](docs/posthoc_analyses.md).

On 238 messages from 17 other native speakers (pre-registered), the rules again beat TF-IDF (+0.206,
Holm p = 0.0003), and putting the deterministic policy with a rule fallback around Jais-3-7B tool
calling adds +0.102; 56 of the 73 items gained over three seeds are invalid tool calls answered by
the fallback. Post hoc (report section 9): the classifiers' deficit appears to come mostly from
their AI-drafted training data. With about 215 human messages from other writers added (leave one
writer out), S3 reaches 0.697 against the rules' 0.723, a difference that is not significant; those
messages come from the same collection as the evaluation set, so source and match are confounded.
Post hoc ([scripts/posthoc_llm_vs_rules.py](scripts/posthoc_llm_vs_rules.py)), no LLM intent
recognizer beats the rules significantly on either set: on new writers Jais-3-3.7B is +0.025
[−0.016, 0.070] and QLoRA Jais 2 +0.011, and only Jais 2 8B (few-shot) differs, worse.

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
final_report/    the course report only: its builder, and the built .docx/.pdf (not committed)
docs/            supporting documents: HLD, task contract, annotation guide, data card, protocols
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

`models/` holds the frozen systems (`models_v2/` and `models_v3/` those of the natural-phrasing
evaluation and protocol v3), `results/` the locked outputs and analyses, `notebooks/` the Kaggle GPU
notebooks, and `app/` the local demo.

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

Collecting contributor messages: [docs/contributor_guide.md](docs/contributor_guide.md).

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

## Natural-phrasing evaluation (v2, pre-registered)

Pre-registered in [docs/protocol_natural.md](docs/protocol_natural.md) before any contributor message
was labelled: the frozen systems and the v2 systems (`src/athar/v2/`: core v1.1, S1 v1.1, grouped-CV
S2/S3, the S9 hybrid) on requests written by independent native speakers for the test scenarios.

```powershell
uv run athar-v2 verify models_v2                                   # v2 freeze (models + sources)
uv run --with openpyxl athar-data check-contrib EXPORT.xlsx --cards contrib/scenario_cards.csv `
    --manifest data/splits/manifest.json --reference data/athar.jsonl data/draft/athar_draft.jsonl `
    --out data/raw/private/contrib/kept.csv --report data/raw/private/contrib/screen.json
uv run athar-data import-contrib data/raw/private/contrib/kept.csv --cards contrib/scenario_cards.csv `
    --out data/raw/private/contrib/label.xlsx --multi-writer          # blind sheet; labels, then build
uv run athar-data lock data/natural/natural.jsonl --out data/natural/manifest.json --note "..."
uv run athar-v2 evaluate --natural data/natural/manifest.json --s8-traces T0 T1 T2 `
    --llm-results R.json ... --out results/natural/evaluation.json   # once
uv run python scripts/natural_tables.py                            # report tables -> results/natural/tables.md
uv run --with openpyxl python scripts/natural_sensitivity.py       # post hoc -> results/natural/sensitivity.md
uv run --with matplotlib python scripts/make_figures.py            # figures, incl. docs/figures/natural_routes.png
```

Status (2026-11-02): 247 messages from 17 writers locked (238 confirmatory); every system scored
once. N1 (S1 > S2) and N3 (S9 > S8) supported, N2 (S1 v1.1 > S1) not; see report section 8 and
`results/natural/`. The labelling sheets are private (`data/raw/private/`), so
`natural_sensitivity.py` runs only where they exist.

## Training and freezing (Phase 3)

```powershell
uv run athar-train fit --manifest data/splits/manifest.json --config configs/train.yaml --out models
uv run athar-train verify models     # model files, config and shared-core sources unchanged
```

Extended baselines (fitted post hoc, after both evaluations; exploratory, final report Table 4): S5 frozen
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

Complete: locked test (2026-10-18), pre-registered natural-phrasing evaluation (2026-11-02) and
the final report ([final_report/build_report.py](final_report/build_report.py)).
Protocol v3 ([docs/protocol_v3.md](docs/protocol_v3.md)), a third registered evaluation on 236 unused
contributor messages, is complete (2026-11-20): human training messages raised S3 by +0.089 and
MARBERTv2 by +0.088 STS (Holm p ≤ 0.002), to the rules' level. It is in the final report.

## Protocol v3: does human-written training data help? (gates G1-G5)

| Gate | State |
|---|---|
| G1 protocol registered | done (`3a69bc0`) |
| G2+G3 systems frozen, predictions committed | done for S3, S6 (5 seeds), the A4 control and the A5 GPU systems (A7, before labelling) |
| G4 two annotations, adjudication, lock | done: 20 items adjudicated jointly, 4 gold corrections (A8, A9), manifest `750e7847` |
| G5 one scoring run | done: `results/v3/evaluation.json`; GPU systems: `results/v3/evaluation_gpu.json` (`python -m athar.v3.gpu_score`) |

```powershell
uv run python -m athar.v3 verify                     # freezes, models, predictions, sources unchanged
uv run python -m athar.v3.amend                      # amendments file still append-only
# G4: each annotator fills data/raw/private/v3/v3_label_<name>.xlsx alone, then together:
uv run --with openpyxl python scripts/adjudicate_natural.py --set v3 sheet
uv run --with openpyxl python scripts/adjudicate_natural.py --set v3 apply
uv run --with openpyxl athar-data build data/raw/private/v3/v3_gold.xlsx --cards contrib/scenario_cards.csv --out data/raw/private/v3/v3.jsonl
uv run athar-data validate data/raw/private/v3/v3.jsonl --strict
uv run athar-data lock data/raw/private/v3/v3.jsonl --out data/v3/manifest.json --note "protocol v3, G4"
# G5, once:
uv run python -m athar.v3.evaluate --manifest data/v3/manifest.json --check
uv run python -m athar.v3.evaluate --manifest data/v3/manifest.json
```

The post hoc version of this question (leave one writer out on the natural set, report Table 5)
reproduces exactly from `scripts/posthoc_encoders_human.py human` with the cached encoders.
