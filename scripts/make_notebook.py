"""Generate notebooks/kaggle_s4_llm.ipynb, the GPU notebook for S4 / S7 / S8.

The notebook is generated, not edited by hand: change the cell sources here, then

    uv run python scripts/make_notebook.py notebooks/kaggle_s4_llm.ipynb
    uv run --extra neural --with pandas python scripts/smoke_kaggle_notebook.py [--split test]

The existing notebook is read for its metadata (kernel spec) and its cells are replaced.
The version string is NOTEBOOK_VERSION in the configuration cell.
"""
import json
import sys
from pathlib import Path

NB = Path(sys.argv[1])
nb = json.loads(NB.read_text(encoding="utf-8"))

MD0 = r"""# ATHAR Chat: LLM baselines (S4: Jais 3 / Jais 2 / Nile-Chat / Qwen3 · S7: Jais 2 + QLoRA · S8: Jais 3 tool calling)

**Why Jais 3 is here.** The proposal named Jais 2 8B as the Arabic LLM baseline. Jais 3 was released on
2026-10-15, after the models were frozen but before the test split was opened, so it is added beside Jais 2
(which stays the pre-registered model) and logged in the protocol (deviations log, 2026-10-15). See
`docs/jais3.md` for the design and what can and cannot be claimed.

| System | What | Status in the protocol |
|---|---|---|
| S4 | instruction-tuned LLM predicts **only the intent**; route, arguments and replies come from the shared, deterministic `athar` pipeline. Models: Jais-3-7B, Jais-3-3.7B, Jais 2 8B (proposal), Nile-Chat 4B (Egyptian), Qwen3 8B (multilingual) | extended, reported separately |
| S7 | Jais 2 8B + QLoRA fine-tuned on the train split | exploratory |
| S8 | **Jais-3-7B does the whole task through native tool calling** (`athar.agent`): it chooses `create_reminder` / `create_alarm` / `ask_clarification` / `decline` and fills the arguments itself. Same response schema, same Safe Task Success | exploratory (2026-10-15) |

**S4 scoring methods (all reported). `prior` is the primary S4 number for the locked test run** (protocol deviations log, 2026-10-11):

| Method | How it works |
|---|---|
| `raw` | constrained label scoring: summed log-probability of each Arabic verbalizer (`تذكير`, `منبه`, `غير مدعوم`) after the chat prompt, as in lm-evaluation-harness `multiple_choice` |
| `calibrated` | contextual calibration (Zhao et al. 2021): divide by the label probabilities the model gives to content-free inputs (secondary) |
| `prior` | **primary**: batch-style calibration (after Zhou et al. 2024): divide by the mean label distribution over 60 unlabeled **train** texts |
| `generate` | greedy generation, parsed; an unparseable answer counts as a failure |

**Chat templates.** `llm.chat_profile(model id)` drives each template. Jais 3 opens a thinking block in its
generation prompt and requires a thinking field on every assistant turn: its few-shot answers get an empty
thinking field and the label is scored after `</ifm|think>`, i.e. exactly where the template writes an answer
with no reasoning (tested against the pinned template). Qwen3 runs with thinking off. The other prompts are
byte-identical to the earlier runs. `RUN_PROMPT_CHECK` prints the exact prompt every model sees.

**Precision.** `auto:<fallback>` = **bf16** on GPUs with native bf16 (A100, L4, H100: the precision every model
was released in), else `<fallback>` on Kaggle T4s; fp16 that overflows is rerun in `nf4-fp32`. Every result
records the precision it used. For the Jais 2 vs Jais 3 comparison, bf16 hardware is preferred.

### Setup (once)
1. **Build the kit locally:** `uv run python scripts/build_kaggle_kit.py`, then upload `dist/athar-chat-kit.zip` as a
   new version of the Kaggle **Dataset** `athar-chat-kit`.
2. **Jais 2 is gated:** accept its terms at https://huggingface.co/inception42/Jais-2-8B-Chat. Jais 3 (Apache-2.0) is not gated.
3. **Token:** *Add-ons → Secrets* → `HF_TOKEN` (a read token). Off Kaggle: `export HF_TOKEN=...` and `ATHAR_KIT_ROOT=<unzipped kit>`.
4. **Notebook options:** *Accelerator* = **GPU T4 x2** (or better), *Internet* = **on**, attach `athar-chat-kit`.
5. Set the `RUN_*` flags for this run (table in the next cell), then **Save Version → Save & Run All** (a commit run keeps its output).

**Test-set discipline:** `SPLIT = "dev"`. The test split is opened only once, in the Phase 4 run, with `UNLOCK_TEST = True`."""

C1 = r'''# ---- configuration --------------------------------------------------------------------------
NOTEBOOK_VERSION = "2026-10-26a"   # printed below and saved in environment.json: check it in every run
print("ATHAR notebook version", NOTEBOOK_VERSION)
import os

# Kaggle mounts datasets at /kaggle/input/<slug>/ or /kaggle/input/datasets/<user>/<slug>/,
# sometimes one folder deeper: locate the kit by its wheel instead of hard-coding the path.
# Off Kaggle (e.g. an A100/H100 server), set ATHAR_KIT_ROOT to the unzipped kit.
ON_KAGGLE = os.path.isdir("/kaggle/input")
INPUT_ROOT = "/kaggle/input" if ON_KAGGLE else os.environ.get("ATHAR_KIT_ROOT", ".")
_wheels, _seen = [], []
for _root, _dirs, _files in os.walk(INPUT_ROOT, followlinks=True):   # dataset dirs can be symlinks
    _seen.append(f"{_root}: {sorted(_dirs + _files)[:8]}")
    _wheels += [os.path.join(_root, f) for f in _files if f.startswith("athar_chat-") and f.endswith(".whl")]
_wheels.sort()
if not _wheels:
    raise FileNotFoundError(f"athar_chat-*.whl not found under {INPUT_ROOT}. If you just uploaded a new "
                            "dataset version, wait until Kaggle finishes processing it, then in the right "
                            "panel remove and re-add athar-chat-kit (or pick its latest version) and "
                            "restart the session. Seen:\n" + "\n".join(_seen[:30]))
WHEEL = _wheels[-1]
KIT = os.path.dirname(WHEEL)
print("kit:", KIT)
MANIFEST = f"{KIT}/data/splits/manifest.json"   # the frozen split (test = 120 author-written items)
# items to score: the main manifest, or an evaluation-only set (all locked test) such as the
# natural-phrasing set: f"{KIT}/data/natural/manifest.json" with SPLIT = "test", UNLOCK_TEST = True
# (docs/protocol_natural.md). Exemplars, prior texts and S7 dev rescoring always use MANIFEST.
EVAL_MANIFEST = MANIFEST
SPLIT = "dev"                                     # "test" only in the Phase 4 run
UNLOCK_TEST = False
GATE_TAU = {}             # test run only: {system name: tau picked on dev}; empty on dev
SHOTS_PER_LABEL = 2                               # few-shot: 2 x 3 labels, from train only
OUT = "/kaggle/working/s4_results" if ON_KAGGLE else os.path.abspath("s4_results")

# (key, Hugging Face id, precision). "auto:<fallback>" = bf16 on GPUs with native bf16 (A100, L4, H100),
# else <fallback> (Kaggle T4s). How each chat template is driven: llm.chat_profile(model id).
MODELS = [
    ("jais3-7b",   "inception42/Jais-3-7B",      "auto:fp16"),   # Jais 3 (2026-10-15), 7B class (9.0B params)
    ("jais3-3.7b", "inception42/Jais-3-3.7B",    "auto:fp32"),   # Jais 3, size-matched to Nile-Chat 4B
    ("jais2-8b",   "inception42/Jais-2-8B-Chat", "auto:fp16"),   # the proposal's model (pre-registered)
    ("nile-4b",    "MBZUAI-Paris/Nile-Chat-4B",  "auto:fp32"),   # Egyptian; Gemma-3 overflows in fp16 on T4
    ("qwen3-8b",   "Qwen/Qwen3-8B",              "auto:fp16"),   # multilingual reference
]
RUN_MODELS = []           # [] = all of MODELS; or some keys, e.g. ["jais3-7b", "jais3-3.7b"], to split the work
# Model revisions are pinned, so a later upload to the Hub cannot change the results (or, for Jais 3,
# the remote model code). Update them only on purpose, and log it.
REVISIONS = {
    "inception42/Jais-3-7B":      "94d69f49b4af702d42fcee122cc06e493a6fe7e1",
    "inception42/Jais-3-3.7B":    "3d5754d5db87cd3a59f0f5b09c7ac39b351ff5ed",
    "inception42/Jais-2-8B-Chat": "da0e1639cd92b508b24120f7f77f5270a8465dc4",
    "MBZUAI-Paris/Nile-Chat-4B":  "2a365667659bbd6786c2760d868e37c506d76ce3",
    "Qwen/Qwen3-8B":              "b968826d9c46dd6066d109eabc6255188de91218",
}
PRIMARY = "prior"         # primary S4 scoring, frozen for the test run (protocol deviations log, 2026-10-11)
# Secondary S4 analysis, declared before the test split was opened (protocol section 6 and log, 2026-10-16):
# the primary setting with the exemplars listed in the user message instead of as chat turns.
S4_INLINE = True
INLINE_CONFIRM = ["jais2-8b", "jais3-7b"]   # the models whose dev result suggested it: Holm over these only

# ---- S8: the whole task through native tool calling (athar.agent) ----------------------------
S8_MODELS = [   # (key, Hugging Face id, precision, chat-template kwargs)
    ("jais3-7b", "inception42/Jais-3-7B", "auto:fp16", {"reasoning_effort": "high", "tool_call_format": "xml"}),
    # ("qwen3-8b", "Qwen/Qwen3-8B", "auto:fp16", {"enable_thinking": True}),   # optional reference
]
S8_SAMPLING = {"temperature": 0.7, "top_p": 0.95, "top_k": 20}   # Jais 3 card: reasoning, Arabic
S8_SEEDS = [0, 1, 2]        # sampled decoding: reported as mean +/- SD over seeds
S8_MAX_NEW_TOKENS = 8192    # reasoning + answer; an answer cut off at the limit scores as a failure (reported)
S8_BATCH = 4                # 2x T4: 4; one 80 GB H100: 16
S8_SERVER = None            # e.g. "http://localhost:8000/v1": a vLLM/SGLang server instead of generate()

# ---- S7: Jais 2 8B + QLoRA (protocol log 2026-10-17) ------------------------------------------
S7_SEEDS = [0, 1, 2]      # GPU fine-tuning is not bit-exact: report mean +/- SD over seeds
S7_RECIPE = {"epochs": 3, "learning_rate": 2e-4, "r": 16, "alpha": 32, "dropout": 0.05,   # pre-registered
             # optimisation = the pre-registered S6 small-data recipe (configs/train.yaml), not tuned:
             "batch_size": 16, "warmup_ratio": 0.1, "max_grad_norm": 1.0, "weight_decay": 0.01,
             "class_balanced": True}
S7_METHODS = ("raw", "calibrated", "prior")   # headline: PRIMARY ("prior"), the same rule as S4
S7_EGYPTIAN = True        # also the 192 native out-of-scope Egyptian requests (primary method, ~5 min/seed)
S7_SAVE_ADAPTERS = True   # LoRA weights per seed -> /kaggle/working/s7_adapters/ (outside s4_results)
# Frozen S7 (protocol log 2026-10-18): the run-6b adapters are loaded and hash-checked, not retrained.
# Attach a second Kaggle dataset with the files of runs/s4/2026-10-18-run6b-s7/s7_adapters/ (any folder
# layout or file names: each adapter is found by the SHA-256 of its content).
S7_FROZEN = True          # False = train new adapters (runs 6/6b)
S7_ADAPTER_SHA256 = {     # adapter_model.safetensors of run 6b (docs/jais3.md section 13)
    0: "64e87a48a6c24f6ff73cd4cfc62946ba28104f218aa9a63a74d196224bb6c62c",
    1: "30abde5994ea9b010f017b8c802a4c1fa19e1275456021d16de63c2d2a86735d",
    2: "8ada29076edce065ac0c6ac38c8d8d0ad9075ac447d740a6e9724b443d16c9ab",
}
S7_ADAPTER_CONFIG_SHA256 = "abdb6553a493d44b74022b12ac328a96de518273aa8cf0f30326bf30c258edfb"   # same for all seeds
S7_GATE_TAU = {           # tau the gate rule picked on dev in run 6b, for these exact weights
    **{f"S7-jais2-qlora-s{k}-{m}": 0.0 for k in (0, 1, 2) for m in ("raw", "calibrated", "prior")},
    "S7-jais2-qlora-s1-calibrated": 0.9,
}

# ---- what this run does: switch ON only what you need (each run is a commit, < 12 h) ----------
# | plan (docs/jais3.md, "Runs")       | flags                                                   | ~time on 2x T4 |
# | 1. prompt check (always first)      | RUN_PROMPT_CHECK only                                   | 10 min   |
# | 2. S4 Jais 3 on dev                 | RUN_S4, RUN_EGYPTIAN_OOD, RUN_MODELS=["jais3-7b","jais3-3.7b"] | 2-3 h |
# | 3. S4 others on dev (real split)    | RUN_S4, RUN_EGYPTIAN_OOD, RUN_MODELS=["jais2-8b","nile-4b","qwen3-8b"] | 3-4 h |
# | 4. Jais 2 prompt-format check       | RUN_SHOT_STYLE                                          | 1-2 h    |
# | 5. S8 Jais 3 tool calling on dev    | RUN_S8 (+ RUN_S8_EGYPTIAN if time allows)               | 1-4 h    |
# | 6. S7 on the real split (done 2026-10-17, old training loop; kept as the record)      | ~4 h     |
# | 6b. S7, S6-recipe training, dev only| RUN_S7_QLORA                                            | ~4.5 h   |
# | 7. locked test run: each system once, SPLIT="test", UNLOCK_TEST=True; 120 items,
# |    so split it into commits under the 12 h limit:                                                         |
# |    7a. RUN_S4, RUN_MODELS=["jais3-7b","jais3-3.7b"], GATE_TAU from dev                         | ~4 h     |
# |    7b. RUN_S4, RUN_MODELS=["jais2-8b","nile-4b","qwen3-8b"], GATE_TAU from dev                 | ~7 h     |
# |    7c. RUN_S7_QLORA with S7_FROZEN: loads the run-6b adapters (2nd dataset, SHA-256 checked),
# |        rescores dev as a reproduction check, then test, with the frozen S7_GATE_TAU           | ~1.5 h   |
# |    7d. RUN_S8                                                                                  | ~2.5 h   |
RUN_PROMPT_CHECK = True   # the exact prompt every model sees + verbalizer token counts (tokenizers only)
RUN_S4 = False            # S4: models x {0, 6 shots} x {raw, calibrated, prior, generate}
RUN_S7_QLORA = False      # S7: fine-tune Jais 2 (QLoRA) on the train split, S7_SEEDS (~85 min each)
RUN_S8 = False            # S8: Jais 3 tool calling, end to end, S8_SEEDS sampled runs
RUN_S8_EGYPTIAN = False   # S8 on the 192 native out-of-scope Egyptian requests (one seed; long)
RUN_MASSIVE_OOD = False   # out-of-distribution intent check on MASSIVE ar-SA
RUN_EGYPTIAN_OOD = False  # native-written Egyptian out-of-scope check (Alhasan/egyptian-nlu, CC BY 4.0)
PAIRED = [("jais2-8b", "jais3-7b"), ("nile-4b", "jais3-3.7b")]   # paired McNemar on the native out-of-scope items
RUN_SHOT_STYLE = False    # few-shot format check (turns / inline / Jais 2 old role), 3 exemplar sets
SHOT_STYLE_MODELS = ["jais2-8b", "jais3-7b"]   # Jais 2 also gets turns-old-role
RUN_SCORING_ABLATION = False # done 2026-10-10 (runs/s4/ablation): sum vs mean vs one-token answers
RUN_S4_DIAGNOSTICS = False   # done 2026-10-11/12 on the drafts (runs/s4/diagnostics); rerun only if asked
DIAG_MODELS = []          # [] = all of MODELS
DIAG_SETS = ("dev", "egyptian", "msa")'''

C2 = r'''!nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv
!pip install -q --no-deps --ignore-requires-python "{WHEEL}"
!pip install -q "transformers==5.19.0" accelerate bitsandbytes peft "tzdata>=2024.1" "pydantic>=2.7"'''

C3 = r'''import gc
import json
import random
import time
from pathlib import Path

import pandas as pd
import torch
import yaml
from huggingface_hub import HfApi, hf_hub_download, login
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

try:  # a Kaggle secret, else the environment (a GPU server)
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN")
except ImportError:
    HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)


# Gated models (Jais 2) need the terms accepted *and* the request approved on the model page, for the
# account that owns HF_TOKEN. Skip any model the token cannot read instead of failing mid-run.
def _can_read(model_id):
    try:
        hf_hub_download(model_id, "config.json", token=HF_TOKEN, revision=REVISIONS.get(model_id))
        return True
    except (GatedRepoError, RepositoryNotFoundError) as e:
        print(f"SKIPPING {model_id}: no access with this HF_TOKEN ({type(e).__name__}). "
              f"Request access at https://huggingface.co/{model_id} and rerun once approved.")
        return False


MODELS = [m for m in MODELS if _can_read(m[1])]
S8_MODELS = [m for m in S8_MODELS if _can_read(m[1])]
if RUN_S7_QLORA and not _can_read("inception42/Jais-2-8B-Chat"):
    RUN_S7_QLORA = False


def selected(models):
    return [m for m in models if not RUN_MODELS or m[0] in RUN_MODELS]


print("models:", [m[0] for m in MODELS], "| this run:", [m[0] for m in selected(MODELS)],
      "| S8:", [m[0] for m in S8_MODELS])

# Stop in the first minute if the kit dataset is older than this notebook (it would fail hours later).
from athar import CONTRACT_VERSION, agent
from athar.data.split import load_split, read_manifest
from athar.eval import harness as H
from athar.eval import ood as _ood
from athar.eval.stats import holm
from athar.intent import llm
from athar.intent.rules import KeywordIntent
from athar.normalize import norm

if not all((hasattr(llm, "answer_suffix"), hasattr(llm, "find_adapter"), hasattr(agent, "ToolCallingSystem"),
            hasattr(_ood, "load_pairs"))):
    raise RuntimeError("The athar-chat-kit dataset is older than this notebook. Upload the current "
                       "dist/athar-chat-kit.zip as a NEW VERSION of the dataset, select that version in "
                       "the right panel, and run again.")

CFG = yaml.safe_load(open(f"{KIT}/configs/eval.yaml", encoding="utf-8"))
CFG["oracle"] = False
CFG["comparisons"] = []          # S4/S7/S8 are reported separately: no tests against the core systems
train = load_split(MANIFEST, "train")
items = load_split(EVAL_MANIFEST, SPLIT, unlock_test=UNLOCK_TEST)
SHOTS = llm.select_shots(train, per_label=SHOTS_PER_LABEL, seed=CFG["seed"])
# label prior for method "prior": 60 unlabeled train requests, fixed seed (no dev/test information)
PRIOR_TEXTS = random.Random(CFG["seed"]).sample([norm(it.text) for it in train], 60)

# The locked test run scores only the pre-registered systems, once (protocol 5-6, log 2026-10-16/17).
if SPLIT == "test":
    _dev_only = [f for f in ("RUN_PROMPT_CHECK", "RUN_SHOT_STYLE", "RUN_SCORING_ABLATION", "RUN_S4_DIAGNOSTICS")
                 if globals()[f]]
    if _dev_only:
        raise RuntimeError(f"Test run: {_dev_only} are dev-only analyses. Switch them off.")
    _no_tau = [f"S4-{k}-{len(SHOTS)}shot-{PRIMARY}" for k, *_ in selected(MODELS)
               if RUN_S4 and f"S4-{k}-{len(SHOTS)}shot-{PRIMARY}" not in GATE_TAU]
    if _no_tau:
        raise RuntimeError(f"Test run: GATE_TAU needs the tau picked on dev for {_no_tau} "
                           "(gate_tau in the dev summary.csv). tau is never picked on test.")
    print("LOCKED TEST RUN:", len(items), "items. Each system is scored on test once; keep every output.")
Path(OUT).mkdir(parents=True, exist_ok=True)
import importlib.metadata as _md
import platform


def _version(pkg):
    try:
        return _md.version(pkg)
    except _md.PackageNotFoundError:
        return None


ENV = {"python": platform.python_version(),
       "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
       "compute_capability": [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())],
       "native_bf16": llm.native_bf16(),
       **{pkg: _version(pkg) for pkg in ("athar-chat", "torch", "transformers", "accelerate", "bitsandbytes", "peft")},
       "revisions": REVISIONS, "primary": PRIMARY, "notebook_version": NOTEBOOK_VERSION,
       "manifest": read_manifest(MANIFEST)["manifest_sha256"], "split": SPLIT,
       "eval_manifest": read_manifest(EVAL_MANIFEST)["manifest_sha256"]}
Path(f"{OUT}/environment.json").write_text(json.dumps(ENV, indent=1), encoding="utf-8")
print("environment:", ENV)
print(f"contract v{CONTRACT_VERSION}; manifest {ENV['manifest'][:12]}; "
      f"train {len(train)}, {SPLIT} {len(items)}; {len(SHOTS)} shots:", [lab for _, lab in SHOTS])'''

C4 = r'''FALLBACK = "nf4-fp32"   # 4-bit weights, fp32 compute: cannot overflow, fits an 8B model on 2x T4


def load_checked(model_id, precision):
    """Load at the resolved precision and check that a forward pass is finite; fp16 overflow -> FALLBACK."""
    prof = llm.chat_profile(model_id)
    used = llm.resolve_precision(precision)
    model, tok = llm.load_llm(model_id, used, token=HF_TOKEN, revision=REVISIONS.get(model_id),
                              trust_remote_code=prof.trust_remote_code)
    probe = llm.LLMIntent(model, tok, shots=SHOTS, **prof.intent_kwargs())
    try:
        probe.label_logprobs("فكرني بكره الساعه 7 الصبح اشتري عيش")
        return model, tok, used
    except FloatingPointError as e:
        print("  ", e, f"-> reloading in {FALLBACK}")
        del model, probe; gc.collect(); torch.cuda.empty_cache()
        model, tok = llm.load_llm(model_id, FALLBACK, token=HF_TOKEN, revision=REVISIONS.get(model_id),
                                  trust_remote_code=prof.trust_remote_code)
        return model, tok, FALLBACK


def build_recognizers(model, tok, key, model_id):
    prof = llm.chat_profile(model_id).intent_kwargs()
    recognizers = {}
    for shots_name, shots in (("0shot", []), (f"{len(SHOTS)}shot", SHOTS)):
        shared = None  # same prompt + shots -> one cache of label scores for all three methods
        for method in ("raw", "calibrated", "prior", "generate"):
            name = f"S4-{key}-{shots_name}-{method}"
            r = llm.LLMIntent(model, tok, name=name, shots=shots, method=method,
                              prior_texts=PRIOR_TEXTS if method == "prior" else (), **prof)
            if shared is None:
                shared = r
            else:
                r._cache, r._cf = shared._cache, None
            recognizers[name] = r
    if S4_INLINE:   # secondary: the primary setting, exemplars listed in the user message
        name = f"S4-{key}-{len(SHOTS)}shot-inline-{PRIMARY}"
        recognizers[name] = llm.LLMIntent(model, tok, name=name, shots=SHOTS, method=PRIMARY,
                                          prior_texts=PRIOR_TEXTS if PRIMARY == "prior" else (),
                                          **{**prof, "shot_style": "inline"})
    return recognizers


def inline_vs_turns(key, systems):
    """Inline vs chat-turn exemplars (same shots, same method): the H1-H3 paired test (family-level
    approximate randomization on STS) with exact McNemar beside it (protocol 5 and 6)."""
    turns, inline = f"S4-{key}-{len(SHOTS)}shot-{PRIMARY}", f"S4-{key}-{len(SHOTS)}shot-inline-{PRIMARY}"
    if turns not in systems or inline not in systems:
        return None
    assert [it["id"] for it in systems[inline]["items"]] == [it["id"] for it in systems[turns]["items"]]
    t = next(iter(H.compare(systems, [(inline, turns)], CFG).values()))
    m = systems[inline]["metrics"]["safe_task_success"], systems[turns]["metrics"]["safe_task_success"]
    return {"model": key, "n": len(systems[inline]["items"]), "STS_inline": m[0], "STS_turns": m[1],
            "difference": t["difference"], "diff_low": t["ci"]["low"], "diff_high": t["ci"]["high"],
            "p_value": t["p_value"], "mcnemar_only_inline": t["mcnemar"]["only_a"],
            "mcnemar_only_turns": t["mcnemar"]["only_b"], "mcnemar_p": t["mcnemar"]["p_value"],
            "family": "confirmatory" if key in INLINE_CONFIRM else "descriptive"}


def summarise(name, rep):
    rep = {**rep, "name": name}
    m, ci = rep["metrics"], rep["ci"]["safe_task_success"]
    cal = rep.get("calibration") or {}
    pm = rep["subgroups"]["domain"]["project_management"]
    return {"system": name, "STS": round(m["safe_task_success"], 3), "STS_low": round(ci["low"], 3),
            "STS_high": round(ci["high"], 3), "intent_F1": round(m["intent_macro_f1"], 3),
            "unsafe": round(m["unsafe_execution_rate"], 3), "invalid": m["invalid_outputs"],
            "ECE": round(cal.get("ece", float("nan")), 3),
            # project managers are the main target users: their requests scored on their own
            "STS_PM": round(pm["safe_task_success"]["estimate"], 3),
            "F1_PM": round(pm["intent_macro_f1"]["estimate"], 3),
            **gate_cols(rep)}


def gate_cols(rep):
    # confidence gate (protocol 6): the tau the dev rule picks, and STS / unsafe execution there.
    # On dev this is the selection itself; for the test run, fix GATE_TAU from the dev run.
    g = rep.get("gate")
    if g is None:
        return {}
    if SPLIT == "test" and rep.get("name") not in GATE_TAU:
        return {}   # never pick tau on test: only systems given a dev tau get gated columns
    tau = GATE_TAU.get(rep.get("name"), g["rule_threshold_on_this_split"])
    row = next(r for r in g["curve"] if r["tau"] == tau)
    return {"gate_tau": tau, "STS_gated": round(row["safe_task_success"], 3),
            "unsafe_gated": round(row["unsafe_execution_rate"], 3), "asks_confirm": round(row["gated"], 3)}


rows, rows_inline = [], []
for key, model_id, precision in (selected(MODELS) if RUN_S4 else []):
    t0 = time.time()
    model, tok, used = load_checked(model_id, precision)
    sha = REVISIONS.get(model_id) or HfApi().model_info(model_id, token=HF_TOKEN).sha
    try:
        recognizers = build_recognizers(model, tok, key, model_id)
        report = H.evaluate(recognizers, items, CFG)
    except FloatingPointError as e:
        # The probe passed but a later input overflowed: rerun the whole model in the fallback,
        # so every number for this model comes from one precision.
        if used == FALLBACK:
            raise
        print("  ", e, f"-> rerunning {key} in {FALLBACK}")
        recognizers = None; del model; gc.collect(); torch.cuda.empty_cache()
        model, tok = llm.load_llm(model_id, FALLBACK, token=HF_TOKEN, revision=REVISIONS.get(model_id),
                                  trust_remote_code=llm.chat_profile(model_id).trust_remote_code)
        used = FALLBACK
        recognizers = build_recognizers(model, tok, key, model_id)
        report = H.evaluate(recognizers, items, CFG)
    report["model"] = {"id": model_id, "revision": sha, "precision": used, "split": SPLIT,
                       "chat_profile": llm.chat_profile(model_id).intent_kwargs(),
                       "minutes": round((time.time() - t0) / 60, 1)}
    Path(f"{OUT}/{key}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    rows += [{**summarise(n, r), "precision": used} for n, r in report["systems"].items()]
    print(pd.DataFrame(rows).tail(len(recognizers)).to_string(index=False))
    pair = inline_vs_turns(key, report["systems"])
    if pair:
        rows_inline.append({**pair, "precision": used})
    del model, recognizers; gc.collect(); torch.cuda.empty_cache()

summary = pd.DataFrame(rows)
if RUN_S4:
    summary.to_csv(f"{OUT}/summary.csv", index=False)
if rows_inline:
    # Holm over the confirmatory models present in this run; a split test run (7a/7b) adjusts
    # over both parts offline, with the same holm() on the saved p-values.
    conf = {r["model"]: r["p_value"] for r in rows_inline if r["family"] == "confirmatory"}
    adj = holm(conf) if conf else {}
    for r in rows_inline:
        r["p_holm_this_run"] = adj.get(r["model"])
    pd.DataFrame(rows_inline).round(4).to_csv(f"{OUT}/s4_inline_vs_turns.csv", index=False)
    print(pd.DataFrame(rows_inline).round(4).to_string(index=False))
summary'''

MD5 = r"""## Prompt check: the exact text each model scores (run first, CPU only)

For every model this writes `prompt_<key>.txt` (the 6-shot prompt for one dev request, up to the point where
the label is scored) and `verbalizer_tokens.csv` (tokens per label answer: the summed score favours short
answers, protocol log 2026-10-09; Jais 3 has a new 250k tokenizer). For S8 it writes `prompt_s8_<key>.txt`
(the system message with the four tools as the Jais 3 template renders them).

`answer_consistent` checks, on each model's real template, that "prompt + label" is exactly how the template
writes a finished answer, so the few-shot answers look like the model's own turns and the label is scored where an
answer belongs. The 2026-10-15a check found that Jais 2 answers under the role `ai` while its few-shot answers were
sent as `assistant`; its chat profile now sends them as `ai` (protocol log 2026-10-15). Every row must be `True`."""

C6 = r'''if RUN_PROMPT_CHECK:
    example = norm(items[0].text)
    lengths = []
    for key, model_id, _ in MODELS:
        prof = llm.chat_profile(model_id)
        tok = llm.load_tokenizer(model_id, token=HF_TOKEN, revision=REVISIONS.get(model_id),
                                 trust_remote_code=prof.trust_remote_code)
        p = llm.LLMIntent(None, tok, shots=SHOTS, **prof.intent_kwargs()).prompt(example)
        Path(f"{OUT}/prompt_{key}.txt").write_text(p, encoding="utf-8")
        lengths.append({"model": key, "answer_consistent": llm.answer_consistent(tok, prof),
                        "answer_role": prof.assistant_role, **llm.verbalizer_lengths(tok),
                        "prompt_tokens": len(tok(p, add_special_tokens=False)["input_ids"])})
        print(f"===== {key} ({model_id}) =====\n{p}\n")
        if key.startswith("jais2"):
            print(f"----- {key} chat template -----\n{tok.chat_template}\n")
    for key, model_id, _, tkw in S8_MODELS:
        tok = llm.load_tokenizer(model_id, token=HF_TOKEN, revision=REVISIONS.get(model_id),
                                 trust_remote_code=llm.chat_profile(model_id).trust_remote_code)
        p = agent.HFToolBackend(None, tok, template_kwargs=tkw).render(
            agent.build_messages(items[0].text, items[0].ref_datetime, items[0].timezone))
        Path(f"{OUT}/prompt_s8_{key}.txt").write_text(p, encoding="utf-8")
        print(f"===== S8 {key}: {len(tok(p, add_special_tokens=False)['input_ids'])} prompt tokens; last 600 chars =====\n{p[-600:]}\n")
    lengths = pd.DataFrame(lengths)
    lengths.to_csv(f"{OUT}/verbalizer_tokens.csv", index=False)
    print(lengths.to_string(index=False))
    if not lengths.answer_consistent.all():
        print("WARNING: a model scores its label somewhere its template does not put an answer:",
              list(lengths[~lengths.answer_consistent].model), "- do not run S4 for it before this is fixed")'''

MD7 = r"""## S7 (exploratory): Jais 2 8B + QLoRA on the train split

The model is fine-tuned on the zero-shot prompt → verbalizer task, with the loss on the answer tokens only. Each answer
ends with the template's own end of turn (`llm.answer_suffix`; Jais 2: `<|eot_id|>`). Pre-registered adapter settings:
4-bit NF4 (fp32 compute), LoRA r = 16, α = 32, dropout 0.05 on all linear layers, LR 2e-4, 3 epochs. The optimisation
follows the protocol's pre-registered S6 small-data recipe (protocol log 2026-10-17): batches of 16 (gradient
accumulation), 10% linear warm-up then linear decay, gradient clipping at 1.0, weight decay 0.01, class-balanced loss.
Run 6 used one step per item at a constant LR, and its `raw` scores collapsed in 2 of 3 seeds (`docs/jais3.md` section 12).
There is no dev search.

Scored like S4 on the zero-shot prompt: `raw`, `calibrated` and `prior`. The headline is `prior`, the same rule as the
S4 primary. GPU fine-tuning is not bit-exact, so results are reported as mean ± SD over `S7_SEEDS`.

**Frozen (`S7_FROZEN = True`, protocol log 2026-10-18):** the adapters trained in run 6b are loaded from a second
Kaggle dataset. Each is found by the SHA-256 of its content (`llm.find_adapter`, so the upload's folder layout and file
names do not matter) and refused unless it matches `S7_ADAPTER_SHA256`. Dev is rescored as a reproduction check,
then test is scored with the τ that run 6b picked on dev (`S7_GATE_TAU`). Dev numbers, τ and test numbers therefore all
describe the weights that were frozen before the test split was opened. Outputs:

| file | content |
|---|---|
| `s7_seeds.csv` | one row per seed × method × split |
| `s7_training.csv` | loss, learning rate and gradient norms per epoch and seed (training runs only) |
| `s7_jais2_qlora_s<seed>_<split>.json` | full report (CIs, robustness, calibration, gate, items) |
| `egyptian_ood_s7.csv`, `egyptian_pred_s7_s<seed>.csv` | out-of-scope recall on the 192 native requests |
| `/kaggle/working/s7_adapters/s<seed>/` | the LoRA weights (download them too) |"""

C8 = r'''if RUN_S7_QLORA:
    import numpy as np

    from athar.eval import ood as OOD

    S7_ID = "inception42/Jais-2-8B-Chat"
    prof7 = llm.chat_profile(S7_ID)
    dev_items = items if SPLIT == "dev" else load_split(MANIFEST, "dev")
    ADAPTERS = "/kaggle/working/s7_adapters" if ON_KAGGLE else os.path.abspath("s7_adapters")
    if S7_FROZEN:   # find every frozen adapter by content now, so a missing dataset fails in the first minute
        FROZEN = {}
        for seed in S7_SEEDS:
            try:
                FROZEN[seed] = llm.find_adapter(INPUT_ROOT, S7_ADAPTER_SHA256[seed], S7_ADAPTER_CONFIG_SHA256,
                                                f"{ADAPTERS}_frozen/s{seed}")
            except FileNotFoundError as e:
                raise FileNotFoundError(
                    f"S7_FROZEN, seed {seed}: {e} -- Attach the Kaggle dataset made from "
                    "runs/s4/2026-10-18-run6b-s7/s7_adapters/ (right panel: Add Input) and wait until it shows as "
                    "ready. Datasets attached now: " + str(sorted(os.listdir(INPUT_ROOT)))) from None
            print(f"frozen S7 adapter, seed {seed}:", FROZEN[seed])
    if S7_EGYPTIAN and not (S7_FROZEN and SPLIT == "test"):
        rows_egy7 = OOD.load(f"{OUT}/egy_test.jsonl")          # pinned file, downloaded once
        texts_egy7 = [norm(x["text"]) for x in rows_egy7]
    s7_rows, s7_train, s7_egy = [], [], []
    for seed in S7_SEEDS:
        t0 = time.time()
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        base, tok = llm.load_llm(S7_ID, "nf4-fp32", token=HF_TOKEN, revision=REVISIONS[S7_ID])
        if S7_FROZEN:   # the run-6b weights, refused if the file is not the frozen one
            history = None
            tuned = llm.load_lora(base, FROZEN[seed], sha256=S7_ADAPTER_SHA256[seed])
        else:
            history = []
            tuned = llm.lora_finetune(base, tok, train, profile=prof7, seed=seed, history=history, **S7_RECIPE)
            s7_train += [{"seed": seed, **h} for h in history]
            pd.DataFrame(s7_train).to_csv(f"{OUT}/s7_training.csv", index=False)   # saved as we go
            if S7_SAVE_ADAPTERS:
                tuned.save_pretrained(f"{ADAPTERS}/s{seed}")
        s7, shared = {}, None
        for m in S7_METHODS:
            name = f"S7-jais2-qlora-s{seed}-{m}"
            r = llm.LLMIntent(tuned, tok, name=name, method=m, prior_texts=PRIOR_TEXTS if m == "prior" else (),
                              **prof7.intent_kwargs())
            if shared is None:
                shared = r
            else:
                r._cache = shared._cache   # same prompt, same weights -> one cache of label scores
            s7[name] = r
        meta = {"id": S7_ID, "adapter": "QLoRA", "recipe": S7_RECIPE, "precision": "nf4-fp32",
                "revision": REVISIONS[S7_ID], "seed": seed, "training": history,
                "frozen": {"from": "run 6b (2026-10-18)", "adapter_sha256": S7_ADAPTER_SHA256[seed]} if S7_FROZEN else None,
                "train_minutes": round((time.time() - t0) / 60, 1)}
        for split_name, split_items in [("dev", dev_items)] + ([("test", items)] if SPLIT == "test" else []):
            rep7 = H.evaluate(s7, split_items, CFG)
            rep7["model"] = {**meta, "split": split_name}
            if split_name == "dev":   # tau for these exact weights, picked on dev (applied to test below)
                picked = {n: s["gate"]["rule_threshold_on_this_split"] for n, s in rep7["systems"].items() if s.get("gate")}
                if S7_FROZEN:   # frozen with the weights in run 6b; the dev rescoring only reproduces it
                    for n, t in picked.items():
                        if t != S7_GATE_TAU[n]:
                            print(f"NOTE {n}: dev rescoring picks tau {t}, frozen tau {S7_GATE_TAU[n]} is used")
                    picked = {n: S7_GATE_TAU[n] for n in picked}
                GATE_TAU.update(picked)
            Path(f"{OUT}/s7_jais2_qlora_s{seed}_{split_name}.json").write_text(
                json.dumps(rep7, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            s7_rows += [{"seed": seed, "split": split_name, **summarise(n, r)} for n, r in rep7["systems"].items()]
        if S7_EGYPTIAN and not (S7_FROZEN and SPLIT == "test"):   # frozen: already scored in run 6b
            r = s7[f"S7-jais2-qlora-s{seed}-{PRIMARY}"]
            res = OOD.evaluate(r, rows_egy7)
            s7_egy.append({"seed": seed, "system": r.name, **{k: v for k, v in res.items() if k != "errors"}})
            pd.DataFrame({"id": [x["id"] for x in rows_egy7], "gold": [x["gold"] for x in rows_egy7],
                          "pred": r.predict(texts_egy7)}).to_csv(f"{OUT}/egyptian_pred_s7_s{seed}.csv", index=False)
        pd.DataFrame(s7_rows).to_csv(f"{OUT}/s7_seeds.csv", index=False)
        _seed_rows = pd.DataFrame(s7_rows)
        print(_seed_rows[_seed_rows.seed == seed].to_string(index=False))
        print(f"seed {seed}: {(time.time() - t0) / 60:.1f} min")
        del base, tuned, s7, shared; gc.collect(); torch.cuda.empty_cache()   # free the GPUs for the next seed
    if s7_egy:
        pd.DataFrame(s7_egy).round(3).to_csv(f"{OUT}/egyptian_ood_s7.csv", index=False)
        print(pd.DataFrame(s7_egy).round(3).to_string(index=False))
    s7_df = pd.DataFrame(s7_rows)
    s7_df.assign(method=s7_df.system.str.rsplit("-", n=1).str[-1]).groupby(["split", "method"])[
        ["STS", "intent_F1", "unsafe", "STS_PM"]].agg(["mean", "std"]).round(3)'''

MD9 = r"""## S8 (exploratory): Jais 3 does the whole task through native tool calling

S1–S7 only choose the intent; S8 asks how safely a tool-calling LLM routes Egyptian requests **on its own**.
`athar.agent` gives Jais-3-7B the request, the moment it was sent (weekday, date, time, Africa/Cairo) and a summary
of the task contract, with four tools: `create_reminder(datetime, reminder_text)`, `create_alarm(datetime)`,
`ask_clarification(tool, missing)` and `decline(reason)`. The call becomes the same response every system emits
and is scored with the same Safe Task Success; no call, several calls or invalid arguments score as failures, and
the cause is kept (`s8_trace_*.jsonl`, `s8_diagnostics.csv`: invalid outputs by cause, truncation, decline-reason
accuracy, executions with the right moment but a different reminder text).

Decoding follows the Jais 3 card (reasoning effort `high`; temperature 0.7, top-p 0.95, top-k 20 for Arabic), so
each request is answered once per seed in `S8_SEEDS` and S8 is reported as mean ± SD over seeds. Robustness
perturbations are not run for S8 (they would multiply the generation cost by ten). With `S8_SERVER` set, the
requests go to a vLLM/SGLang server started with the model's `k2_horizon` reasoning and tool parsers (the card's
recommended serving; much faster on an H100)."""

C10 = r'''if RUN_S8:
    from athar.eval import ood as OOD

    s8_rows, s8_diag = [], []
    cfg_s8 = {**CFG, "perturbations": []}    # robustness would multiply the generation cost by ten
    for key, model_id, precision, tkw in S8_MODELS:
        t0 = time.time()
        model = tok = None
        used = "server" if S8_SERVER else llm.resolve_precision(precision)
        if not S8_SERVER:
            model, tok = llm.load_llm(model_id, used, token=HF_TOKEN, revision=REVISIONS.get(model_id),
                                      trust_remote_code=llm.chat_profile(model_id).trust_remote_code)
        systems = {}
        for seed in S8_SEEDS:
            if S8_SERVER:
                backend = agent.OpenAIToolBackend(S8_SERVER, model_id, chat_template_kwargs=tkw,
                                                  max_tokens=S8_MAX_NEW_TOKENS, sampling=S8_SAMPLING, seed=seed)
            else:
                backend = agent.HFToolBackend(model, tok, template_kwargs=tkw, max_new_tokens=S8_MAX_NEW_TOKENS,
                                              sampling=S8_SAMPLING, seed=seed, batch_size=S8_BATCH)
            systems[f"S8-{key}-s{seed}"] = agent.ToolCallingSystem(backend, name=f"S8-{key}-s{seed}")
        report = H.evaluate(systems, items, cfg_s8)
        for name, system in systems.items():
            pd.DataFrame(system.trace_rows()).to_json(f"{OUT}/s8_trace_{name}.jsonl", orient="records", lines=True,
                                                      force_ascii=False)
            s8_diag.append({"system": name, **agent.diagnose(items, system.outputs, system.trace)})
        report["model"] = {"id": model_id, "revision": REVISIONS.get(model_id), "precision": used, "split": SPLIT,
                           "template_kwargs": tkw, "sampling": S8_SAMPLING, "seeds": S8_SEEDS,
                           "max_new_tokens": S8_MAX_NEW_TOKENS, "backend": "server" if S8_SERVER else "transformers",
                           "minutes": round((time.time() - t0) / 60, 1)}
        Path(f"{OUT}/s8_{key}.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        s8_rows += [{**summarise(n, r), "precision": used} for n, r in report["systems"].items()]
        for base, s in report["seed_summaries"].items():
            sts = s["safe_task_success"]
            print(f"{base}: STS {sts['mean']:.3f} +/- {sts['sd']:.3f} over {len(s['seeds'])} seeds "
                  f"(unsafe {s['unsafe_execution_rate']['mean']:.3f})")
        if RUN_S8_EGYPTIAN:   # native out-of-scope Egyptian requests, first seed only (long: reasoning per item)
            rows_egy = OOD.load(f"{OUT}/egy_test.jsonl")
            first = systems[f"S8-{key}-s{S8_SEEDS[0]}"]
            first.trace = []
            res = OOD.evaluate(first, rows_egy)
            Path(f"{OUT}/egyptian_ood_s8_{key}.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"S8 {key}: native out-of-scope recall {res['recall_unsupported']:.3f} "
                  f"{tuple(round(x, 3) for x in res['recall_unsupported_ci95'])}")
        del model, systems; gc.collect(); torch.cuda.empty_cache()
    pd.DataFrame(s8_diag).to_csv(f"{OUT}/s8_diagnostics.csv", index=False)
    s8_summary = pd.DataFrame(s8_rows)
    s8_summary.to_csv(f"{OUT}/s8_summary.csv", index=False)
    print(pd.DataFrame(s8_diag).to_string(index=False))
    s8_summary'''

MD11 = r"""## Out-of-distribution check: MASSIVE ar-SA (intent only)

MASSIVE (Amazon, CC BY 4.0) is Saudi-localised MSA, not Egyptian, and it has no datetime gold, so this check uses
**intent labels only** and is reported separately. Mapping: `alarm_set` → `create_alarm`; `alarm_remove`,
`calendar_remove` → `unsupported` (deletion); `weather_query`, `play_music`, `email_sendemail`, `calendar_query` →
`unsupported` (other tool); `calendar_set` **excluded** (it mixes one-off reminders with calendar events). A
balanced sample of up to 100 per mapped label is used (seed 0). S1 (rules) is included as a reference."""

C12 = r'''if RUN_MASSIVE_OOD:
    from datasets import load_dataset
    from sklearn.metrics import f1_score

    ds = load_dataset("mteb/amazon_massive_intent", "ar", split="test")
    keep = {"alarm_set": "create_alarm", "alarm_remove": "unsupported", "calendar_remove": "unsupported",
            "weather_query": "unsupported", "play_music": "unsupported", "email_sendemail": "unsupported",
            "calendar_query": "unsupported"}
    label_col = "label_text" if "label_text" in ds.column_names else "label"
    df = pd.DataFrame({"text": ds["text"], "intent": ds[label_col]})
    df = df[df.intent.isin(keep)].assign(gold=lambda d: d.intent.map(keep))
    df = pd.concat([g.sample(min(len(g), 100), random_state=0) for _, g in df.groupby("gold")])
    texts, gold = [norm(t) for t in df.text], list(df.gold)
    ood = [{"system": "S1", "macro_F1": f1_score(gold, KeywordIntent().predict(texts), average="macro")}]
    for key, model_id, precision in selected(MODELS):
        model, tok, used = load_checked(model_id, precision)
        r = llm.LLMIntent(model, tok, name=key, shots=SHOTS, method=PRIMARY,
                          prior_texts=PRIOR_TEXTS if PRIMARY == "prior" else (),
                          **llm.chat_profile(model_id).intent_kwargs())
        ood.append({"system": f"S4-{key}-{len(SHOTS)}shot-{PRIMARY}", "precision": used,
                    "macro_F1": f1_score(gold, r.predict(texts), average="macro")})
        del model, r; gc.collect(); torch.cuda.empty_cache()
    ood = pd.DataFrame(ood).round(3)
    ood.to_csv(f"{OUT}/massive_ood.csv", index=False)
    print(f"MASSIVE ar test, {len(df)} items:", df.gold.value_counts().to_dict())
    ood'''

MD13 = r"""## Out-of-distribution check: native-written Egyptian (Alhasan/egyptian-nlu)

The 200-item test split of `Alhasan/egyptian-nlu` (CC BY 4.0, derived from MASSIVE) was written and annotated by a
native Egyptian speaker; revision and SHA-256 are pinned in `athar.eval.ood`. Same mapping as MASSIVE: 196 items,
only 4 of which set an alarm, so the main number is **out-of-scope recall** (n = 192): does the model abstain on real
out-of-scope Egyptian requests? `caught@0.6` / `caught@0.7`: how many wrong in-scope predictions the confidence gate
would stop. **Paired comparisons** (`PAIRED`, e.g. Jais 2 vs Jais 3): exact McNemar on the same 192 items
(`egyptian_paired.csv`). These items informed the choice of the S4 method (protocol log 2026-10-10/11), so they are
supporting evidence; the locked author-written test split is the unbiased one."""

C14 = r'''if RUN_EGYPTIAN_OOD:
    from athar.eval import ood as OOD
    from athar.eval.stats import mcnemar_exact
    from athar.gate import confidences

    rows_egy = OOD.load(f"{OUT}/egy_test.jsonl")          # downloads the pinned file once
    texts_egy = [norm(r["text"]) for r in rows_egy]
    egy = [{"system": "S1", **{k: v for k, v in OOD.evaluate(KeywordIntent(), rows_egy).items() if k != "errors"}}]
    egy_pred = {}
    for key, model_id, precision in selected(MODELS):
        model, tok, used = load_checked(model_id, precision)
        r = llm.LLMIntent(model, tok, name=key, shots=SHOTS, method=PRIMARY,
                          prior_texts=PRIOR_TEXTS if PRIMARY == "prior" else (),
                          **llm.chat_profile(model_id).intent_kwargs())
        res = OOD.evaluate(r, rows_egy)
        pred = r.predict(texts_egy)
        egy_pred[key] = pred
        conf = confidences(r, texts_egy, pred)
        wrong = [c for c, p, x in zip(conf, pred, rows_egy, strict=True) if x["gold"] == "unsupported" and p != "unsupported"]
        egy.append({"system": f"S4-{key}-{len(SHOTS)}shot-{PRIMARY}", "precision": used,
                    **{k: v for k, v in res.items() if k != "errors"},
                    "caught@0.6": sum(c < 0.6 for c in wrong), "caught@0.7": sum(c < 0.7 for c in wrong)})
        Path(f"{OUT}/egyptian_ood_{key}_errors.json").write_text(json.dumps(res["errors"], ensure_ascii=False, indent=1), encoding="utf-8")
        pd.DataFrame({"id": [x["id"] for x in rows_egy], "gold": [x["gold"] for x in rows_egy], "pred": pred}).to_csv(
            f"{OUT}/egyptian_pred_{key}.csv", index=False)
        del model, r; gc.collect(); torch.cuda.empty_cache()
    egy = pd.DataFrame(egy).round(3)
    egy.to_csv(f"{OUT}/egyptian_ood.csv", index=False)
    oos = [i for i, x in enumerate(rows_egy) if x["gold"] == "unsupported"]
    paired = []
    for a, b in PAIRED:
        if a in egy_pred and b in egy_pred:
            sa = [egy_pred[a][i] == "unsupported" for i in oos]
            sb = [egy_pred[b][i] == "unsupported" for i in oos]
            paired.append({"a": a, "b": b, "n": len(oos), "recall_a": sum(sa) / len(oos),
                           "recall_b": sum(sb) / len(oos), **mcnemar_exact(sa, sb)})
    if paired:
        pd.DataFrame(paired).round(4).to_csv(f"{OUT}/egyptian_paired.csv", index=False)
        print(pd.DataFrame(paired).round(4).to_string(index=False))
    print(OOD.ATTRIBUTION)
    egy'''

MD15 = r"""## Prompt-format check: does the role mismatch explain Jais 2's few-shot instability?

On the drafts, Jais 2's dev STS ranged from 0.375 to 0.85 across exemplar sets, and few-shot prompts hurt it while
they helped the other models. Those runs sent the few-shot answers as `assistant` while Jais 2 answers as `ai`. The
same three exemplar sets, scored with the primary method, in three formats:

| format | few-shot answers |
|---|---|
| `turns` | earlier chat turns under the model's own answer role (Jais 2: `ai`): **the S4 format from 2026-10-15** |
| `turns-old-role` | Jais 2 only: earlier chat turns under `assistant`, the format of the runs up to 2026-10-12 |
| `inline` | listed in the user message: no assistant turns at all |

Dev only. The fix was decided from the template before any of this was run; this run measures how much the
mismatch mattered, it does not choose the format (protocol log 2026-10-15)."""

C16 = r'''if RUN_SHOT_STYLE:
    cfg_fast = {**CFG, "perturbations": [], "oracle": False}
    rows_ss = []
    for key, model_id, precision in MODELS:
        if key not in SHOT_STYLE_MODELS:
            continue
        t0 = time.time()
        model, tok, used = load_checked(model_id, precision)
        prof = llm.chat_profile(model_id).intent_kwargs()
        formats = {"turns": {}, "inline": {"shot_style": "inline"}}
        if prof["assistant_role"] != "assistant":
            formats["turns-old-role"] = {"assistant_role": "assistant"}
        for k in range(3):
            shots = llm.select_shots(train, per_label=SHOTS_PER_LABEL, seed=CFG["seed"] + k)
            for style, change in formats.items():
                name = f"S4-{key}-shotseed{k}-{style}-{PRIMARY}"
                r = llm.LLMIntent(model, tok, name=name, shots=shots, method=PRIMARY,
                                  prior_texts=PRIOR_TEXTS, **{**prof, **change})
                rep = H.evaluate({name: r}, items, cfg_fast)["systems"][name]
                m = rep["metrics"]
                rows_ss.append({"model": key, "shot_seed": k, "shot_style": style, "precision": used,
                                "STS": round(m["safe_task_success"], 3), "intent_F1": round(m["intent_macro_f1"], 3),
                                "unsafe": round(m["unsafe_execution_rate"], 3),
                                "success": [it["success"] for it in rep["items"]]})
        print(f"{key}: {(time.time() - t0) / 60:.1f} min")
        del model, r; gc.collect(); torch.cuda.empty_cache()
    ss = pd.DataFrame(rows_ss)
    ss.to_csv(f"{OUT}/shot_style.csv", index=False)
    ss.groupby(["model", "shot_style"])[["STS", "intent_F1", "unsafe"]].agg(["mean", "std"]).round(3)'''

MD17 = r"""## Scoring ablation (done 2026-10-10): is the summed score biased against `غير مدعوم`?

Scores the 6-shot prompt three ways (`sum`, `mean` per token, one-token `number` answers), each `raw` and
`calibrated`, on dev, native Egyptian out-of-scope recall and MASSIVE macro-F1. Logged in the protocol
(2026-10-09/10); kept for reproducibility."""

C18 = r'''if RUN_SCORING_ABLATION:
    from datasets import load_dataset
    from sklearn.metrics import f1_score

    from athar.eval import ood as OOD

    egy_rows = OOD.load(f"{OUT}/egy_test.jsonl")
    ds = load_dataset("mteb/amazon_massive_intent", "ar", split="test")
    keep = {"alarm_set": "create_alarm", "alarm_remove": "unsupported", "calendar_remove": "unsupported",
            "weather_query": "unsupported", "play_music": "unsupported", "email_sendemail": "unsupported",
            "calendar_query": "unsupported"}
    label_col = "label_text" if "label_text" in ds.column_names else "label"
    mdf = pd.DataFrame({"text": ds["text"], "intent": ds[label_col]})
    mdf = mdf[mdf.intent.isin(keep)].assign(gold=lambda d: d.intent.map(keep))
    mdf = pd.concat([g.sample(min(len(g), 100), random_state=0) for _, g in mdf.groupby("gold")])
    m_texts, m_gold = [norm(t) for t in mdf.text], list(mdf.gold)
    cfg_fast = {**CFG, "perturbations": []}
    variants = {"sum": {}, "mean": {"length_norm": True}, "number": {"answer_format": "number"}}

    abl = []
    for key, model_id, precision in selected(MODELS):
        t0 = time.time()
        model, tok, used = load_checked(model_id, precision)
        prof = llm.chat_profile(model_id).intent_kwargs()
        caches = {}   # one set of forward passes per answer format; sum and mean share it
        for vname, kw in variants.items():
            for method in ("raw", "calibrated"):
                name = f"S4-{key}-{len(SHOTS)}shot-{method}-{vname}"
                r = llm.LLMIntent(model, tok, name=name, shots=SHOTS, method=method, **prof, **kw)
                r._cache = caches.setdefault(kw.get("answer_format", "word"), {})
                rep = H.evaluate({name: r}, items, cfg_fast)["systems"][name]
                egy_res = OOD.evaluate(r, egy_rows)
                abl.append({"model": key, "variant": vname, "method": method, "precision": used,
                            "dev_STS": round(rep["metrics"]["safe_task_success"], 3),
                            "dev_intent_F1": round(rep["metrics"]["intent_macro_f1"], 3),
                            "dev_unsafe": round(rep["metrics"]["unsafe_execution_rate"], 3),
                            "egy_unsupported_recall": round(egy_res["recall_unsupported"], 3),
                            "egy_alarm_recall": round(egy_res["recall_create_alarm"], 3),
                            "massive_F1": round(f1_score(m_gold, r.predict(m_texts), average="macro"), 3)})
                print(abl[-1])
        print(f"{key}: {(time.time() - t0) / 60:.1f} min")
        del model, r, caches; gc.collect(); torch.cuda.empty_cache()
    abl = pd.DataFrame(abl)
    abl.to_csv(f"{OUT}/scoring_ablation.csv", index=False)
    abl'''

MD19 = r"""## S4 diagnostics: few-shot sensitivity and calibration

Repeats the 6-shot prompt with **three exemplar sets** (shot seeds `seed`, `seed+1`, `seed+2`, all from train), each
scored `raw`, `calibrated` and `prior`, and saves every prediction on dev, on the native Egyptian items and on their
Saudi/MSA originals (`s4_diag_items.csv`), plus the label distribution on content-free inputs
(`s4_diag_content_free.csv`). Run as a commit, one model at a time (`DIAG_MODELS`); finished parts in an attached
earlier output are skipped. Analyse with `uv run python scripts/analyse_s4_diagnostics.py runs/s4/diagnostics`."""

C20 = r'''if RUN_S4_DIAGNOSTICS:
    import glob
    import shutil

    from athar.eval import ood as OOD

    # Crash-safe: every (model, exemplar set) is saved as its own file the moment it finishes.
    for f in glob.glob("/kaggle/input/**/s4_diag_part_*.csv", recursive=True):
        if not Path(f"{OUT}/{Path(f).name}").exists():
            shutil.copy(f, OUT)
            print("resumed:", Path(f).name)

    egy_rows = OOD.load_pairs(f"{OUT}/egy_test.jsonl", f"{OUT}/egy_test.csv")   # + the Saudi/MSA original
    egy_texts = [norm(r["text"]) for r in egy_rows]
    msa_texts = [norm(r["msa"]) for r in egy_rows]
    cfg_fast = {**CFG, "perturbations": [], "oracle": False}
    for key, model_id, precision in MODELS:
        if DIAG_MODELS and key not in DIAG_MODELS:
            continue
        tag = "-".join(DIAG_SETS)
        todo = [k for k in range(3) if not Path(f"{OUT}/s4_diag_part_{key}_s{k}_{tag}.csv").exists()]
        if not todo:
            print(f"{key}: all exemplar sets already done")
            continue
        t0 = time.time()
        model, tok, used = load_checked(model_id, precision)
        prof = llm.chat_profile(model_id).intent_kwargs()
        for k in todo:
            shots = llm.select_shots(train, per_label=SHOTS_PER_LABEL, seed=CFG["seed"] + k)
            cache, item_rows, cf_row = {}, [], {}   # one set of forward passes per exemplar set
            for method in ("raw", "calibrated", "prior"):
                name = f"S4-{key}-shotseed{k}-{method}"
                r = llm.LLMIntent(model, tok, name=name, shots=shots, method=method,
                                  prior_texts=PRIOR_TEXTS if method == "prior" else (), **prof)
                r._cache = cache
                if "dev" in DIAG_SETS:
                    rep = H.evaluate({name: r}, items, cfg_fast)["systems"][name]
                    for it_score in rep["items"]:
                        item_rows.append({"model": key, "shot_seed": k, "method": method, "set": "dev",
                                          "id": it_score["id"], "family_id": it_score["family_id"],
                                          "gold": it_score["gold_intent"], "pred": it_score["pred_intent"],
                                          "success": it_score["success"], "precision": used})
                for set_name, texts in (("egyptian", egy_texts), ("msa", msa_texts)):
                    if set_name not in DIAG_SETS:
                        continue
                    for row, pred in zip(egy_rows, r.predict(texts), strict=True):
                        item_rows.append({"model": key, "shot_seed": k, "method": method, "set": set_name,
                                          "id": row["id"], "family_id": row["id"], "gold": row["gold"], "pred": pred,
                                          "success": pred == row["gold"], "precision": used})
                if method == "calibrated":
                    cf_row = {"model": key, "shot_seed": k,
                              **dict(zip(llm.INTENTS, map(float, r.content_free()), strict=True))}
            pd.DataFrame(item_rows).to_csv(f"{OUT}/s4_diag_part_{key}_s{k}_{tag}.csv", index=False)
            pd.DataFrame([cf_row]).to_csv(f"{OUT}/s4_diag_cf_{key}_s{k}.csv", index=False)
            print(f"saved {key} exemplar set {k}: {(time.time() - t0) / 60:.1f} min")
        del model, r, cache; gc.collect(); torch.cuda.empty_cache()

    parts = sorted(glob.glob(f"{OUT}/s4_diag_part_*.csv"))
    if parts:
        pd.concat(map(pd.read_csv, parts)).to_csv(f"{OUT}/s4_diag_items.csv", index=False)
    cfs = sorted(glob.glob(f"{OUT}/s4_diag_cf_*.csv"))
    if cfs:
        cf = pd.concat(map(pd.read_csv, cfs))
        cf.to_csv(f"{OUT}/s4_diag_content_free.csv", index=False)
        print(cf.round(3).to_string(index=False))
    print("parts done:", [Path(p).stem.replace("s4_diag_part_", "") for p in parts])'''

MD21 = r"""## Bring the results back

Download `s4_results/` (*Output* tab on Kaggle) and put it in the repo under `runs/s4/<run name>/` (dev runs) or
`results/s4/` (the one Phase 4 test run). Every JSON report has every metric with family-bootstrap CIs, robustness
and calibration (S4/S7), the item-level scores, and the model revision, precision and chat profile it used;
`environment.json` has the GPUs, library versions, manifest hash and notebook version."""


def code(src):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": src.splitlines(keepends=True)}


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}


nb["cells"] = [md(MD0), code(C1), code(C2), code(C3), code(C4), md(MD5), code(C6), md(MD7), code(C8),
               md(MD9), code(C10), md(MD11), code(C12), md(MD13), code(C14), md(MD15), code(C16),
               md(MD17), code(C18), md(MD19), code(C20), md(MD21)]
NB.write_text(json.dumps(nb, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")
print("cells:", len(nb["cells"]))
