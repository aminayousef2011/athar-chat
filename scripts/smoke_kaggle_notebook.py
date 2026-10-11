"""Run notebooks/kaggle_s4_llm.ipynb end to end on CPU before spending GPU hours on Kaggle.

    uv run python scripts/build_kaggle_kit.py
    uv run --extra neural --with pandas python scripts/smoke_kaggle_notebook.py [--flags all|as-is]

Every code cell is executed in order, in one namespace, with Kaggle replaced by stand-ins:
``/kaggle/input`` holds the unzipped kit, ``/kaggle/working`` is a temporary folder, the
secrets client and the Hugging Face Hub are fakes, MASSIVE is a 20-row fake, and every model is
a tiny randomly initialised GPT-2 with a word-level tokenizer built locally (nothing is
downloaded; the Egyptian NLU files come from ``data/external/egyptian-nlu``). Bootstrap and
permutation counts are cut to 50. The numbers are meaningless; what is checked is that every
cell runs, every flag combination writes its files, and the files have the expected shape.

``--flags all`` (default) switches every RUN_* flag on with all diagnostics sets; ``as-is``
keeps the flags as they are in the notebook, i.e. exactly the next Kaggle run.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import types
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "notebooks" / "kaggle_s4_llm.ipynb"
KIT_ZIP = ROOT / "dist" / "athar-chat-kit.zip"
EGY = ROOT / "data" / "external" / "egyptian-nlu"
TEMPLATE = ("{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} </s> {% endfor %}"
            "{% if add_generation_prompt %}<assistant> {% endif %}")


def tiny_lm(texts: list[str]):  # type: ignore[no-untyped-def]
    import tokenizers
    import torch
    import transformers

    from athar.intent import llm

    pre = tokenizers.pre_tokenizers.Whitespace()
    words = {w for t in [llm.SYSTEM_PROMPT, llm.SYSTEM_PROMPT_NUMBER, *llm.VERBALIZERS.values(),
                         *llm.NUMBERS.values(), *llm.CONTENT_FREE, *texts] for w, _ in pre.pre_tokenize_str(t)}
    vocab = {w: i for i, w in enumerate(["[UNK]", "</s>", "<system>", "<user>", "<assistant>", "<", ">", *sorted(words)])}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", eos_token="</s>",
                                               pad_token="</s>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=len(vocab), n_positions=2048, n_embd=32, n_layer=2, n_head=2)
    return transformers.GPT2LMHeadModel(cfg).eval(), tok


def install_fakes(vocab_texts: list[str], calls: dict[str, list]) -> None:  # type: ignore[type-arg]
    """kaggle_secrets, the Hub functions the notebook calls, datasets.load_dataset, llm.load_llm."""
    from athar.intent import llm

    secrets = types.ModuleType("kaggle_secrets")

    class UserSecretsClient:
        def get_secret(self, name: str) -> str:
            return "fake-token"

    secrets.UserSecretsClient = UserSecretsClient  # type: ignore[attr-defined]
    sys.modules["kaggle_secrets"] = secrets

    import huggingface_hub

    huggingface_hub.login = lambda token=None: None
    huggingface_hub.hf_hub_download = lambda repo_id, filename, token=None: f"/fake/{repo_id}/{filename}"

    class HfApi:
        def model_info(self, model_id: str, token: str | None = None):  # type: ignore[no-untyped-def]
            return types.SimpleNamespace(sha="fake-sha")

    huggingface_hub.HfApi = HfApi

    datasets = types.ModuleType("datasets")

    class FakeDS(dict):  # type: ignore[type-arg]
        column_names = ["text", "label_text"]

    def load_dataset(name: str, config: str, split: str) -> FakeDS:
        intents = ["alarm_set", "weather_query", "play_music", "calendar_set", "alarm_remove"] * 4
        return FakeDS(text=[f"نص تجريبي رقم {i}" for i in range(len(intents))], label_text=intents)

    datasets.load_dataset = load_dataset  # type: ignore[attr-defined]
    sys.modules["datasets"] = datasets

    model, tok = tiny_lm(vocab_texts)

    def load_llm(model_id: str, precision: str = "fp16", token: str | None = None, revision: str | None = None):  # type: ignore[no-untyped-def]
        calls["load_llm"].append((model_id, precision, revision))
        return model, tok

    llm.load_llm = load_llm


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--flags", choices=["all", "as-is"], default="all")
    p.add_argument("--keep", action="store_true", help="keep the temporary folder")
    a = p.parse_args(argv)
    if not KIT_ZIP.exists():
        sys.exit("dist/athar-chat-kit.zip is missing: run `uv run python scripts/build_kaggle_kit.py` first")
    if not (EGY / "egy_test.jsonl").exists() or not (EGY / "egy_test.csv").exists():
        sys.exit("data/external/egyptian-nlu is missing: run `uv run athar-ood ...` once to download it")

    tmp = Path(tempfile.mkdtemp(prefix="athar-smoke-"))
    kaggle_in, kaggle_out = tmp / "input", tmp / "working"
    with zipfile.ZipFile(KIT_ZIP) as z:
        z.extractall(kaggle_in / "datasets" / "someone" / "athar-chat-kit")
    out = kaggle_out / "s4_results"
    out.mkdir(parents=True)
    for f in ("egy_test.jsonl", "egy_test.csv"):
        shutil.copy(EGY / f, out / f)

    import json as _json

    from athar.normalize import norm

    vocab_texts = [norm(_json.loads(line)["text"]) for line in
                   (ROOT / "data" / "draft" / "athar_draft.jsonl").read_text(encoding="utf-8").splitlines()]
    calls: dict[str, list] = {"load_llm": []}  # type: ignore[type-arg]
    install_fakes(vocab_texts, calls)

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict[str, object] = {"__name__": "__notebook__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!"))
        src = src.replace("/kaggle/input", str(kaggle_in)).replace("/kaggle/working", str(kaggle_out))
        print(f"--- cell {i}", flush=True)
        exec(compile(src, f"cell {i}", "exec"), ns)  # noqa: S102 (our own notebook)
        if "RUN_S4" in src and "PRIMARY" in src and i == 1 and a.flags == "all":
            ns.update(RUN_S4=True, RUN_S7_QLORA=True, S7_SEEDS=[0], RUN_MASSIVE_OOD=True, RUN_SCORING_ABLATION=True,
                      RUN_S4_DIAGNOSTICS=True, DIAG_MODELS=[], DIAG_SETS=("dev", "egyptian", "msa"),
                      RUN_EGYPTIAN_OOD=True)
        if "CFG = yaml.safe_load" in src:
            ns["CFG"].update(resamples=50, permutations=50)  # type: ignore[union-attr]

    files = sorted(f.name for f in out.iterdir())
    print("\nwrote:", files)
    problems = check(out, ns, calls, a.flags)
    for msg in problems:
        print("PROBLEM:", msg)
    if a.keep:
        print("kept", tmp)
    else:
        shutil.rmtree(tmp)
    print("SMOKE TEST", "FAILED" if problems else "PASSED")
    return 1 if problems else 0


def check(out: Path, ns: dict[str, object], calls: dict[str, list], flags: str) -> list[str]:  # type: ignore[type-arg]
    import pandas as pd

    problems = []
    if not (out / "environment.json").exists():
        problems.append("environment.json not written")
    revs = ns["REVISIONS"]
    for model_id, _precision, revision in calls["load_llm"]:
        if revision != revs.get(model_id):  # type: ignore[attr-defined]
            problems.append(f"{model_id} loaded at revision {revision!r}, not the pinned one")
    if ns.get("RUN_S4"):
        for key, *_ in ns["MODELS"]:  # type: ignore[attr-defined]
            rep = json.loads((out / f"{key}.json").read_text(encoding="utf-8"))
            names = set(rep["systems"])
            for method in ("raw", "calibrated", "prior", "generate"):
                if not any(n.endswith(f"-6shot-{method}") for n in names):
                    problems.append(f"{key}: no 6-shot {method} system")
            if not all("gate" in s for n, s in rep["systems"].items() if not n.endswith("generate")):
                problems.append(f"{key}: gate analysis missing")
        summary = pd.read_csv(out / "summary.csv")
        if summary.STS.isna().any():
            problems.append("summary.csv has empty STS")
    if ns.get("RUN_S7_QLORA") and not (out / "s7_seeds.csv").exists():
        problems.append("s7_seeds.csv not written")
    if ns.get("RUN_MASSIVE_OOD") and not (out / "massive_ood.csv").exists():
        problems.append("massive_ood.csv not written")
    if ns.get("RUN_EGYPTIAN_OOD"):
        egy = pd.read_csv(out / "egyptian_ood.csv")
        if not egy.system.str.endswith(f"-{ns['PRIMARY']}").iloc[1:].all():
            problems.append("Egyptian check did not use the primary method")
    if ns.get("RUN_S4_DIAGNOSTICS"):
        items = pd.read_csv(out / "s4_diag_items.csv")
        want = set(ns["DIAG_SETS"])  # type: ignore[arg-type]
        if set(items.set) != want:
            problems.append(f"diagnostics sets {sorted(set(items.set))}, expected {sorted(want)}")
        n_models = len(ns["DIAG_MODELS"]) or len(ns["MODELS"])  # type: ignore[arg-type]
        if items.groupby(["model", "shot_seed", "method"]).ngroups != n_models * 9:
            problems.append("diagnostics: not every model x exemplar set x method was scored")
        if "msa" in want:
            m = items[items.set == "msa"]
            if len(m) != n_models * 9 * 196:
                problems.append(f"diagnostics: {len(m)} MSA rows, expected {n_models * 9 * 196}")
        if not (out / "s4_diag_content_free.csv").exists():
            problems.append("content-free probabilities not written")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
