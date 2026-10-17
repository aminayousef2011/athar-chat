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

``--split test`` rehearses the locked test run (SPLIT="test", UNLOCK_TEST, GATE_TAU, only the
test-run systems) **without reading the test split**: ``load_split(..., "test")`` returns the dev
items. It checks the test-only code paths: the guards, S7 scoring dev and test with the same
weights, and gated columns only for systems with a dev tau.
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
# Jais 2 writes roles literally but answers under "ai" (tests/test_llm.py JAIS2_STYLE): the stand-in
# does the same for Jais 2 ids, so its chat profile (assistant_role "ai") is consistent, as on Kaggle
JAIS2_TEMPLATE = TEMPLATE.replace("<assistant> {% endif %}", "<ai> {% endif %}")


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
    huggingface_hub.hf_hub_download = lambda repo_id, filename, token=None, revision=None: f"/fake/{repo_id}/{filename}"

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
    import copy

    tok_jais2 = copy.deepcopy(tok)
    tok_jais2.chat_template = JAIS2_TEMPLATE

    def tokenizer_for(model_id: str):  # type: ignore[no-untyped-def]
        return tok_jais2 if model_id.startswith("inception42/Jais-2-") else tok

    def load_llm(model_id: str, precision: str = "fp16", token: str | None = None, revision: str | None = None,  # type: ignore[no-untyped-def]
                 trust_remote_code: bool = False):
        calls["load_llm"].append((model_id, precision, revision, trust_remote_code))
        return model, tokenizer_for(model_id)

    def load_tokenizer(model_id: str, token: str | None = None, revision: str | None = None,  # type: ignore[no-untyped-def]
                       trust_remote_code: bool = False):
        calls["load_tokenizer"].append((model_id, revision, trust_remote_code))
        return tokenizer_for(model_id)

    llm.load_llm = load_llm
    llm.load_tokenizer = load_tokenizer


def stand_in_for_test() -> None:
    """``load_split(..., "test")`` returns the dev items: the rehearsal never opens the test split."""
    from athar.data import split

    real = split.load_split

    def load_split(manifest, name, unlock_test=False):  # type: ignore[no-untyped-def]
        if name == "test":
            assert unlock_test, "the notebook must pass unlock_test=True for the test split"
            return real(manifest, "dev")
        return real(manifest, name)

    split.load_split = load_split  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--flags", choices=["all", "as-is"], default="all")
    p.add_argument("--keep", action="store_true", help="keep the temporary folder")
    p.add_argument("--split", choices=["dev", "test"], default="dev",
                   help="test: rehearse the locked test run on dev items")
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
    calls: dict[str, list] = {"load_llm": [], "load_tokenizer": []}  # type: ignore[type-arg]
    install_fakes(vocab_texts, calls)
    if a.split == "test":
        stand_in_for_test()

    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    ns: dict[str, object] = {"__name__": "__notebook__"}
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        src = "\n".join(line for line in src.splitlines() if not line.lstrip().startswith("!"))
        # forward slashes: a Windows path inside a string literal would read as escapes ("\U...")
        src = src.replace("/kaggle/input", kaggle_in.as_posix()).replace("/kaggle/working", kaggle_out.as_posix())
        print(f"--- cell {i}", flush=True)
        exec(compile(src, f"cell {i}", "exec"), ns)  # noqa: S102 (our own notebook)
        if "RUN_S4" in src and "PRIMARY" in src and i == 1:
            ns.update(S8_MAX_NEW_TOKENS=4, S8_SEEDS=[0, 1])   # the tiny model has 2048 positions
            if a.flags == "all":
                ns.update(RUN_PROMPT_CHECK=True, RUN_S4=True, RUN_S7_QLORA=True, S7_SEEDS=[0], RUN_MASSIVE_OOD=True,
                          RUN_SCORING_ABLATION=True, RUN_S4_DIAGNOSTICS=True, DIAG_MODELS=[],
                          DIAG_SETS=("dev", "egyptian", "msa"), RUN_EGYPTIAN_OOD=True, RUN_S8=True,
                          RUN_S8_EGYPTIAN=True, RUN_SHOT_STYLE=True, RUN_MODELS=[])
            if a.split == "test":   # the test-run commits: only the pre-registered systems
                ns.update(SPLIT="test", UNLOCK_TEST=True, RUN_PROMPT_CHECK=False, RUN_SHOT_STYLE=False,
                          RUN_SCORING_ABLATION=False, RUN_S4_DIAGNOSTICS=False, RUN_MASSIVE_OOD=False,
                          RUN_EGYPTIAN_OOD=False, RUN_S8_EGYPTIAN=False, RUN_S4=True, RUN_S7_QLORA=True,
                          S7_SEEDS=[0], RUN_S8=True,
                          GATE_TAU={f"S4-{k}-6shot-{ns['PRIMARY']}": 0.5 for k, *_ in ns["MODELS"]})  # type: ignore[attr-defined]
        if "CFG = yaml.safe_load" in src:
            ns["CFG"].update(resamples=50, permutations=50)  # type: ignore[union-attr]

    files = sorted(f.name for f in out.iterdir())
    print("\nwrote:", files)
    problems = check(out, ns, calls, a.flags)
    if a.split == "test":
        problems += check_test_run(out, ns)
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
    from athar.intent import llm

    for model_id, _precision, revision, remote in calls["load_llm"]:
        if revision != revs.get(model_id):  # type: ignore[attr-defined]
            problems.append(f"{model_id} loaded at revision {revision!r}, not the pinned one")
        if remote != llm.chat_profile(model_id).trust_remote_code:
            problems.append(f"{model_id} loaded with trust_remote_code={remote}")
    for model_id, revision, _remote in calls["load_tokenizer"]:
        if revision != revs.get(model_id):  # type: ignore[attr-defined]
            problems.append(f"{model_id} tokenizer at revision {revision!r}, not the pinned one")
    if ns.get("RUN_PROMPT_CHECK"):
        for key, *_ in ns["MODELS"]:  # type: ignore[attr-defined]
            if not (out / f"prompt_{key}.txt").exists():
                problems.append(f"prompt_{key}.txt not written")
        if not (out / "verbalizer_tokens.csv").exists():
            problems.append("verbalizer_tokens.csv not written")
    if ns.get("RUN_S8"):
        s8 = pd.read_csv(out / "s8_summary.csv")
        diag = pd.read_csv(out / "s8_diagnostics.csv")
        n_runs = len(ns["S8_MODELS"]) * len(ns["S8_SEEDS"])  # type: ignore[arg-type]
        if len(s8) != n_runs or len(diag) != n_runs:
            problems.append(f"S8: {len(s8)} summary rows and {len(diag)} diagnostics rows, expected {n_runs}")
        for key, *_ in ns["S8_MODELS"]:  # type: ignore[attr-defined]
            rep = json.loads((out / f"s8_{key}.json").read_text(encoding="utf-8"))
            if f"S8-{key}" not in rep["seed_summaries"]:
                problems.append(f"S8 {key}: no seed summary")
            if ns.get("RUN_S8_EGYPTIAN") and not (out / f"egyptian_ood_s8_{key}.json").exists():
                problems.append(f"S8 {key}: native Egyptian check not written")
        if len(list(out.glob("s8_trace_*.jsonl"))) != n_runs:
            problems.append("S8: one trace per seed expected")
    if ns.get("RUN_SHOT_STYLE"):
        ss = pd.read_csv(out / "shot_style.csv")
        want = {m[0]: {"turns", "inline"} | ({"turns-old-role"} if llm.chat_profile(m[1]).assistant_role != "assistant" else set())
                for m in ns["MODELS"] if m[0] in ns["SHOT_STYLE_MODELS"]}  # type: ignore[attr-defined,operator]
        got = {k: set(g.shot_style) for k, g in ss.groupby("model")}
        if got != want:
            problems.append(f"shot_style.csv formats {got}, expected {want}")
    if ns.get("RUN_PROMPT_CHECK"):
        vt = pd.read_csv(out / "verbalizer_tokens.csv")
        if "answer_consistent" not in vt.columns or set(vt.model) != {m[0] for m in ns["MODELS"]}:  # type: ignore[attr-defined]
            problems.append("verbalizer_tokens.csv lacks the answer-consistency check")
    if ns.get("RUN_EGYPTIAN_OOD") and ns.get("PAIRED"):
        paired = pd.read_csv(out / "egyptian_paired.csv")
        if len(paired) != len(ns["PAIRED"]):  # type: ignore[arg-type]
            problems.append("egyptian_paired.csv: a pair is missing")
    if ns.get("RUN_S4"):
        for key, *_ in ns["MODELS"]:  # type: ignore[attr-defined]
            rep = json.loads((out / f"{key}.json").read_text(encoding="utf-8"))
            names = set(rep["systems"])
            for method in ("raw", "calibrated", "prior", "generate"):
                if not any(n.endswith(f"-6shot-{method}") for n in names):
                    problems.append(f"{key}: no 6-shot {method} system")
            if not all("gate" in s for n, s in rep["systems"].items() if not n.endswith("generate")):
                problems.append(f"{key}: gate analysis missing")
            if ns.get("S4_INLINE") and f"S4-{key}-6shot-inline-prior" not in names:
                problems.append(f"{key}: no 6-shot inline secondary system")
        summary = pd.read_csv(out / "summary.csv")
        if summary.STS.isna().any():
            problems.append("summary.csv has empty STS")
        if ns.get("S4_INLINE"):
            inline = pd.read_csv(out / "s4_inline_vs_turns.csv")
            n_models, n_confirm = len(ns["MODELS"]), len(ns["INLINE_CONFIRM"])  # type: ignore[arg-type]
            if len(inline) != n_models or inline.p_holm_this_run.notna().sum() != n_confirm:
                problems.append("s4_inline_vs_turns.csv: a model or a Holm value is missing")
    if ns.get("RUN_S7_QLORA"):
        problems += check_s7(out, ns)
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


def check_s7(out: Path, ns: dict[str, object]) -> list[str]:
    import pandas as pd

    problems = []
    seeds, splits = list(ns["S7_SEEDS"]), ["dev"] + (["test"] if ns["SPLIT"] == "test" else [])  # type: ignore[call-overload]
    rows = pd.read_csv(out / "s7_seeds.csv")
    got = {(s, sp, n.rsplit("-", 1)[-1]) for s, sp, n in zip(rows.seed, rows.split, rows.system, strict=True)}
    want = {(s, sp, m) for s in seeds for sp in splits for m in ns["S7_METHODS"]}  # type: ignore[attr-defined]
    if got != want:
        problems.append(f"s7_seeds.csv rows {sorted(got)}, expected {sorted(want)}")
    training = pd.read_csv(out / "s7_training.csv")
    if len(training) != len(seeds) * ns["S7_RECIPE"]["epochs"] or not training.loss.notna().all():  # type: ignore[index]
        problems.append("s7_training.csv: an epoch record is missing")
    for s in seeds:
        for sp in splits:
            if not (out / f"s7_jais2_qlora_s{s}_{sp}.json").exists():
                problems.append(f"S7 seed {s}: no {sp} report")
        if ns.get("S7_SAVE_ADAPTERS") and not (Path(str(ns["ADAPTERS"])) / f"s{s}" / "adapter_config.json").exists():
            problems.append(f"S7 seed {s}: adapter not saved")
    if ns.get("S7_EGYPTIAN") and len(pd.read_csv(out / "egyptian_ood_s7.csv")) != len(seeds):
        problems.append("egyptian_ood_s7.csv: a seed is missing")
    return problems


def check_test_run(out: Path, ns: dict[str, object]) -> list[str]:
    """Test rehearsal: tau never picked on test; S7 dev and test describe the same weights."""
    import pandas as pd

    problems = []
    env = json.loads((out / "environment.json").read_text(encoding="utf-8"))
    if env["split"] != "test":
        problems.append("environment.json does not record the test split")
    summary = pd.read_csv(out / "summary.csv")
    tau = ns["GATE_TAU"]
    gated = summary[summary.gate_tau.notna()] if "gate_tau" in summary else summary.iloc[:0]
    if set(gated.system) - set(tau):  # type: ignore[arg-type]
        problems.append(f"gated columns on test for systems without a dev tau: {sorted(set(gated.system) - set(tau))}")  # type: ignore[arg-type]
    if not (gated.gate_tau == gated.system.map(tau)).all():  # type: ignore[arg-type]
        problems.append("a test row is gated at a tau other than its dev tau")
    s7 = pd.read_csv(out / "s7_seeds.csv")
    test_rows = s7[s7.split == "test"]
    for _, row in test_rows.iterrows():
        dev = json.loads((out / f"s7_jais2_qlora_s{row.seed}_dev.json").read_text(encoding="utf-8"))
        rule = dev["systems"][row.system]["gate"]["rule_threshold_on_this_split"]
        if row.gate_tau != rule:
            problems.append(f"{row.system}: test tau {row.gate_tau} is not its dev tau {rule}")
    for s in ns["S7_SEEDS"]:  # type: ignore[attr-defined]
        dev = json.loads((out / f"s7_jais2_qlora_s{s}_dev.json").read_text(encoding="utf-8"))
        test = json.loads((out / f"s7_jais2_qlora_s{s}_test.json").read_text(encoding="utf-8"))
        if dev["model"]["training"] != test["model"]["training"]:
            problems.append(f"S7 seed {s}: dev and test reports come from different training runs")
    return problems


if __name__ == "__main__":
    raise SystemExit(main())
