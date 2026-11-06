"""Does S4 score the label tokens the model would actually see? (plan step 1.4, 2026-11-06)

``athar.intent.llm.LLMIntent.token_logprobs`` tokenizes the prompt and each label verbalizer
separately and concatenates the ids (``p_ids + c_ids``). A tokenizer may split the joined text
differently at the boundary (a merge across the answer prefix and the label, or a different
first piece of the label). This script compares, for every model at its pinned revision and for
the prompts S4 really used (0-shot, 6-shot as chat turns, 6-shot inline), the separate ids with
``tok(prompt + label)``. It reports, per model and setting, how many (request, label) pairs
differ and shows the first difference. Tokenizers only: no model weights, CPU.

    uv run python scripts/check_label_tokenization.py [--n 40] [--token HF_TOKEN]
    -> results/posthoc/label_tokenization.json

Gated models (Jais 2) are skipped without a token and listed as such.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from athar.data.split import load_split  # noqa: E402
from athar.intent import llm  # noqa: E402
from athar.normalize import norm  # noqa: E402

# the five S4 models and the revisions pinned in scripts/make_notebook.py
MODELS = {
    "inception42/Jais-3-7B": "94d69f49b4af702d42fcee122cc06e493a6fe7e1",
    "inception42/Jais-3-3.7B": "3d5754d5db87cd3a59f0f5b09c7ac39b351ff5ed",
    "inception42/Jais-2-8B-Chat": "da0e1639cd92b508b24120f7f77f5270a8465dc4",
    "MBZUAI-Paris/Nile-Chat-4B": "2a365667659bbd6786c2760d868e37c506d76ce3",
    "Qwen/Qwen3-8B": "b968826d9c46dd6066d109eabc6255188de91218",
}
SHOTS_PER_LABEL, SEED = 2, 0   # as in the notebook (SHOTS_PER_LABEL, CFG["seed"])


def check_model(model_id: str, revision: str, texts: list[str], shots: list[tuple[str, str]],
                token: str | None) -> dict[str, Any]:
    prof = llm.chat_profile(model_id)
    try:
        tok = llm.load_tokenizer(model_id, token=token, revision=revision, trust_remote_code=prof.trust_remote_code)
    except Exception as e:  # gated or unreachable
        return {"status": "skipped", "reason": f"{type(e).__name__}: {str(e).splitlines()[0][:200]}"}
    labels = llm.VERBALIZERS
    sep_label = {lab: tok(w, add_special_tokens=False)["input_ids"] for lab, w in labels.items()}
    out: dict[str, Any] = {"status": "checked", "revision": revision,
                           "label_ids": {lab: ids for lab, ids in sep_label.items()},
                           "answer_consistent": llm.answer_consistent(tok, prof), "settings": {}}
    settings = {"0shot": ([], "turns"), "6shot-turns": (shots, "turns"), "6shot-inline": (shots, "inline")}
    for name, (sh, style) in settings.items():
        probe = llm.LLMIntent(llm._NoModel(), tok, shots=sh, shot_style=style, **prof.intent_kwargs())
        pairs = differ = 0
        first: dict[str, Any] | None = None
        for text in texts:
            prompt = probe.prompt(text)
            p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
            for lab, w in labels.items():
                pairs += 1
                joint = tok(prompt + w, add_special_tokens=False)["input_ids"]
                scored = p_ids + sep_label[lab]
                if joint != scored:
                    differ += 1
                    if first is None:
                        k = next((i for i, (a, b) in enumerate(zip(joint, scored, strict=False)) if a != b),
                                 min(len(joint), len(scored)))
                        first = {"text": text, "label": lab, "position_from_end": len(scored) - k,
                                 "scored_tail": tok.convert_ids_to_tokens(scored[max(0, k - 2):]),
                                 "joint_tail": tok.convert_ids_to_tokens(joint[max(0, k - 2):])}
        out["settings"][name] = {"pairs": pairs, "differ": differ, "first_difference": first}
    return out


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    p.add_argument("--n", type=int, default=40, help="requests per setting (from dev, then natural)")
    p.add_argument("--token", default=None)
    a = p.parse_args(argv)
    from athar.intent.neural import use_system_certificates
    use_system_certificates()   # behind a TLS-intercepting proxy, as the other download scripts do
    llm_mod_shots = list(load_split(REPO / "data/splits/manifest.json", "train"))
    shots = llm.select_shots(llm_mod_shots, per_label=SHOTS_PER_LABEL, seed=SEED)
    pool = [norm(it.text) for it in load_split(REPO / "data/splits/manifest.json", "dev")]
    pool += [norm(it.text) for it in load_split(REPO / "data/natural/manifest.json", "test", unlock_test=True)]
    texts = random.Random(SEED).sample(pool, min(a.n, len(pool)))
    result: dict[str, Any] = {"note": "plan step 1.4; tokenizers only; prompts as S4 built them (normalized text, "
                                      "train exemplars, notebook seed)", "n_requests": len(texts), "models": {}}
    for model_id, rev in MODELS.items():
        r = check_model(model_id, rev, texts, shots, a.token)
        result["models"][model_id] = r
        if r["status"] != "checked":
            print(f"{model_id}: SKIPPED ({r['reason']})")
            continue
        cells = ", ".join(f"{k} {v['differ']}/{v['pairs']}" for k, v in r["settings"].items())
        print(f"{model_id}: answer_consistent={r['answer_consistent']}; differing pairs: {cells}")
        for k, v in r["settings"].items():
            if v["first_difference"]:
                f = v["first_difference"]
                print(f"   {k}: label {f['label']}: scored {f['scored_tail']} vs joint {f['joint_tail']}")
    out = REPO / "results" / "posthoc" / "label_tokenization.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("wrote", out.relative_to(REPO))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
