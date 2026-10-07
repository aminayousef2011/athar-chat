"""S4/S7 LLM recognizers, tested offline with a tiny randomly initialised GPT-2 and a word-level
tokenizer with a chat template, both built locally (nothing downloaded)."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")

from pathlib import Path  # noqa: E402

from athar.data.io import load_items  # noqa: E402
from athar.eval import harness as H  # noqa: E402
from athar.intent import llm  # noqa: E402
from athar.normalize import norm  # noqa: E402
from athar.schema import INTENTS  # noqa: E402

PILOT = Path(__file__).resolve().parents[1] / "data" / "pilot" / "pilot.jsonl"
FAST = {"seed": 1, "level": 0.95, "resamples": 100, "permutations": 100, "alpha": 0.05,
        "calibration_bins": 10, "oracle": False}

TEMPLATE = ("{% for m in messages %}<{{ m['role'] }}> {{ m['content'] }} </s> {% endfor %}"
            "{% if add_generation_prompt %}<assistant> {% endif %}")


def tiny_lm(texts):
    pre = tokenizers.pre_tokenizers.Whitespace()
    words = {w for t in [llm.SYSTEM_PROMPT, *llm.VERBALIZERS.values(), *llm.CONTENT_FREE, *texts]
             for w, _ in pre.pre_tokenize_str(t)}
    vocab = {w: i for i, w in enumerate(["[UNK]", "</s>", "<system>", "<user>", "<assistant>", "<", ">",
                                         *sorted(words)])}
    core = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="[UNK]"))
    core.pre_tokenizer = pre
    tok = transformers.PreTrainedTokenizerFast(tokenizer_object=core, unk_token="[UNK]", eos_token="</s>",
                                               pad_token="</s>")
    tok.chat_template = TEMPLATE
    torch.manual_seed(0)
    cfg = transformers.GPT2Config(vocab_size=len(vocab), n_positions=1024, n_embd=32, n_layer=2, n_head=2)
    return transformers.GPT2LMHeadModel(cfg).eval(), tok


@pytest.fixture(scope="module")
def pilot_and_lm():
    items = load_items(PILOT)
    model, tok = tiny_lm([norm(it.text) for it in items])
    return items, model, tok


@pytest.mark.parametrize("text, label", [
    ("تذكير", "create_reminder"), (" منبه.", "create_alarm"), ("غير مدعوم", "unsupported"),
    ("الطلب ده غير مدعوم مش تذكير", "unsupported"), ("أكيد! تذكير", "create_reminder"), ("ok", "invalid"),
])
def test_parse_label(text, label):
    assert llm.parse_label(text) == label


def test_shots_are_balanced_distinct_and_deterministic(pilot_and_lm):
    items, _, _ = pilot_and_lm
    shots = llm.select_shots(items, per_label=1, seed=3)
    assert [lab for _, lab in shots] == list(INTENTS)
    assert shots == llm.select_shots(items, per_label=1, seed=3)
    fams = {it.family_id for it in items if norm(it.text) in {t for t, _ in shots}}
    assert len(fams) == len(shots)


def test_prompt_uses_the_chat_template_and_shots(pilot_and_lm):
    items, model, tok = pilot_and_lm
    shots = llm.select_shots(items, per_label=1)
    r = llm.LLMIntent(model, tok, shots=shots)
    p = r.prompt("فكرني بكره")
    assert p.startswith("<system>") and p.endswith("<assistant> ")
    assert p.count("<user>") == len(shots) + 1


def test_scoring_and_calibration(pilot_and_lm):
    items, model, tok = pilot_and_lm
    texts = [norm(it.text) for it in items[:6]]
    raw = llm.LLMIntent(model, tok, method="raw")
    cal = llm.LLMIntent(model, tok, method="calibrated")
    for t in texts:
        lp = raw.label_logprobs(t)
        assert lp.shape == (3,) and np.all(lp < 0)
        assert np.isclose(cal.probabilities(t).sum(), 1.0)
    assert set(raw.predict(texts)) <= set(INTENTS)
    # contextual calibration maps the content-free inputs' average to the uniform distribution
    cf = np.mean([llm._softmax(cal.label_logprobs(t)) for t in llm.CONTENT_FREE], axis=0)
    adj = cf / cal.content_free()
    assert np.allclose(adj / adj.sum(), 1 / 3)


def test_generate_method_and_invalid_outputs_score_as_failures(pilot_and_lm):
    items, model, tok = pilot_and_lm
    gen = llm.LLMIntent(model, tok, name="S4-gen", method="generate", max_new_tokens=3)
    preds = gen.predict([norm(it.text) for it in items[:4]])
    assert all(p in (*INTENTS, "invalid") for p in preds)
    rep = H.evaluate({"S4-gen": gen}, items[:4], {**FAST, "perturbations": ["digits"], "comparisons": []})
    n_invalid = sum(p == "invalid" for p in preds)
    assert rep["systems"]["S4-gen"]["metrics"]["invalid_outputs"] == n_invalid


def test_harness_runs_an_llm_recognizer_with_calibration(pilot_and_lm):
    items, model, tok = pilot_and_lm
    s4 = llm.LLMIntent(model, tok, name="S4", shots=llm.select_shots(items, per_label=1))
    rep = H.evaluate({"S4": s4}, items[:6], {**FAST, "perturbations": ["politeness"], "comparisons": []})
    assert rep["systems"]["S4"]["calibration"] is not None
    assert 0.0 <= rep["systems"]["S4"]["metrics"]["safe_task_success"] <= 1.0


def test_lora_examples_mask_the_prompt(pilot_and_lm):
    items, _, tok = pilot_and_lm
    for (ids, labels), it in zip(llm.lora_examples(tok, items), items, strict=True):
        answer = [x for x in labels if x != -100]
        assert answer == tok(llm.VERBALIZERS[it.intent], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        assert labels[: len(ids) - len(answer)] == [-100] * (len(ids) - len(answer))


def test_lora_finetune_trains_only_adapters_and_lowers_the_loss(pilot_and_lm):
    pytest.importorskip("peft")
    items, _, _ = pilot_and_lm
    model, tok = tiny_lm([norm(it.text) for it in items])
    base_before = {n: p.detach().clone() for n, p in model.named_parameters()}
    losses = []
    tuned = llm.lora_finetune(model, tok, items, epochs=30, learning_rate=1e-3, r=8, alpha=16, dropout=0.0,
                              target_modules=["c_attn", "c_proj"], log=lambda s: losses.append(float(s.split()[-1])))
    assert losses[-1] < losses[0] - 0.2
    for n, p in tuned.named_parameters():
        if "lora_" not in n:
            key = n.replace("base_model.model.", "").replace(".base_layer", "")
            assert torch.equal(p.detach(), base_before[key]), n  # base weights stay frozen
    preds = llm.LLMIntent(tuned, tok, method="raw").predict([norm(it.text) for it in items])
    assert set(preds) <= set(INTENTS)
