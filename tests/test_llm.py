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
    words = {w for t in [llm.SYSTEM_PROMPT, llm.SYSTEM_PROMPT_NUMBER, *llm.VERBALIZERS.values(),
                                 *llm.NUMBERS.values(), *llm.CONTENT_FREE, *texts]
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


def test_length_norm_is_the_mean_of_the_same_token_scores(pilot_and_lm):
    items, model, tok = pilot_and_lm
    text = norm(items[0].text)
    summed = llm.LLMIntent(model, tok, method="raw")
    mean = llm.LLMIntent(model, tok, method="raw", length_norm=True)
    mean._cache = summed._cache                      # same prompt and answers: one forward pass
    per_token = summed.token_logprobs(text)
    assert [len(t) for t in per_token] == [len(summed.label_ids[lab]) for lab in INTENTS]
    np.testing.assert_allclose(summed.label_logprobs(text), [t.sum() for t in per_token])
    np.testing.assert_allclose(mean.label_logprobs(text), [t.mean() for t in per_token])
    assert len(mean.predict([text])) == 1


def test_number_answers_are_one_token_and_use_their_own_prompt(pilot_and_lm):
    items, model, tok = pilot_and_lm
    num = llm.LLMIntent(model, tok, method="calibrated", answer_format="number",
                        shots=llm.select_shots(items, per_label=1))
    assert all(len(ids) == 1 for ids in num.label_ids.values())
    prompt = num.prompt(norm(items[0].text))
    assert "رد برقم واحد بس" in prompt and "<assistant> 3 </s>" in prompt   # shots answer with digits
    assert set(num.predict([norm(it.text) for it in items[:5]])) <= set(INTENTS)
    with pytest.raises(ValueError):
        llm.LLMIntent(model, tok, method="generate", answer_format="number")
    with pytest.raises(ValueError):
        llm.LLMIntent(model, tok, answer_format="letters")


def test_prior_calibration_uses_unlabeled_train_texts(pilot_and_lm):
    items, model, tok = pilot_and_lm
    texts = [norm(it.text) for it in items]
    prior = llm.LLMIntent(model, tok, method="prior", prior_texts=texts[:8])
    expected = np.mean([llm._softmax(prior.label_logprobs(t)) for t in texts[:8]], axis=0)
    np.testing.assert_allclose(prior.label_prior(), expected)
    # dividing by the prior maps the prior texts' average to the uniform distribution
    adj = expected / prior.label_prior()
    assert np.allclose(adj / adj.sum(), 1 / 3)
    assert np.isclose(prior.probabilities(texts[9]).sum(), 1.0)
    with pytest.raises(ValueError, match="prior_texts"):
        llm.LLMIntent(model, tok, method="prior")


# --------------------------------------------------------------------------- Jais 3 / profiles
JAIS3_TEMPLATE = Path(__file__).resolve().parent / "fixtures" / "jais3_chat_template.jinja"
JAIS3_SHA256 = "38964d6fc5e411718d88f7a5c1b3664b3d97208c6ac93160b6c9eadc63c14057"
JAIS3_TAGS = ["<|ifm|endoftext|>", "<|ifm|im_start|>", "<|ifm|im_end|>", "<ifm|think>", "</ifm|think>"]


def jais3_tokenizer(texts):
    """A word-level tokenizer that renders prompts with the real (pinned) Jais 3 chat template."""
    import hashlib

    assert hashlib.sha256(JAIS3_TEMPLATE.read_bytes()).hexdigest() == JAIS3_SHA256
    _, tok = tiny_lm(texts)
    tok.add_special_tokens({"bos_token": "<|ifm|begin_of_text|>", "additional_special_tokens": JAIS3_TAGS})
    tok.chat_template = JAIS3_TEMPLATE.read_text(encoding="utf-8")
    return tok


def test_default_profile_keeps_the_earlier_prompts_byte_identical(pilot_and_lm):
    items, model, tok = pilot_and_lm
    shots = llm.select_shots(items, per_label=1)
    text = norm(items[0].text)
    # the message list the runs before 2026-10-15 used
    old = [{"role": "system", "content": llm.SYSTEM_PROMPT}]
    for s, lab in shots:
        old += [{"role": "user", "content": s}, {"role": "assistant", "content": llm.VERBALIZERS[lab]}]
    old.append({"role": "user", "content": text})
    assert llm.build_messages(text, shots) == old
    expected = tok.apply_chat_template(old, tokenize=False, add_generation_prompt=True)
    for model_id in ("inception42/Jais-2-8B-Chat", "MBZUAI-Paris/Nile-Chat-4B"):
        r = llm.LLMIntent(model, tok, shots=shots, **llm.chat_profile(model_id).intent_kwargs())
        assert r.prompt(text) == expected


def test_chat_profiles():
    jais3 = llm.chat_profile("inception42/Jais-3-7B")
    assert jais3 is llm.chat_profile("inception42/Jais-3-3.7B") and jais3.trust_remote_code
    assert jais3.answer_prefix == "</ifm|think>" and jais3.assistant_fields == {"reasoning_content": ""}
    assert llm.chat_profile("Qwen/Qwen3-8B").intent_kwargs()["template_kwargs"] == {"enable_thinking": False}
    assert llm.chat_profile("inception42/Jais-2-8B-Chat") == llm.ChatProfile()


def test_jais3_scores_the_label_exactly_where_its_template_writes_an_answer(pilot_and_lm):
    items, model, _ = pilot_and_lm
    tok = jais3_tokenizer([norm(it.text) for it in items])
    shots = llm.select_shots(items, per_label=1)
    text = norm(items[0].text)
    prof = llm.chat_profile("inception42/Jais-3-7B")
    r = llm.LLMIntent(model, tok, shots=shots, **prof.intent_kwargs())
    prompt = r.prompt(text)
    assert prompt.startswith("<|ifm|begin_of_text|><|ifm|im_start|>system\n" + llm.SYSTEM_PROMPT)
    assert prompt.endswith("<|ifm|im_start|>assistant\n<ifm|think>\n</ifm|think>")
    # every exemplar answer is rendered as "empty thinking, then the label"
    assert prompt.count("<ifm|think>\n</ifm|think>") == len(shots) + 1
    # scoring "prompt + label" == the template rendering a finished answer with no reasoning
    msgs = llm.build_messages(text, shots, assistant_fields=prof.assistant_fields)
    done = msgs + [{"role": "assistant", "content": llm.VERBALIZERS["create_alarm"], "reasoning_content": ""}]
    rendered = tok.apply_chat_template(done, tokenize=False, **prof.template_kwargs)
    assert rendered == prompt + llm.VERBALIZERS["create_alarm"] + "<|ifm|im_end|>"


def test_jais3_template_rejects_shots_without_a_thinking_field(pilot_and_lm):
    items, model, _ = pilot_and_lm
    tok = jais3_tokenizer([norm(it.text) for it in items])
    r = llm.LLMIntent(model, tok, shots=llm.select_shots(items, per_label=1))   # default profile
    with pytest.raises(Exception, match="thinking field"):
        r.prompt("فكرني")
    zero = llm.LLMIntent(model, tok, **llm.chat_profile("inception42/Jais-3-7B").intent_kwargs())
    assert zero.prompt("فكرني").count("<|ifm|im_start|>user\n") == 1


def test_inline_shots_put_the_exemplars_in_the_user_message(pilot_and_lm):
    items, model, tok = pilot_and_lm
    shots = llm.select_shots(items, per_label=1)
    msgs = llm.build_messages("فكرني", shots, shot_style="inline")
    assert [m["role"] for m in msgs] == ["system", "user"]
    content = msgs[1]["content"]
    assert content.startswith(llm.INLINE_HEADER) and content.endswith(f"{llm.INLINE_REQUEST} فكرني")
    assert content.count(llm.INLINE_ANSWER) == len(shots)
    assert llm.build_messages("فكرني", [], shot_style="inline") == llm.build_messages("فكرني", [])
    r = llm.LLMIntent(model, tok, shots=shots, shot_style="inline", method="raw")
    assert r.prompt("فكرني").count("<user>") == 1 and len(r.predict(["فكرني"])) == 1
    with pytest.raises(ValueError):
        llm.build_messages("x", shots, shot_style="table")
    with pytest.raises(ValueError):
        llm.LLMIntent(model, tok, shot_style="table")


def test_answer_prefix_is_used_by_scoring_and_generation(pilot_and_lm):
    items, model, tok = pilot_and_lm
    r = llm.LLMIntent(model, tok, method="generate", answer_prefix="تذكير", max_new_tokens=2)
    assert r.prompt("فكرني").endswith("<assistant> تذكير")
    assert r.predict(["فكرني"])[0] in (*INTENTS, "invalid")


def test_precision_resolution(monkeypatch):
    assert llm.resolve_precision("fp16") == "fp16"
    monkeypatch.setattr(llm, "native_bf16", lambda: False)
    assert llm.resolve_precision("auto:nf4-fp32") == "nf4-fp32"
    monkeypatch.setattr(llm, "native_bf16", lambda: True)
    assert llm.resolve_precision("auto:fp32") == "bf16"
    for bad in ("fp8", "auto:int3"):
        with pytest.raises(ValueError):
            llm.resolve_precision(bad)


def test_native_bf16_needs_every_gpu_at_compute_capability_8(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert llm.native_bf16() is False
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (7, 5) if i else (8, 0))
    assert llm.native_bf16() is False          # one T4 is enough to rule bf16 out
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda i: (9, 0))
    assert llm.native_bf16() is True


def test_load_llm_passes_revision_precision_and_remote_code(monkeypatch):
    seen = {}

    class Fake:
        def eval(self):
            return self

    def tok_from(model_id, **kw):
        seen["tok"] = kw
        return "tok"

    def model_from(model_id, **kw):
        seen["model"] = kw
        return Fake()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tok_from)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", model_from)
    monkeypatch.setattr(llm, "native_bf16", lambda: True)
    llm.load_llm("inception42/Jais-3-7B", "auto:nf4-fp32", revision="abc", trust_remote_code=True)
    assert seen["tok"]["trust_remote_code"] and seen["tok"]["revision"] == "abc"
    assert seen["model"]["dtype"] == torch.bfloat16 and seen["model"]["trust_remote_code"]
    llm.load_llm("Qwen/Qwen3-8B", "fp32")
    assert seen["model"]["dtype"] == torch.float32 and not seen["model"]["trust_remote_code"]
    llm.load_llm("inception42/Jais-2-8B-Chat", "nf4-fp32")
    assert seen["model"]["quantization_config"].bnb_4bit_compute_dtype == torch.float32


def test_verbalizer_lengths(pilot_and_lm):
    _, _, tok = pilot_and_lm
    assert llm.verbalizer_lengths(tok) == {"create_reminder": 1, "create_alarm": 1, "unsupported": 2}
    assert set(llm.verbalizer_lengths(tok, "number").values()) == {1}


def test_hf_tool_backend_renders_jais3_tools_and_generates(pilot_and_lm):
    from athar import agent
    from athar.timeexpr import cairo

    items, model, _ = pilot_and_lm
    tok = jais3_tokenizer([norm(it.text) for it in items])
    msgs = agent.build_messages(items[0].text, cairo("2026-10-07T10:00"))
    for fmt, marker in (("json", '{"name": <function-name>'), ("xml", "<ifm|arg_key>$PARAMETER_NAME")):
        backend = agent.HFToolBackend(model, tok, template_kwargs={"reasoning_effort": "high", "tool_call_format": fmt})
        rendered = backend.render(msgs)
        assert rendered.startswith("<|ifm|begin_of_text|><|ifm|im_start|>system\n# Tools\n")
        assert all(name in rendered for name in agent.TOOL_NAMES) and marker in rendered
        assert "Wednesday 2026-10-07 at 10:00" in rendered and rendered.endswith("assistant\n<ifm|think>\n")
    _, plain_tok = tiny_lm([norm(it.text) for it in items])
    gen = agent.HFToolBackend(model, plain_tok, max_new_tokens=3, batch_size=2)
    out = gen.complete([agent.build_messages(it.text, it.ref_datetime) for it in items[:3]])
    assert len(out) == 3 and all(c.n_tokens <= 3 and c.finish_reason in ("stop", "length") for c in out)
    sampled = agent.HFToolBackend(model, plain_tok, max_new_tokens=2, sampling={"temperature": 0.7, "top_k": 20})
    assert len(sampled.complete([agent.build_messages(items[0].text, items[0].ref_datetime)])) == 1
