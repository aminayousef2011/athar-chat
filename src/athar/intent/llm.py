"""S4 (and exploratory S7): instruction-tuned LLMs as intent recognizers.

The LLM predicts only the intent label, like every other system; the route and the arguments
still come from the shared policy and extractor, so S4 is directly comparable to S1-S6.

Methods (all reported). ``prior`` is the primary S4 number for the locked test run (protocol
deviations log, 2026-10-11; before it ``raw``, and the pre-registered ``calibrated``).
Contextual calibration assumes a content-free input deserves a uniform label distribution,
which fails for a label set with an abstain class: an empty input *is* out of scope (Jais 2
gives it 0.99 ``unsupported``), so dividing by it suppresses abstention. ``prior`` estimates
the label bias from real unlabeled train requests instead.
- ``raw``: constrained label scoring. The answer is the label whose verbalizer has the highest
  summed log-probability after the chat-formatted prompt (lm-evaluation-harness
  ``multiple_choice``-style scoring; no parsing, no invalid outputs).
- ``calibrated``: the same scores after contextual calibration (Zhao et al. 2021, "Calibrate
  Before Use"): label probabilities are divided by those the model gives to content-free inputs
  ("N/A", "", "[MASK]") under the same prompt and shots, then renormalized.
- ``prior``: calibration with a label prior estimated from real, unlabeled inputs instead of
  content-free ones (batch calibration, Zhou et al. 2024, ICLR), here a fixed sample of *train*
  texts (``prior_texts``), so no test-time information is used.
- ``generate``: greedy generation, parsed to a label. Output that names no label is
  ``"invalid"`` and scores as a failure. Reported because first-token scores and generated
  answers of chat models can disagree (Wang et al. 2024, arXiv 2402.14499).

Scoring ablation (protocol deviations log, 2026-10-09; not the pre-registered S4 number): the
summed score favours short verbalizers, and ``غير مدعوم`` is longer than the in-scope labels
in the tokenizers checked. ``length_norm=True`` scores the mean log-probability per verbalizer
token instead; ``answer_format="number"`` numbers the options 1-3 and scores a one-token digit
answer, so the labels have the same length.

Few-shot exemplars come from the train split only (fixed seed, one per family, balanced over
labels). The models are loaded at pinned revisions; see notebooks/kaggle_s4_llm.ipynb.

Chat profiles (protocol deviations log, 2026-10-15). Reasoning models need two things the
older chat models did not, so each model id maps to a :class:`ChatProfile`:
- Jais 3 (``inception42/Jais-3-*``) opens a thinking block in its generation prompt
  (``<ifm|think>``) and requires a thinking field on every assistant turn. Its profile gives
  each few-shot answer an empty thinking field and scores the label after ``</ifm|think>``, so
  the scored text is exactly how the template renders an answer with no reasoning.
- Jais 2 (``inception42/Jais-2-*``): its template writes each message's role literally but
  opens the answer as ``<|start_header_id|>ai<|end_header_id|>``, so few-shot answers sent
  with role ``assistant`` appeared under a header the model is never asked to answer under
  (seen in the 2026-10-15 prompt check). Its profile sends the few-shot answers with role
  ``ai``, the template's own answer role. Before this fix (all Jais 2 runs up to 2026-10-12)
  the few-shot prompts mixed the two; zero-shot prompts and S7 were unaffected.
- Qwen3 turns thinking off through its template (``enable_thinking=False``), as before.
Nile-Chat and Qwen3 prompts are byte-identical to the runs before 2026-10-15 (tested).
``shot_style="inline"`` (a prompt-format check) puts the exemplars in the user message instead
of as earlier chat turns; ``"turns"`` stays the primary.
"""

from __future__ import annotations

import random
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from athar.normalize import norm
from athar.schema import INTENTS, GoldItem

#: Arabic verbalizers (one answer word/phrase per label)
VERBALIZERS: dict[str, str] = {
    "create_reminder": "تذكير",
    "create_alarm": "منبه",
    "unsupported": "غير مدعوم",
}
#: one-token answers for ``answer_format="number"``
NUMBERS: dict[str, str] = {"create_reminder": "1", "create_alarm": "2", "unsupported": "3"}
CONTENT_FREE = ("N/A", "", "[MASK]")

SYSTEM_PROMPT = """إنت مساعد بيصنّف طلب المستخدم لأداة تذكيرات ومنبهات. رد بكلمة واحدة بس من دول:
تذكير: لو المستخدم عايز تذكير واحد ليه هو بحاجة، مرة واحدة (حتى لو الوقت أو المحتوى ناقص).
منبه: لو عايز منبه أو يصحى في وقت معين، مرة واحدة.
غير مدعوم: لو الطلب مسح أو تعديل لتذكير أو منبه موجود، أو متكرر (كل يوم، كل أسبوع)، أو لحد تاني، أو فيه أكتر من طلب، أو محتاج أداة تانية (تقويم، إيميل، تايمر، الطقس، مكالمة).
متكتبش أي حاجة غير الكلمة."""

SYSTEM_PROMPT_NUMBER = """إنت مساعد بيصنّف طلب المستخدم لأداة تذكيرات ومنبهات. رد برقم واحد بس:
1: تذكير، لو المستخدم عايز تذكير واحد ليه هو بحاجة، مرة واحدة (حتى لو الوقت أو المحتوى ناقص).
2: منبه، لو عايز منبه أو يصحى في وقت معين، مرة واحدة.
3: غير مدعوم، لو الطلب مسح أو تعديل لتذكير أو منبه موجود، أو متكرر (كل يوم، كل أسبوع)، أو لحد تاني، أو فيه أكتر من طلب، أو محتاج أداة تانية (تقويم، إيميل، تايمر، الطقس، مكالمة).
متكتبش أي حاجة غير الرقم."""


def select_shots(train: Sequence[GoldItem], per_label: int = 2, seed: int = 0) -> list[tuple[str, str]]:
    """Balanced few-shot exemplars from train: ``per_label`` per intent, distinct families,
    interleaved (reminder, alarm, unsupported, ...) so no label sits last by construction."""
    rng = random.Random(seed)
    by_label: dict[str, list[GoldItem]] = {lab: [] for lab in INTENTS}
    for it in sorted(train, key=lambda x: x.id):
        by_label[it.intent].append(it)
    used: set[str] = set()
    picked: dict[str, list[GoldItem]] = {}
    for lab in INTENTS:
        pool = by_label[lab][:]
        rng.shuffle(pool)
        picked[lab] = []
        for it in pool:
            if len(picked[lab]) < per_label and it.family_id not in used:
                picked[lab].append(it)
                used.add(it.family_id)
    shots = []
    for k in range(per_label):
        for lab in INTENTS:
            if k < len(picked[lab]):
                shots.append((norm(picked[lab][k].text), lab))
    return shots


#: ``shot_style="inline"``: the exemplars as one block in the user message
INLINE_HEADER = "أمثلة:"
INLINE_REQUEST = "الطلب:"
INLINE_ANSWER = "الرد:"


def build_messages(text: str, shots: Sequence[tuple[str, str]], answer_format: str = "word", *,
                   shot_style: str = "turns", assistant_fields: Mapping[str, str] | None = None,
                   assistant_role: str = "assistant") -> list[dict[str, str]]:
    """Chat messages for one request. ``turns``: each exemplar is a user/assistant exchange;
    ``inline``: the exemplars are listed in the user message (no shots: both are the same).
    ``assistant_fields`` are added to every assistant turn (Jais 3: an empty thinking field);
    ``assistant_role`` is the role those turns carry (Jais 2: ``ai``)."""
    if shot_style not in ("turns", "inline"):
        raise ValueError(f"unknown shot_style {shot_style!r}")
    answers = NUMBERS if answer_format == "number" else VERBALIZERS
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_NUMBER if answer_format == "number" else SYSTEM_PROMPT}]
    if shot_style == "inline" and shots:
        blocks = "\n\n".join(f"{INLINE_REQUEST} {shot_text}\n{INLINE_ANSWER} {answers[lab]}" for shot_text, lab in shots)
        msgs.append({"role": "user", "content": f"{INLINE_HEADER}\n\n{blocks}\n\n{INLINE_REQUEST} {text}"})
        return msgs
    for shot_text, lab in shots:
        msgs += [{"role": "user", "content": shot_text},
                 {"role": assistant_role, "content": answers[lab], **(assistant_fields or {})}]
    msgs.append({"role": "user", "content": text})
    return msgs


@dataclass(frozen=True)
class ChatProfile:
    """How a model's chat template is driven (see the module docstring)."""

    template_kwargs: Mapping[str, Any] = field(default_factory=dict)
    #: appended to the rendered generation prompt before the label is scored or generated
    answer_prefix: str = ""
    #: added to every few-shot assistant turn
    assistant_fields: Mapping[str, str] = field(default_factory=dict)
    #: the role few-shot answers carry: the one the template opens the answer with
    assistant_role: str = "assistant"
    trust_remote_code: bool = False

    def intent_kwargs(self) -> dict[str, Any]:
        """Keyword arguments for :class:`LLMIntent`."""
        return {"template_kwargs": dict(self.template_kwargs), "answer_prefix": self.answer_prefix,
                "assistant_fields": dict(self.assistant_fields), "assistant_role": self.assistant_role}


#: Jais 3 renders an answer with no reasoning as ``<ifm|think>\n</ifm|think>answer``; its
#: generation prompt (reasoning_effort "high") ends with ``<ifm|think>\n``
JAIS3_THINK_CLOSE = "</ifm|think>"
CHAT_PROFILES: dict[str, ChatProfile] = {
    "inception42/Jais-3-": ChatProfile(template_kwargs={"reasoning_effort": "high"},
                                       answer_prefix=JAIS3_THINK_CLOSE,
                                       assistant_fields={"reasoning_content": ""}, trust_remote_code=True),
    "inception42/Jais-2-": ChatProfile(assistant_role="ai"),
    "Qwen/Qwen3-": ChatProfile(template_kwargs={"enable_thinking": False}),
}


def chat_profile(model_id: str) -> ChatProfile:
    """The profile whose key is a prefix of ``model_id``; the empty default otherwise."""
    for prefix, profile in CHAT_PROFILES.items():
        if model_id.startswith(prefix):
            return profile
    return ChatProfile()


def answer_consistent(tokenizer: Any, profile: ChatProfile, text: str = "فكرني بكرة",
                      answer: str = VERBALIZERS["create_reminder"]) -> bool:
    """True when "prompt + answer" is how the model's own template writes a finished answer
    turn, i.e. the label is scored where the template puts an answer, and few-shot answers
    (built the same way) look like the model's own turns. Template-agnostic: run it on every
    model's real template (the notebook's prompt check does)."""
    kw = dict(profile.template_kwargs)
    msgs = build_messages(text, [], assistant_fields=profile.assistant_fields, assistant_role=profile.assistant_role)
    prompt = str(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw))
    turn = {"role": profile.assistant_role, "content": answer, **profile.assistant_fields}
    done = str(tokenizer.apply_chat_template(msgs + [turn], tokenize=False, **kw))
    return done.startswith(prompt + profile.answer_prefix + answer)


def parse_label(generated: str) -> str:
    """Map generated text to a label: the earliest verbalizer it contains, else ``invalid``."""
    g = norm(generated)
    hits = []
    for lab, word in VERBALIZERS.items():
        m = re.search(re.escape(norm(word)), g)
        if m:
            hits.append((m.start(), -len(word), lab))  # "غير مدعوم" beats a later "تذكير"
    return min(hits)[2] if hits else "invalid"


class LLMIntent:
    """An instruction-tuned causal LM behind the IntentRecognizer protocol."""

    def __init__(self, model: Any, tokenizer: Any, name: str = "S4", *,
                 shots: Sequence[tuple[str, str]] = (), method: str = "calibrated",
                 template_kwargs: dict[str, Any] | None = None, max_new_tokens: int = 12,
                 length_norm: bool = False, answer_format: str = "word",
                 prior_texts: Sequence[str] = (), answer_prefix: str = "",
                 assistant_fields: Mapping[str, str] | None = None, shot_style: str = "turns",
                 assistant_role: str = "assistant") -> None:
        if shot_style not in ("turns", "inline"):
            raise ValueError(f"unknown shot_style {shot_style!r}")
        if method not in ("raw", "calibrated", "prior", "generate"):
            raise ValueError(f"unknown method {method!r}")
        if answer_format not in ("word", "number"):
            raise ValueError(f"unknown answer_format {answer_format!r}")
        if method == "generate" and answer_format != "word":
            raise ValueError("generate parses words; use answer_format='word'")
        if method == "prior" and not prior_texts:
            raise ValueError("method 'prior' needs prior_texts (unlabeled train texts)")
        self.length_norm, self.answer_format = length_norm, answer_format
        self.prior_texts = list(prior_texts)
        self._prior: NDArray[np.float64] | None = None
        self.name, self.model, self.tok = name, model, tokenizer
        self.shots, self.method = list(shots), method
        self.template_kwargs = template_kwargs or {}
        self.answer_prefix, self.shot_style = answer_prefix, shot_style
        self.assistant_fields, self.assistant_role = dict(assistant_fields or {}), assistant_role
        self.max_new_tokens = max_new_tokens
        # per-token log-probabilities of each answer, by text: instances with the same prompt,
        # shots and answer format can share it, whatever their method or length_norm
        self._cache: dict[str, list[NDArray[np.float64]]] = {}
        self._cf: NDArray[np.float64] | None = None
        answers = NUMBERS if answer_format == "number" else VERBALIZERS
        self.label_ids = {lab: self.tok(w, add_special_tokens=False)["input_ids"] for lab, w in answers.items()}

    # -- prompt ---------------------------------------------------------------------------
    def prompt(self, text: str) -> str:
        """The exact text the label is scored (or generated) after."""
        msgs = build_messages(text, self.shots, self.answer_format, shot_style=self.shot_style,
                              assistant_fields=self.assistant_fields, assistant_role=self.assistant_role)
        return str(self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                **self.template_kwargs)) + self.answer_prefix

    def _device(self) -> Any:
        return next(self.model.parameters()).device

    # -- constrained scoring -------------------------------------------------------------
    def label_logprobs(self, text: str) -> NDArray[np.float64]:
        """Log-probability of each answer after the prompt, in INTENTS order: summed over its
        tokens, or their mean with ``length_norm``."""
        per_token = self.token_logprobs(text)
        agg = np.mean if self.length_norm else np.sum
        return np.array([float(agg(t)) for t in per_token], dtype=np.float64)

    def token_logprobs(self, text: str) -> list[NDArray[np.float64]]:
        if text in self._cache:
            return self._cache[text]
        import torch

        p_ids = self.tok(self.prompt(text), add_special_tokens=False)["input_ids"]
        scores = []
        with torch.no_grad():
            for lab in INTENTS:
                c_ids = self.label_ids[lab]
                ids = torch.tensor([p_ids + c_ids], device=self._device())
                logits = self.model(input_ids=ids).logits[0].float()
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"{self.name}: non-finite logits (fp16 overflow?); "
                                             f"reload in 4-bit or fp32")
                lp = torch.log_softmax(logits[len(p_ids) - 1:len(p_ids) - 1 + len(c_ids)], dim=-1)
                tok_lp = lp.gather(1, torch.tensor(c_ids, device=lp.device)[:, None])[:, 0]
                scores.append(np.asarray(tok_lp.cpu().numpy(), dtype=np.float64))
        self._cache[text] = scores
        return scores

    def content_free(self) -> NDArray[np.float64]:
        if self._cf is None:
            p = np.mean([_softmax(self.label_logprobs(t)) for t in CONTENT_FREE], axis=0)
            self._cf = p
        return self._cf

    def label_prior(self) -> NDArray[np.float64]:
        """Mean label distribution over ``prior_texts`` (batch-calibration prior)."""
        if self._prior is None:
            self._prior = np.mean([_softmax(self.label_logprobs(t)) for t in self.prior_texts], axis=0)
        return self._prior

    def probabilities(self, text: str) -> NDArray[np.float64]:
        p = _softmax(self.label_logprobs(text))
        if self.method in ("calibrated", "prior"):
            ref = self.content_free() if self.method == "calibrated" else self.label_prior()
            p = p / np.maximum(ref, 1e-12)
            p = p / p.sum()
        return p

    # -- generation ----------------------------------------------------------------------
    def generate(self, text: str) -> str:
        import torch

        enc = self.tok(self.prompt(text), add_special_tokens=False, return_tensors="pt").to(self._device())
        with torch.no_grad():  # input ids and mask only: some remote-code models reject token_type_ids
            out = self.model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                      max_new_tokens=self.max_new_tokens, do_sample=False,
                                      pad_token_id=self.tok.pad_token_id or self.tok.eos_token_id)
        return str(self.tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True))

    # -- IntentRecognizer ----------------------------------------------------------------
    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        if self.method == "generate":
            return [parse_label(self.generate(t)) for t in normalized_texts]
        return [INTENTS[int(np.argmax(self.probabilities(t)))] for t in normalized_texts]

    def predict_proba(self, normalized_texts: Sequence[str], labels: Sequence[str]) -> list[list[float]]:
        rows = []
        for t in normalized_texts:
            p = self.probabilities(t) if self.method != "generate" else _softmax(self.label_logprobs(t))
            rows.append([float(p[INTENTS.index(lab)]) for lab in labels])
        return rows


def _softmax(z: NDArray[np.float64]) -> NDArray[np.float64]:
    e = np.exp(z - z.max())
    return np.asarray(e / e.sum(), dtype=np.float64)


# --------------------------------------------------------------------------- loading
PRECISIONS = ("fp16", "bf16", "fp32", "nf4", "nf4-fp32")


def native_bf16() -> bool:
    """True when every visible GPU computes bf16 natively (compute capability 8.0+: A100, L4,
    H100, ...). T4s (7.5) do not, which is why the Kaggle runs need fp16/fp32/4-bit."""
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return False
    return all(torch.cuda.get_device_capability(i) >= (8, 0) for i in range(torch.cuda.device_count()))


def resolve_precision(precision: str) -> str:
    """``auto:<fallback>`` -> bf16 on GPUs with native bf16 (the precision the models were
    released in), else the fallback; a plain precision is returned unchanged."""
    if precision.startswith("auto:"):
        fallback = precision.removeprefix("auto:")
        if fallback not in PRECISIONS:
            raise ValueError(f"unknown fallback precision {fallback!r}")
        return "bf16" if native_bf16() else fallback
    if precision not in PRECISIONS:
        raise ValueError(f"unknown precision {precision!r}")
    return precision


def load_tokenizer(model_id: str, token: str | None = None, revision: str | None = None,
                   trust_remote_code: bool = False) -> Any:
    """The tokenizer (and chat template) alone: enough to render and inspect prompts."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, token=token, revision=revision, trust_remote_code=trust_remote_code)


def verbalizer_lengths(tokenizer: Any, answer_format: str = "word") -> dict[str, int]:
    """Tokens per label answer: the summed score favours short answers (protocol log 2026-10-09)."""
    answers = NUMBERS if answer_format == "number" else VERBALIZERS
    return {lab: len(tokenizer(w, add_special_tokens=False)["input_ids"]) for lab, w in answers.items()}


def load_llm(model_id: str, precision: str = "fp16", token: str | None = None,
             revision: str | None = None, trust_remote_code: bool = False) -> tuple[Any, Any]:
    """Load a chat LLM for scoring. ``precision``: fp16 / bf16 / fp32 (sharded over the visible
    GPUs with ``device_map="auto"``), nf4 (bitsandbytes 4-bit, fp16 compute), nf4-fp32 (4-bit
    weights, fp32 compute: no fp16 overflow, and it fits an 8B model on T4s where fp32 does not),
    or ``auto:<fallback>`` (:func:`resolve_precision`). ``trust_remote_code`` runs the model's
    own code from the Hub (Jais 3); pin ``revision`` so that code cannot change."""
    import torch
    from transformers import AutoModelForCausalLM

    precision = resolve_precision(precision)
    tok = load_tokenizer(model_id, token=token, revision=revision, trust_remote_code=trust_remote_code)
    kw: dict[str, Any] = {"device_map": "auto", "token": token, "revision": revision,
                          "trust_remote_code": trust_remote_code}
    if precision in ("nf4", "nf4-fp32"):
        from transformers import BitsAndBytesConfig

        compute = torch.float32 if precision == "nf4-fp32" else torch.float16
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                       bnb_4bit_compute_dtype=compute)
    else:
        kw["dtype"] = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[precision]
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw).eval()
    return model, tok


# --------------------------------------------------------------------------- S7 (exploratory)
def lora_examples(tokenizer: Any, train: Sequence[GoldItem]) -> list[tuple[list[int], list[int]]]:
    """(input ids, labels) per training item: the zero-shot chat prompt followed by the label's
    verbalizer and EOS. The prompt is masked (-100), so the loss covers the answer tokens only."""
    helper = LLMIntent(_NoModel(), tokenizer, shots=())
    out = []
    for it in train:
        p_ids = tokenizer(helper.prompt(norm(it.text)), add_special_tokens=False)["input_ids"]
        c_ids = helper.label_ids[it.intent] + ([tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else [])
        out.append((p_ids + c_ids, [-100] * len(p_ids) + c_ids))
    return out


class _NoModel:
    """Placeholder model for prompt construction only."""


def lora_finetune(model: Any, tokenizer: Any, train: Sequence[GoldItem], *, epochs: int = 3,
                  learning_rate: float = 2e-4, r: int = 16, alpha: int = 32, dropout: float = 0.05,
                  seed: int = 0, target_modules: Sequence[str] | str = "all-linear",
                  log: Any = print) -> Any:
    """LoRA on the zero-shot prompt -> verbalizer task, loss on the verbalizer tokens only.
    Returns the adapted model (use it with ``LLMIntent(..., shots=())``)."""
    import torch
    from peft import LoraConfig, get_peft_model

    torch.manual_seed(seed)
    random.seed(seed)
    if getattr(model, "is_loaded_in_4bit", False):
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model)
    modules = target_modules if isinstance(target_modules, str) else list(target_modules)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
                                             task_type="CAUSAL_LM", target_modules=modules))
    examples = lora_examples(tokenizer, train)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=learning_rate)
    device = next(model.parameters()).device
    model.train()
    for epoch in range(epochs):
        random.shuffle(examples)
        total = 0.0
        for ids, labels in examples:
            out = model(input_ids=torch.tensor([ids], device=device), labels=torch.tensor([labels], device=device))
            out.loss.backward()
            opt.step()
            opt.zero_grad()
            total += out.loss.item()
        log(f"  LoRA epoch {epoch + 1}/{epochs} loss {total / len(examples):.4f}")
    return model.eval()
