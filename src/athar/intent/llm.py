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
"""

from __future__ import annotations

import random
import re
from collections.abc import Sequence
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


def build_messages(text: str, shots: Sequence[tuple[str, str]], answer_format: str = "word") -> list[dict[str, str]]:
    answers = NUMBERS if answer_format == "number" else VERBALIZERS
    msgs = [{"role": "system", "content": SYSTEM_PROMPT_NUMBER if answer_format == "number" else SYSTEM_PROMPT}]
    for shot_text, lab in shots:
        msgs += [{"role": "user", "content": shot_text}, {"role": "assistant", "content": answers[lab]}]
    msgs.append({"role": "user", "content": text})
    return msgs


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
                 prior_texts: Sequence[str] = ()) -> None:
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
        self.max_new_tokens = max_new_tokens
        # per-token log-probabilities of each answer, by text: instances with the same prompt,
        # shots and answer format can share it, whatever their method or length_norm
        self._cache: dict[str, list[NDArray[np.float64]]] = {}
        self._cf: NDArray[np.float64] | None = None
        answers = NUMBERS if answer_format == "number" else VERBALIZERS
        self.label_ids = {lab: self.tok(w, add_special_tokens=False)["input_ids"] for lab, w in answers.items()}

    # -- prompt ---------------------------------------------------------------------------
    def prompt(self, text: str) -> str:
        return str(self.tok.apply_chat_template(build_messages(text, self.shots, self.answer_format), tokenize=False,
                                                add_generation_prompt=True, **self.template_kwargs))

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
        with torch.no_grad():
            out = self.model.generate(**enc, max_new_tokens=self.max_new_tokens, do_sample=False,
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
def load_llm(model_id: str, precision: str = "fp16", token: str | None = None,
             revision: str | None = None) -> tuple[Any, Any]:
    """Load a chat LLM for scoring. ``precision``: fp16 / bf16 / fp32 (sharded over the visible
    GPUs with ``device_map="auto"``), nf4 (bitsandbytes 4-bit, fp16 compute) or nf4-fp32 (4-bit
    weights, fp32 compute: no fp16 overflow, and it fits an 8B model on T4s where fp32 does not)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id, token=token, revision=revision)
    kw: dict[str, Any] = {"device_map": "auto", "token": token, "revision": revision}
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
