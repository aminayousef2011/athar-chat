"""The post hoc learned reminder-text tagger (scripts/posthoc_learned_text.py): its word labels follow
the gold reminder text, and a tagger trained on a few messages returns the content words."""

import importlib.util
from pathlib import Path

import pytest

from athar.schema import GoldItem

_SPEC = importlib.util.spec_from_file_location(
    "posthoc_learned_text", Path(__file__).resolve().parents[1] / "scripts/posthoc_learned_text.py")
P = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(P)

REF = "2026-10-07T10:00:00+03:00"


def item(i: int, text: str, route: str = "ACT", reminder_text: str | None = None,
         targets: list[str] | None = None, intent: str = "create_reminder") -> GoldItem:
    row = {"id": f"t{i:03d}", "family_id": f"fam-{i}", "text": text, "ref_datetime": REF,
           "intent": intent, "route": route}
    if route == "ACT":
        row["args"] = {"datetime": "2026-10-08T09:00:00+03:00", **({"reminder_text": reminder_text} if reminder_text else {})}
    else:
        row["clarify_targets"] = targets or []
    return GoldItem.model_validate(row)


def test_labels_mark_the_gold_reminder_words():
    it = item(1, "فكرني بكرة الساعة 9 الصبح اراجع التقرير", reminder_text="اراجع التقرير")
    toks, timed, _, _ = P.tokens(it)
    lab = P.labels(it, toks)
    assert [w for (_, _, w), y in zip(toks, lab, strict=True) if y] == ["اراجع", "التقرير"]
    assert timed and not any(lab[i] for i in timed)   # time words are never content


def test_labels_strip_a_leading_ba_as_the_metric_does():
    it = item(2, "فكرني بالاجتماع بكرة الساعة 9 الصبح", reminder_text="الاجتماع")
    toks, *_ = P.tokens(it)
    assert sum(P.labels(it, toks)) == 1


def test_missing_reminder_text_is_all_negative_and_alarms_are_skipped():
    clar = item(3, "اعمللي تذكير", route="CLARIFY", targets=["time", "reminder_text"])
    toks, *_ = P.tokens(clar)
    assert P.labels(clar, toks) == [0] * len(toks)
    alarm = item(4, "صحيني بكرة الساعة 9 الصبح", intent="create_alarm")
    assert P.labels(alarm, P.tokens(alarm)[0]) is None


@pytest.mark.parametrize("rule", [False, True])
def test_tagger_learns_content_words(rule):
    tasks = ["اراجع التقرير", "اكلم العميل", "ابعت الايميل", "اجهز العرض", "اقفل الفاتورة", "احضر الاجتماع"]
    train = [item(10 + i, f"فكرني بكرة الساعة 9 الصبح {t}", reminder_text=t) for i, t in enumerate(tasks)]
    train += [item(20 + i, f"فكرني {t} بكرة الساعة 9 الصبح", reminder_text=t) for i, t in enumerate(tasks)]
    tagger = P.TextTagger(rule)
    stats = tagger.fit(train)
    assert stats["messages"] == len(train)
    out = tagger.text(item(99, "فكرني بكرة الساعة 9 الصبح اكلم العميل", reminder_text="اكلم العميل"))
    assert out is not None and "فكرني" not in out and "الساعه" not in out
