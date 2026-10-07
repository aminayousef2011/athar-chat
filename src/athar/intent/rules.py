"""S1: hand-written keyword rules for intent recognition.

The lists are written from the task contract's scope (section 1), not from the dataset, and may
be extended after reading the *train* split only. They are frozen with the models (their file
hash is recorded in models/freeze.json) before the test split is opened.

Decision, on normalized text:
1. a recurrence cue or a request for another person anywhere -> unsupported;
2. a deletion, modification or other-tool cue *before* the first self-request cue (or with no
   self-request cue at all) -> unsupported. After ``فكرني`` the same words are reminder content:
   ``فكرني اتصل بماما`` is a reminder, ``اتصل بماما`` is another tool;
3. two self-request verbs (``فكرني ... وصحيني``) -> unsupported (multi_action);
4. otherwise the kind of the first self-request cue: alarm or reminder; none -> unsupported.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

from athar.lexicon import _n, alt

B, E = r"(?<!\w)", r"(?!\w)"
CONJ = r"(?:و\s?)?"

#: self-request cues that name the tool
ALARM_CUES = _n([
    "منبه", "المنبه", "منبة", "alarm", "an alarm", "the alarm", "الارم", "ألارم", "صحيني", "صحّيني",
    "قومني", "صحيني بدري", "wake me", "wake me up", "صحيه", "يرن",
])
REMINDER_CUES = _n([
    "فكرني", "فكريني", "تفكرني", "تفكريني", "ذكرني", "ذكريني", "تذكرني", "تذكريني", "متنسانيش",
    "ما تنسانيش", "نبهني", "نبهيني", "تذكير", "ريمايندر", "ريماندر", "reminder", "a reminder",
    "remind me",
])
#: verbs, the subset of the cues that each count as one request (for multi_action)
REQUEST_VERBS = _n([
    "فكرني", "فكريني", "تفكرني", "تفكريني", "ذكرني", "ذكريني", "تذكرني", "تذكريني", "متنسانيش",
    "نبهني", "نبهيني", "صحيني", "صحّيني", "قومني", "remind me", "wake me",
])

RECURRING = _n([
    "كل يوم", "كل أسبوع", "كل اسبوع", "كل شهر", "كل سنة", "كل ساعة", "كل ليلة", "كل يوم الصبح",
    "كل سبت", "كل حد", "كل اتنين", "كل تلات", "كل أربع", "كل خميس", "كل جمعة", "يوميا", "يومياً",
    "أسبوعيا", "اسبوعيا", "شهريا", "على طول كل", "every", "daily", "weekly", "monthly",
    "every day", "everyday",
])
#: imperative "remind" with no first-person object, or "remind" + someone else
OTHER_PERSON = _n([
    "فكر", "فكري", "فكره", "فكرها", "فكرهم", "ذكر", "ذكره", "ذكرها", "ذكرهم", "remind him",
    "remind her", "remind them",
])
RX_REMIND_OTHER = re.compile(rf"{B}remind\s+(?!me{E})\w+")

DELETION = _n([
    "امسح", "امسحي", "امسحلي", "مسح", "الغي", "الغى", "إلغي", "ألغي", "لغي", "شيل", "شيلي",
    "متفكرنيش", "ما تفكرنيش", "delete", "cancel", "remove", "turn off", "stop",
])
MODIFICATION = _n([
    "غير", "غيري", "غيرلي", "عدل", "عدلي", "عدللي", "أجل", "اجل", "أجللي", "بدل", "خليه بدل",
    "change", "edit", "move", "reschedule", "postpone", "snooze", "update",
])
OTHER_TOOL = _n([
    "تايمر", "timer", "stopwatch", "ستوب ووتش", "عداد", "الجو", "الطقس", "weather", "اتصل",
    "اتصلي", "كلملي", "رن على", "call", "ابعت", "ابعتلي", "ابعتله", "رسالة", "مسج", "message",
    "text", "send", "email", "ايميل", "شغل", "شغلي", "شغللي", "play", "music", "اغنية",
])


def _rx(words: Sequence[str]) -> re.Pattern[str]:
    return re.compile(rf"{B}{CONJ}{alt(words)}{E}")


RX_ALARM, RX_REMINDER = _rx(ALARM_CUES), _rx(REMINDER_CUES)
RX_VERB = _rx(REQUEST_VERBS)
RX_RECURRING, RX_OTHER_PERSON = _rx(RECURRING), _rx(OTHER_PERSON)
RX_BLOCKERS = _rx(DELETION + MODIFICATION + OTHER_TOOL)


def classify(nt: str) -> str:
    """Intent of one normalized text."""
    if RX_RECURRING.search(nt) or RX_OTHER_PERSON.search(nt) or RX_REMIND_OTHER.search(nt):
        return "unsupported"
    cues = [(m.start(), "create_alarm") for m in RX_ALARM.finditer(nt)]
    cues += [(m.start(), "create_reminder") for m in RX_REMINDER.finditer(nt)]
    first = min(cues)[0] if cues else len(nt) + 1
    blocker = RX_BLOCKERS.search(nt)
    if blocker and blocker.start() < first:
        return "unsupported"
    if len({m.group().removeprefix("و").strip() for m in RX_VERB.finditer(nt)}) > 1:
        return "unsupported"
    return min(cues)[1] if cues else "unsupported"


class KeywordIntent:
    """S1. Stateless: nothing is learned, so there is no model file."""

    name = "S1"

    def predict(self, normalized_texts: Sequence[str]) -> list[str]:
        return [classify(t) for t in normalized_texts]
