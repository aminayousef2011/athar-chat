"""Detection and resolution of the supported date/time expressions (task contract section 3).

``detect`` finds non-overlapping mentions in *normalized* text, in priority order:
relative offsets, explicit dates, day words, weekdays, vague dates, noon/midnight, clock times,
day periods, vague times. ``resolve`` combines them with the reference datetime into one
timezone-aware datetime or a set of clarification gaps. The resolver never guesses: anything
the contract does not resolve becomes a gap.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from zoneinfo import ZoneInfo

from athar import lexicon as L

# clarification gap names (the closed vocabulary of the task contract)
DATE, TIME, AM_PM, PAST, REMINDER_TEXT = "date", "time", "am_pm", "past", "reminder_text"
GAP_ORDER = (DATE, TIME, AM_PM, PAST, REMINDER_TEXT)

B = r"(?<!\w)"
E = r"(?!\w)"
CONJ = r"(?:و\s?)?"  # optional conjunction proclitic: "وبكره", "و بكره"


@dataclass(frozen=True)
class Clock:
    hour: int  # hour as said: 1-12 on the 12 h clock, 0-23 when is_24h
    minute: int  # offset in minutes, may be negative ("7 إلا ربع" -> hour 7, minute -15)
    is_24h: bool
    marker: str | None = None  # "am" / "pm" from ص/م/am/pm directly after the time
    early: bool = False  # midnight words: belong to the end of a night


@dataclass(frozen=True)
class DateRef:
    candidates: frozenset[date] | None  # None = ambiguous by contract
    why: str


@dataclass(frozen=True)
class Mention:
    kind: str  # relative | date | clock | period | vague_time
    start: int
    end: int
    value: object = None


@dataclass(frozen=True)
class Temporal:
    datetime: datetime | None
    gaps: frozenset[str]
    mentions: tuple[Mention, ...]

    @property
    def spans(self) -> list[tuple[int, int]]:
        return [(m.start, m.end) for m in self.mentions]


# --------------------------------------------------------------------------- patterns
_AR = lambda d: [k for k in d if not k.isascii()]  # noqa: E731
_EN = lambda d: [k for k in d if k.isascii()]  # noqa: E731

_UNITS_MIN = L.alt(L.MINUTE_UNITS)
_UNITS_HOUR = L.alt(L.HOUR_UNITS)
_FRACTIONS = {"نص": 0.5, "نصف": 0.5, "ربع": 0.25, "تلت": 1 / 3, "ثلث": 1 / 3, "half": 0.5, "half an": 0.5}
_AMT = rf"(?P<amt>\d{{1,3}}|{L.alt(L.AMOUNT_WORDS)})"

RX_RELATIVE = [
    re.compile(
        rf"{B}{CONJ}بعد\s+(?:{_AMT}\s*(?P<unit>{_UNITS_MIN}|{_UNITS_HOUR})"
        rf"|(?P<dual>{L.alt(L.MINUTE_UNITS_DUAL + L.HOUR_UNITS_DUAL)})"
        rf"|(?P<bare>{L.alt(_AR(L.HOUR_UNITS))}))(?:\s*و\s*(?P<frac>نص|ربع))?{E}"
    ),
    re.compile(rf"{B}in\s+{_AMT}\s*(?P<unit>{_UNITS_MIN}|{_UNITS_HOUR}){E}"),
]

_MON = L.alt(L.MONTHS)
_YOM = rf"(?:{CONJ}(?:يوم|في)\s+)?"
_ORD = r"(?:\s*(?:st|nd|rd|th))?"
RX_DATE_ISO = re.compile(rf"{B}(?P<y>\d{{4}})-(?P<m>\d{{1,2}})-(?P<d>\d{{1,2}}){E}")
RX_DATE_DMY = re.compile(
    rf"{B}{_YOM}(?P<d>\d{{1,2}})\s*[/\-]\s*(?P<m>\d{{1,2}})(?:\s*[/\-]\s*(?P<y>\d{{4}}|\d{{2}}))?{E}"
)
RX_DATE_D_MON = re.compile(
    rf"{B}{_YOM}(?:ال)?(?P<d>\d{{1,2}}){_ORD}\s*(?:of\s+)?(?:شهر\s+)?(?P<mon>{_MON})(?:\s*(?P<y>\d{{4}}))?{E}"
)
RX_DATE_MON_D = re.compile(
    rf"{B}(?:on\s+)?(?P<mon>{L.alt(_EN(L.MONTHS))})\s+(?:the\s+)?(?P<d>\d{{1,2}}){_ORD}(?:\s*,?\s*(?P<y>\d{{4}}))?{E}"
)
RX_DATE_DOM_ONLY = [
    re.compile(rf"{B}{CONJ}يوم\s+(?P<d>\d{{1,2}}){E}(?!\s*[:.]\d)"),
    re.compile(rf"{B}(?:on\s+)?the\s+(?P<d>\d{{1,2}})\s*(?:st|nd|rd|th){E}"),
]
RX_DAY_WORD = re.compile(rf"{B}{CONJ}(?P<w>{L.alt(L.DAY_WORDS)}){E}")
RX_TONIGHT = re.compile(rf"{B}{CONJ}(?P<w>{L.alt(L.TONIGHT_WORDS)}){E}")
RX_WEEKDAY_AR = re.compile(
    rf"{B}{CONJ}(?:يوم\s+)?(?P<wd>{L.alt(_AR(L.WEEKDAYS))})"
    rf"(?:\s+(?P<mod>{L.alt(L.WEEKDAY_NEXT + L.WEEKDAY_THIS)}))?{E}"
)
RX_WEEKDAY_EN = re.compile(
    rf"{B}(?:on\s+)?(?:(?P<mod>{L.alt(L.WEEKDAY_NEXT_EN + L.WEEKDAY_THIS_EN)})\s+)?(?P<wd>{L.alt(_EN(L.WEEKDAYS))}){E}"
)
RX_VAGUE_DATE = re.compile(rf"{B}{CONJ}{L.alt(L.VAGUE_DATE)}{E}")

_SAA = rf"(?:{CONJ}(?:ال)?ساعه)"
RX_NOON_MIDNIGHT = re.compile(
    rf"{B}(?:{_SAA}\s+|at\s+)?(?:(?P<mid>{L.alt(L.MIDNIGHT_WORDS)})|(?P<noon>{L.alt(L.NOON_WORDS)})){E}"
)
RX_HOUR_DIGIT = re.compile(r"(?<![\w/:.\-])(?P<h>\d{1,2})(?![\d/])")
RX_HOUR_WORD = re.compile(rf"{B}(?P<h>{L.alt(L.HOUR_WORDS)}){E}")
RX_CLOCK_PREFIX = re.compile(rf"(?:{B}{_SAA}|{B}at|@)\s*$")
_MINW = rf"(?P<n>{L.alt(L.MINUTE_WORDS)}|\d{{1,2}})"
_MU = rf"(?:\s*{L.alt(L.MINUTE_UNITS)})?"
RX_MIN_COLON = re.compile(r"[:.](?P<n>\d{2})(?!\d)")
RX_MIN_WORDS = [  # (pattern, sign, base) -> minute offset = base + sign * n
    (re.compile(rf"\s*و\s*(?:نص|نصف)\s*الا\s*{_MINW}{_MU}{E}"), -1, 30),
    (re.compile(rf"\s*و\s*(?:نص|نصف)\s*و\s*{_MINW}{_MU}{E}"), +1, 30),
    (re.compile(rf"\s*الا\s*{_MINW}{_MU}{E}"), -1, 0),
    (re.compile(rf"\s*و\s*(?P<n>{L.alt(L.MINUTE_WORDS)}){_MU}{E}"), +1, 0),
    (re.compile(rf"\s*و\s*(?P<n>\d{{1,2}})\s*{L.alt(L.MINUTE_UNITS)}{E}"), +1, 0),
]
RX_MARKER = re.compile(rf"\s*(?P<mk>{L.alt(L.AM_MARKERS + L.PM_MARKERS)})(?!\w)")
RX_PERIOD = re.compile(rf"{B}{CONJ}(?P<p>{L.alt(L.PERIODS)}){E}")
RX_PERIOD_AHEAD = re.compile(rf"\s+{CONJ}{L.alt(L.PERIODS)}{E}")
RX_VAGUE_TIME = re.compile(rf"{B}{CONJ}{L.alt(L.VAGUE_TIME)}{E}")


class _Claims:
    def __init__(self) -> None:
        self.spans: list[tuple[int, int]] = []

    def free(self, s: int, e: int) -> bool:
        return all(e <= a or s >= b for a, b in self.spans)

    def take(self, s: int, e: int) -> None:
        self.spans.append((s, e))


# --------------------------------------------------------------------------- detection
def _amount(tok: str) -> float:
    if tok.isdigit():
        return float(tok)
    if tok in _FRACTIONS:
        return _FRACTIONS[tok]
    return float(L.AMOUNT_WORDS[tok])  # type: ignore[arg-type]


def _explicit_date(d: int, m: int, y: int | None, ref: date) -> DateRef:
    try:
        if y is not None:
            return DateRef(frozenset({date(y + 2000 if y < 100 else y, m, d)}), "explicit")
        cand = date(ref.year, m, d)
        if cand < ref:
            cand = date(ref.year + 1, m, d)
        return DateRef(frozenset({cand}), "explicit, next occurrence")
    except ValueError:
        return DateRef(None, "invalid calendar date")


def _weekday(wd: int, mod: str | None, ref: date) -> DateRef:
    delta = (wd - ref.weekday()) % 7
    if mod in ("next",):
        return DateRef(frozenset({ref + timedelta(days=delta or 7)}), "weekday, next")
    if mod in ("this",):
        return DateRef(frozenset({ref + timedelta(days=delta)}), "weekday, this")
    if delta == 0:  # bare weekday equal to today: today or in a week?
        return DateRef(frozenset({ref, ref + timedelta(days=7)}), "weekday equals today")
    return DateRef(frozenset({ref + timedelta(days=delta)}), "weekday")


def _mod_kind(mod: str | None) -> str | None:
    if mod is None:
        return None
    if mod in L.WEEKDAY_NEXT or mod in L.WEEKDAY_NEXT_EN:
        return "next"
    return "this"


def detect(text: str, ref: datetime) -> list[Mention]:
    """Find temporal mentions in normalized ``text`` relative to aware datetime ``ref``."""
    claims = _Claims()
    out: list[Mention] = []
    today = ref.date()

    def add(kind: str, s: int, e: int, value: object) -> None:
        claims.take(s, e)
        out.append(Mention(kind, s, e, value))

    # 1. relative offsets
    for rx in RX_RELATIVE:
        for m in rx.finditer(text):
            if not claims.free(*m.span()):
                continue
            g = m.groupdict()
            if g.get("dual"):
                minutes = 2.0 if g["dual"] in L.MINUTE_UNITS_DUAL else 120.0
            elif g.get("bare"):
                minutes = 60.0
            else:
                amt = _amount(g["amt"])
                minutes = amt * (60 if g["unit"] in L.HOUR_UNITS else 1)
            if g.get("frac"):
                minutes += 60 * _FRACTIONS[g["frac"]]
            add("relative", *m.span(), timedelta(minutes=round(minutes)))

    # 2. explicit dates (most specific first), then day-of-month-only (ambiguous)
    for rx in (RX_DATE_ISO, RX_DATE_D_MON, RX_DATE_MON_D, RX_DATE_DMY):
        for m in rx.finditer(text):
            if not claims.free(*m.span()):
                continue
            g = m.groupdict()
            month = int(L.MONTHS[g["mon"]]) if g.get("mon") else int(g["m"])  # type: ignore[arg-type]
            year = int(g["y"]) if g.get("y") else None
            add("date", *m.span(), _explicit_date(int(g["d"]), month, year, today))
    for rx in RX_DATE_DOM_ONLY:
        for m in rx.finditer(text):
            if claims.free(*m.span()):
                add("date", *m.span(), DateRef(None, "day of month without month"))

    # 3. day words and "tonight"
    for m in RX_DAY_WORD.finditer(text):
        if claims.free(*m.span()):
            off = int(L.DAY_WORDS[m["w"]])
            add("date", *m.span(), DateRef(frozenset({today + timedelta(days=off)}), "day word"))
    for m in RX_TONIGHT.finditer(text):
        if claims.free(*m.span()):
            add("date", *m.span(), DateRef(frozenset({today}), "tonight"))
            out.append(Mention("period", m.start(), m.end(), "night"))

    # 4. weekdays
    for rx in (RX_WEEKDAY_AR, RX_WEEKDAY_EN):
        for m in rx.finditer(text):
            if claims.free(*m.span()):
                wd = int(L.WEEKDAYS[m["wd"]])
                add("date", *m.span(), _weekday(wd, _mod_kind(m["mod"]), today))

    # 5. vague dates
    for m in RX_VAGUE_DATE.finditer(text):
        if claims.free(*m.span()):
            add("date", *m.span(), DateRef(None, "vague date"))

    # 6. noon / midnight
    for m in RX_NOON_MIDNIGHT.finditer(text):
        if claims.free(*m.span()):
            clock = Clock(0, 0, True, early=True) if m["mid"] else Clock(12, 0, True)
            add("clock", *m.span(), clock)

    # 7. clock times
    cands = [(m, False) for m in RX_HOUR_DIGIT.finditer(text)]
    cands += [(m, True) for m in RX_HOUR_WORD.finditer(text)]
    for m, is_word in sorted(cands, key=lambda c: c[0].start()):
        hs, he = m.span()
        if not claims.free(hs, he):
            continue
        raw = m["h"]
        hour = int(L.HOUR_WORDS[raw]) if is_word else int(raw)  # type: ignore[arg-type]
        prefix = RX_CLOCK_PREFIX.search(text, 0, hs)
        start = prefix.start() if prefix else hs
        pos, minute, has_colon, has_wordmin = he, 0, False, False
        if not is_word and (mc := RX_MIN_COLON.match(text, pos)):
            minute, pos, has_colon = int(mc["n"]), mc.end(), True
        else:
            for rx, sign, base in RX_MIN_WORDS:
                if mw := rx.match(text, pos):
                    n = mw["n"]
                    val = int(n) if n.isdigit() else int(L.MINUTE_WORDS[n])  # type: ignore[arg-type]
                    minute, pos, has_wordmin = base + sign * val, mw.end(), True
                    break
        marker = None
        if mk := RX_MARKER.match(text, pos):
            marker = "am" if mk["mk"] in L.AM_MARKERS else "pm"
            pos = mk.end()
        period_ahead = bool(RX_PERIOD_AHEAD.match(text, pos))

        if is_word:
            accept = bool(prefix) or marker is not None or period_ahead
        else:
            accept = bool(prefix) or has_colon or has_wordmin or marker is not None or period_ahead
        if not accept or not claims.free(start, pos):
            continue
        if has_colon and not 0 <= minute <= 59:
            continue
        if marker and not 1 <= hour <= 12:
            continue
        if not 0 <= hour <= 23:
            continue
        is_24h = (not is_word) and (hour == 0 or hour >= 13 or (has_colon and raw.startswith("0")))
        add("clock", start, pos, Clock(hour, minute, is_24h, marker))

    # 8. periods, 9. vague times
    for m in RX_PERIOD.finditer(text):
        if claims.free(*m.span()):
            add("period", *m.span(), L.PERIODS[m["p"]])
    for m in RX_VAGUE_TIME.finditer(text):
        if claims.free(*m.span()):
            add("vague_time", *m.span(), m.group(0))

    return sorted(out, key=lambda x: (x.start, x.kind))


# --------------------------------------------------------------------------- resolution
def to_24h(hour: int, period: str) -> tuple[int, bool] | None:
    """Convert a 12 h clock hour under a period to (hour 0-23, early_night) or None if inconsistent.

    ``early_night`` marks 12-5 "at night", which belongs to the *next* calendar day; combined
    with an explicit date it is ambiguous by contract.
    """
    h = hour
    if period == "am":
        return (0 if h == 12 else h), False
    if period == "pm":
        return (12 if h == 12 else h + 12), False
    if period == "morning":
        return (h, False) if 1 <= h <= 11 else None
    if period == "dawn":
        return (h, False) if 3 <= h <= 6 else None
    if period == "noon":
        return {11: (11, False), 12: (12, False), 1: (13, False), 2: (14, False), 3: (15, False)}.get(h)
    if period == "afternoon":
        return (12, False) if h == 12 else ((h + 12, False) if 1 <= h <= 6 else None)
    if period == "maghrib":
        return (h + 12, False) if 5 <= h <= 8 else None
    if period == "night":
        if 6 <= h <= 11:
            return h + 12, False
        return (0 if h == 12 else h), True
    raise ValueError(f"unknown period {period!r}")


def resolve(mentions: list[Mention], ref: datetime, tz_name: str = "Africa/Cairo") -> Temporal:
    if ref.tzinfo is None:
        raise ValueError("reference datetime must be timezone-aware")
    # Work in the IANA zone, not ref's fixed offset, so dates across a DST change get the
    # right UTC offset (Cairo moves from +03:00 to +02:00 after the last Thursday of October).
    tz = ZoneInfo(tz_name)
    ref = ref.astimezone(tz)
    gaps: set[str] = set()
    rel = [m.value for m in mentions if m.kind == "relative"]
    dates = [m.value for m in mentions if m.kind == "date"]
    clocks = list(dict.fromkeys(m.value for m in mentions if m.kind == "clock"))
    periods = {m.value for m in mentions if m.kind == "period"}

    def done(dt: datetime | None) -> Temporal:
        return Temporal(dt, frozenset(gaps), tuple(mentions))

    # relative offsets stand alone
    if rel:
        if len(rel) > 1 or dates or clocks:
            gaps.add(TIME)
            return done(None)
        dt = (ref.astimezone(UTC) + rel[0]).astimezone(tz)
        return done(dt.replace(second=0, microsecond=0))

    # date: intersect candidate sets; ambiguity or conflict -> gap
    the_date: date | None = None
    if dates:
        cands: frozenset[date] | None = None
        ambiguous = False
        for d in dates:
            assert isinstance(d, DateRef)
            if d.candidates is None:
                ambiguous = True
            else:
                cands = d.candidates if cands is None else cands & d.candidates
        if ambiguous or cands is None or len(cands) != 1:
            gaps.add(DATE)
        else:
            the_date = next(iter(cands))

    # clock
    hhmm: tuple[int, int] | None = None
    early = False
    if len(clocks) != 1:
        gaps.add(TIME)
    else:
        c = clocks[0]
        assert isinstance(c, Clock)
        if c.is_24h:
            hour24, early = c.hour, c.early
        else:
            ps = periods | ({c.marker} if c.marker else set())
            if not ps:
                gaps.add(AM_PM)
                hour24 = None
            else:
                conv = {to_24h(c.hour, p) for p in ps}
                if None in conv:
                    gaps.add(TIME)
                    hour24 = None
                elif len(conv) > 1:
                    gaps.add(AM_PM)
                    hour24 = None
                else:
                    hour24, early = next(iter(conv))  # type: ignore[misc]
        if hour24 is not None:
            total = hour24 * 60 + c.minute
            hhmm = ((total // 60) % 24, total % 60)
            day_shift = total // (24 * 60)
            if early and dates:
                gaps.add(DATE)  # "بكره الساعة 2 بالليل": which night?
            elif the_date is not None:
                the_date = the_date + timedelta(days=day_shift)

    if gaps:
        return done(None)
    assert hhmm is not None
    if the_date is not None:
        dt = datetime.combine(the_date, time(*hhmm), tzinfo=tz)
    else:  # next-occurrence convention: today if still ahead (incl. a repeated DST hour), else tomorrow
        tomorrow = ref.date() + timedelta(days=1)
        options = [datetime.combine(ref.date(), time(*hhmm, fold=f), tzinfo=tz) for f in (0, 1)]
        dt = next((o for o in options if is_after(o, ref)),
                  datetime.combine(tomorrow, time(*hhmm), tzinfo=tz))
    if not is_after(dt, ref):
        gaps.add(PAST)
        return done(None)
    return done(dt)


def is_after(a: datetime, b: datetime) -> bool:
    """Instant comparison. Plain ``a > b`` with a shared ZoneInfo compares wall clocks and
    ignores ``fold``, which is wrong inside the repeated hour after a DST change."""
    return a.astimezone(UTC) > b.astimezone(UTC)


def parse_temporal(normalized_text: str, ref: datetime, tz_name: str = "Africa/Cairo") -> Temporal:
    ref = ref.astimezone(ZoneInfo(tz_name))
    return resolve(detect(normalized_text, ref), ref, tz_name)


def cairo(iso: str, tz: str = "Africa/Cairo") -> datetime:
    """Parse an ISO datetime; naive values are interpreted in ``tz``."""
    dt = datetime.fromisoformat(iso)
    return dt if dt.tzinfo else dt.replace(tzinfo=ZoneInfo(tz))
