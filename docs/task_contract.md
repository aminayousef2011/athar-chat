# ATHAR Chat task contract (v1.0, frozen 2026-10-07)

This contract is the single source of truth for annotation, the shared extractor and the
policy. Every rule below has at least one executable case in `tests/`. It was frozen at
v1.0 on 2026-10-07, before dataset annotation. Changes after this date need a version bump,
a dated entry in section 8 and relabelling of affected items; once models are frozen, its hash
is checked by the locked test run.

## 1. Scope

| In scope | Out of scope (-> `ABSTAIN`) | `unsupported_reason` |
|---|---|---|
| one reminder for the user | deleting / cancelling | `deletion` |
| one alarm for the user | editing an existing item | `modification` |
| | recurring schedules (`كل يوم`, `every Monday`) | `recurring` |
| | items for another person (`فكّر أحمد ...`) | `other_person` |
| | more than one action in one request | `multi_action` |
| | other tools (weather, calls, timers ...) | `other_tool` |

Single turn only. Arabic script is the core scope; Arabic-English code-switching is included;
broad Arabizi (`bokra el sob7`) is excluded.

Each item carries `ref_datetime` (ISO-8601 with offset) and `timezone` (default
`Africa/Cairo`). All resolution happens in the IANA zone, so Cairo's DST change
(+03:00 -> +02:00 after the last Thursday of October) is applied correctly.

## 2. Tools and required arguments

| Tool | Required | Not used |
|---|---|---|
| `create_reminder` | `datetime` (unique, future), `reminder_text` | |
| `create_alarm` | `datetime` (unique, future) | `reminder_text` |

Routes: **ACT** = supported intent, no blocking gap. **CLARIFY** = supported intent with at
least one blocking gap. **ABSTAIN** = unsupported intent (even when arguments are complete).

## 3. Clarification targets (closed set, fixed order)

| Target | When |
|---|---|
| `date` | a date was mentioned but is ambiguous, conflicting or invalid |
| `time` | no usable clock time: missing, vague (`بدري`, `بعدين`, `الصبح` alone), several, or inconsistent with its period (`8 الضهر`) |
| `am_pm` | a 1-12 clock time with no period/marker, or with periods that disagree |
| `past` | everything resolved, but the instant is not after `ref_datetime` |
| `reminder_text` | reminder with no content, or vague content (`حاجة`, `something`); never for alarms |

A CLARIFY item succeeds only when the predicted set equals the gold set exactly.
Note: a **missing date is never a gap** (next-occurrence rule, 4.1).

## 4. Time-expression conventions

### 4.1 Dates
- `النهارده/اليوم/today` = ref date; `بكرة/tomorrow` = +1; `بعد بكرة` = +2; `الليلة/tonight` = ref date + night period.
- **Bare weekday** = nearest occurrence 1-6 days ahead. If it equals today's weekday it is
  ambiguous (today or in 7 days) -> `date`. *(decided v1.0)*
- **Weekday + `الجاي`/`next`** = nearest occurrence 1-7 days ahead (7 when it is today's weekday).
  Some speakers read `الخميس الجاي` as "Thursday of next week"; the contract does not. *(decided v1.0)*
- **Weekday + `ده`/`this`** = occurrence 0-6 days ahead (0 = today).
- Explicit dates are **day/month** (`15/10`, `15-10-2026`, `15 أكتوبر`, `october 15`, ISO
  `2026-10-15`). Without a year: this year if not already past, else next year.
- Ambiguous by contract -> `date`: day-of-month only (`يوم 15`, `the 15th`), vague spans
  (`الأسبوع الجاي`, `next week`, `الويكند`, `آخر الشهر`), invalid dates (`31/2`), and two mentions
  that resolve to different days (`بكرة الجمعة` when tomorrow is Thursday).
- **No date mentioned** + resolved time -> next occurrence: today if still ahead (including the
  repeated DST hour), otherwise tomorrow.

### 4.2 Clock times
- A number counts as a time only with evidence: `الساعة`/`at` before it, `:mm`, a minute word
  (`ونص`), a marker (`ص/م/am/pm`), or a following period word. `اشتري 2 لتر` is not a time.
  Hour **words** (`سبعة`) need `الساعة`, a marker or a period word.
- Minutes: `ونص` 30, `وربع` 15, `وتلت` 20, `إلا ربع` -15, `إلا تلت` -20, `ونص إلا خمسة` 25,
  `ونص وخمسة` 35, `و10 دقايق` 10, `:mm`.
- 24-hour forms need no period: hour 0 or 13-23, or a leading zero (`07:00`).
- `noon` = 12:00. `midnight`/`نص الليل` = 00:00 of the night's end (ambiguous with a date).
- Relative offsets resolve exactly: `بعد ساعة/ساعتين/ساعة ونص/نص ساعة/ربع ساعة/10 دقايق`,
  `in 20 minutes`. *(decided v1.0: in scope; common in requests)*

### 4.3 Periods -> 24 h (for hour h said on a 12 h clock)

| Period words | Valid h | Result |
|---|---|---|
| `ص`, `am` | 1-12 | h (12 -> 00) |
| `م`, `pm`, `مساءً`, `evening` | 1-12 | h+12 (12 -> 12) |
| `الصبح`, `صباحاً`, `morning` | 1-11 | h |
| `الفجر` | 3-6 | h |
| `الضهر`, `الظهر` | 11, 12, 1-3 | 11, 12, 13-15 |
| `بعد الضهر`, `العصر`, `afternoon` | 12, 1-6 | 12, 13-18 |
| `المغرب` | 5-8 | 17-20 |
| `بالليل`, `الليل`, `العشا`, `at night` | 6-11 | 18-23 |
| `بالليل` | 12, 1-5 | 00-05 of the *next* calendar day: fine without a date, `date` gap with one (`بكرة الساعة 2 بالليل`) *(decided v1.0)* |

An h outside the valid range -> `time`; two periods giving different hours -> `am_pm`.

## 5. Reminder text

Reminder text = the request minus (a) request framing (`فكرني`, `ممكن تفكرني`, `remind me`,
`اعملي تذكير`, `set a reminder` ...), (b) politeness/fillers (`لو سمحت`, `please`), (c) every
temporal expression, (d) function words left at either edge (`إن/إني`, `عشان`, `في`, `to` ...),
and (e) the `ب` of a leading `بال...`. Internal words are kept (`أكلم أحمد عشان عيد ميلاده`).

Gold `reminder_text` is written in the user's spelling; comparison uses `canonical_text()`
(normalize + same stripping) on both sides. Annotators may add `reminder_text_alts` when two
readings are equally correct.

## 6. Response templates (gender-neutral, decided v1.0)

| Case | Template |
|---|---|
| ACT reminder | `تمام، هفكرك بـ«{text}» يوم {weekday} {d} {month} الساعة {h:mm} {period}.` |
| ACT alarm | `تمام، المنبه هيرن يوم ... الساعة ...` |
| `date` | `أنهي يوم بالظبط؟` |
| `time` | `أفكرك الساعة كام؟` / `المنبه يرن الساعة كام؟` |
| `am_pm` | `الساعة {h} {day} ولا بالليل؟`, `{day}` = `الصبح` (6-11), `الضهر` (12, 1-3), `العصر` (4-5), as in the period table (no single hour: `الصبح ولا بالليل؟`) |
| `past` | `الوقت ده عدّى خلاص، أخليه إمتى؟` |
| `reminder_text` | `أفكرك بإيه؟` |
| ABSTAIN | `معلش، أنا أقدر أعمل تذكير أو منبه لمرة واحدة بس، ومقدرش أنفذ الطلب ده.` |

Replies avoid gendered second-person forms (`تقصد/تقصدي`, `عايز/عايزة`, `تحب/تحبي`,
`ليك/ليكي`): questions are phrased from the assistant's side (`أفكرك ...؟`, `أخليه إمتى؟`).
`هفكرك` / `أفكرك` are written identically for both genders. The wording still needs a
native-speaker read before user-facing use; it does not affect any score.

When several out-of-scope reasons apply, `unsupported_reason` is the **first in the order
of the section 1 table**: `deletion`, `modification`, `recurring`, `other_person`,
`multi_action`, `other_tool` *(decided v1.0)*; `other` only when none fits.

## 7. Known limitations (documented, not fixed)
- `2 ونص كيلو` would be read as 2:30 (number + minute word with no other evidence).
- `بعد الساعة 5` ("after 5") is treated as `الساعة 5`.
- Recurrence, other-person and multi-action detection is the **intent recognizer's** job, by
  design; the extractor does not block them.

## 8. Version history

| Version | Date | Change |
|---|---|---|
| 0.1 | 2026-10-07 | Draft with [DECIDE] items |
| 1.0 | 2026-10-07 | Frozen. Confirmed: same-weekday -> `date`; `X الجاي` = nearest 1-7 days; `بالليل` 12-5 = next calendar day, `date` gap with an explicit date; relative offsets in scope; reason priority = table order. Changed: gender-neutral templates. |
