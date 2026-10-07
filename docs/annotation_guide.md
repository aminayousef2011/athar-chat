# ATHAR Chat annotation guide (v0.1, follows task contract v0.1)

This guide is for the people who write and label the dataset: the author and the second
annotator. It turns the [task contract](task_contract.md) into a procedure. When the two
disagree, the contract wins and this guide gets fixed. Annotation starts only after the
contract is frozen at v1.0.

## 1. What you produce

One JSON object per line in `data/athar.jsonl`, 300 lines in total. Each line is a single-turn
request in Egyptian Arabic (code-switching allowed), together with the label a careful
assistant should produce:

| Route | When |
|---|---|
| `ACT` | supported request (one reminder or one alarm for the speaker), every required argument resolved |
| `CLARIFY` | supported request, at least one argument missing or ambiguous |
| `ABSTAIN` | anything outside scope, even when its arguments are complete |

**Do not run any of the systems while you write or label.** Your labels are the reference the
systems are scored against. If you look at what the system says, your labels drift toward it.

## 2. Writing items

### 2.1 Families and pairs
A **family** (`family_id`) is one scenario: one task told in different ways. All items of a
family go into the same split, so paraphrases never leak from training into test.

- Put an item in an existing family when it is a paraphrase or a variant of the same request
  (same task, same intent). Start a new family for a new task.
- A **pair** (`pair_id`) is two items of one family that differ only in what makes one complete
  and the other incomplete: `فكرني بكرة الساعة 7 الصبح أشتري عيش` (ACT) and
  `فكرني بكرة الساعة 7 أشتري عيش` (CLARIFY `am_pm`). Exactly two items, the same intent, one ACT
  and one CLARIFY. The paired evaluation compares how a system handles both.
- Keep families small: 1 to 4 items. The splitter moves whole families. Big families make the
  test targets hard to hit.

### 2.2 Coverage targets for the 300 items
The locked test split needs **50 ACT, 50 CLARIFY, 20 ABSTAIN** from families written by people
(`source` `author` or `contributor`). Aim for about **125 ACT, 125 CLARIFY, 50 ABSTAIN** overall
so the splitter has room. Within that:

- both tools: reminders and alarms (alarms need no reminder text);
- every clarification target at least 10 times: `date`, `time`, `am_pm`, `past`, `reminder_text`,
  including some items with two targets;
- every unsupported reason at least 5 times: `deletion`, `modification`, `recurring`,
  `other_person`, `multi_action`, `other_tool`;
- code-switched items (Arabic and English words in one request) in all three routes,
  roughly a quarter of the data. Pure Arabizi (`bokra el sob7`) is out of scope;
- varied `ref_datetime` values: mornings, evenings, late night, every weekday, and some
  items whose resolved time falls after Cairo's DST change (section 4.2);
- natural variety: politeness (`لو سمحت`), hedges (`حوالي`), word order, typos people actually
  make, Arabic-Indic digits (`٧`) and Western digits (`7`).

### 2.3 Style rules
- Write the way people actually type to an assistant: short, informal, Egyptian. Avoid MSA
  unless people really say it that way (`صباحاً` is fine).
- Use invented names only (`أحمد`, `منى`). Never real phone numbers, addresses or private details.
- Do not copy the pilot items in `data/pilot/`. They were drafted by an assistant to test the
  tooling and are never part of the dataset.

## 3. Labelling procedure

Label each item in this order. Each step uses only the request text and `ref_datetime`.

**Step 1: intent.** Is this one reminder or one alarm, for the speaker, happening once?
- yes: `create_reminder` or `create_alarm` (wake-up or ring requests are alarms: `صحيني`,
  `ظبطلي منبه`). Go to step 2.
- no: `intent: unsupported`, `route: ABSTAIN`, set `unsupported_reason`, and stop. If several
  reasons apply, take the first in this order: `deletion`, `modification`, `recurring`,
  `other_person`, `multi_action`, `other_tool` **[DECIDE]**. `other` is for out-of-scope
  requests that fit none of these; explain it in `notes`.

**Step 2: arguments.** Resolve the date and time with the contract rules (section 4). Resolve the
reminder text for reminders (section 5). Never guess. If the contract does not resolve a value,
it is a gap.

**Step 3: gaps.** List every gap that applies, from the closed set below. Do not stop at the
first one.

| Target | Typical trigger |
|---|---|
| `date` | ambiguous day: bare weekday equal to today's weekday, `يوم 15`, `الأسبوع الجاي`, `الويكند`, a date that conflicts with its weekday, `31/2`, a date with `الساعة 2 بالليل` |
| `time` | no clock time (`فكرني بكرة`), vague time (`بدري`, `بعدين`, `الصبح` alone), two different times, an hour outside its period (`8 الضهر`) |
| `am_pm` | an hour from 1 to 12 with no period or marker (`الساعة 7`), or two periods that disagree |
| `past` | everything resolves but the instant is not after `ref_datetime` |
| `reminder_text` | a reminder with no content, or vague content (`فكرني بحاجة`); never for alarms |

A missing date is **not** a gap: `الساعة 9 بالليل` alone means the next 21:00 (today if still
ahead, otherwise tomorrow). `past` applies only once everything else has resolved.

**Step 4: route.** No gaps: `ACT`. One or more gaps: `CLARIFY` with `clarify_targets` listing all
of them. The validator rejects inconsistent combinations (for example ACT with targets, or
`reminder_text` on an alarm).

## 4. Fields

| Field | Required | How to fill it |
|---|---|---|
| `id` | yes | unique, lower case, `[a-z0-9_-]`, e.g. `r0001`. Never reuse an id. |
| `family_id` | yes | scenario id, e.g. `f0001` (section 2.1) |
| `pair_id` | pairs only | shared by exactly the two items of a pair |
| `text` | yes | the request exactly as a user would type it |
| `ref_datetime` | yes | when the user sends it: ISO-8601 **with the offset in force at that instant**, e.g. `2026-10-07T10:00:00+03:00` |
| `timezone` | no | IANA zone, default `Africa/Cairo` |
| `intent` | yes | `create_reminder`, `create_alarm`, `unsupported` |
| `route` | yes | `ACT`, `CLARIFY`, `ABSTAIN` |
| `clarify_targets` | CLARIFY | every gap, any order (stored in the fixed contract order) |
| `unsupported_reason` | ABSTAIN | section 3, step 1 |
| `args.datetime` | ACT | the resolved instant, ISO-8601 with offset; fill it on CLARIFY items too when only `reminder_text` is missing |
| `args.reminder_text` | ACT reminders | section 5; fill it on CLARIFY reminders too when it is resolved |
| `args.reminder_text_alts` | optional | other readings that are equally correct |
| `code_switched` | yes if true | `true` when the text mixes Arabic and Latin-script words |
| `source` | yes | `author` (you), `contributor` (another native speaker), `assistant_draft` (machine-drafted, never in test) |
| `notes` | optional | anything a reviewer should know, e.g. the rule you applied |

### 4.1 Writing datetimes
Write the local wall-clock time with the zone's offset, to the minute:
`2026-10-08T07:00:00+03:00`. Do not convert to UTC.

### 4.2 Daylight saving time in Cairo
Egypt observes DST from the last Friday of April to the last Thursday of October. In 2026 the
offset is `+03:00` up to 29 October and `+02:00` from 30 October. A reminder set on 27 October
for Friday 30 October at 8 in the morning is `2026-10-30T08:00:00+02:00`. The validator rejects
an offset that does not match the zone at that instant, so a wrong offset is caught, not
silently read as a different time.

## 5. Reminder text

The reminder text is the request minus:
1. the request framing (`فكرني`, `ممكن تفكرني`, `اعملي تذكير`, `remind me`, ...);
2. politeness and fillers (`لو سمحت`, `please`, `معلش`);
3. every date and time expression (`بكرة`, `الساعة 7 الصبح`, `بعد ساعة`);
4. function words left at either edge (`إن/إني`, `عشان`, `في`, `to`, ...);
5. the `ب` of a leading `بال...` (`بالاجتماع` → `الاجتماع`).

Keep everything inside, in the user's own spelling, including English words:

| Request | `reminder_text` |
|---|---|
| `فكرني بكرة الساعة ٧ الصبح أشتري عيش` | `أشتري عيش` |
| `فكرني إني أكلم ماما بكرة الساعة 5 العصر` | `أكلم ماما` |
| `ممكن تفكرني أكلم أحمد بكرة الساعة 5 العصر عشان عيد ميلاده` | `أكلم أحمد عشان عيد ميلاده` |
| `فكرني بكرة الساعة 3 العصر عندي meeting مع الـ team` | `عندي meeting مع الـ team` |

Scoring compares normalized forms (spelling variants such as `ة/ه` and `أ/ا` do not matter).
When two readings are equally right, put the second in `reminder_text_alts`.

## 6. Code-switching
Mark `code_switched: true` when the request mixes Arabic script with Latin-script words
(`ظبطلي alarm الساعة 5 الفجر`). The validator warns when the flag and the text disagree.

## 7. Quality control

1. **Validate as you go.** `uv run athar-data validate data/athar.jsonl` lists errors (contract
   violations) and warnings (likely slips). Fix both. Before splitting, it must pass with
   `--strict`.
2. **Agreement on 60 items.**
   `uv run athar-data agreement-sheet data/athar.jsonl --out runs/agreement/sheet.csv` draws a
   route-stratified sample (one item per family where possible) and writes a blind CSV. It
   opens in Excel or Google Sheets and holds the request, the reference time in words, and
   empty label columns. The second annotator fills `intent`, `route`, `clarify_targets`
   (space separated), `unsupported_reason`, `datetime` (`2026-10-08 07:00`) and
   `reminder_text`, using only this guide. Then:
   `uv run athar-data kappa data/athar.jsonl runs/agreement/sheet.csv --annotator "<name>" --out docs/agreement`
   reports agreement and Cohen's κ per field.
   **No second annotator:** do a dated self-review. Label the blank sheet yourself at least
   seven days after the original labelling, without looking at the dataset, and pass
   `--annotator self-review`.
3. **Adjudicate.** Discuss every disagreement the report lists. Correct the gold where it was
   wrong, add a sentence to `notes`, and fix this guide or the contract if a rule was unclear.
   The reported κ is the pre-adjudication value.
4. **Split once.** `uv run athar-data split data/athar.jsonl --config data/split_config.yaml --out data/splits/manifest.json`,
   then `uv run athar-data verify data/splits/manifest.json`. Commit the dataset, config and
   manifest together. From then on, any edit to an item fails verification. The test split
   stays locked until the Phase 4 evaluation run.
5. **Check train and dev only.** `uv run athar-data crosscheck --manifest data/splits/manifest.json --split dev`
   compares gold with the shared core given the gold intent. It refuses the test split.

## 8. Worked examples

All of these are pilot items (`data/pilot/pilot.jsonl`), with `ref_datetime`
Wednesday 7 October 2026, 10:00 Cairo unless stated.

| Request | Label | Why |
|---|---|---|
| `صحيني بكرة الساعة 6 ونص الصبح` | ACT alarm, `2026-10-08T06:30:00+03:00` | `ونص` = 30 minutes; alarms have no text |
| `ظبطلي alarm الساعة 5 الفجر` (ref 22:30) | ACT alarm, `2026-10-08T05:00:00+03:00` | no date: 05:00 today has passed, so tomorrow |
| `ظبطلي alarm الساعة 5` (ref 22:30) | CLARIFY `am_pm` | hour 1-12 without period |
| `فكرني يوم الأربع الساعة 10 الصبح أروح للدكتور` | CLARIFY `date` | bare weekday that is today: today or in 7 days? |
| `فكرني الأربع الجاي الساعة 10 الصبح أروح للدكتور` | ACT, `2026-10-14T10:00:00+03:00` | `الجاي` on today's weekday = 7 days ahead |
| `فكرني بحاجة بعد ساعة` | CLARIFY `reminder_text`, `args.datetime` 11:00 | time resolves, content is vague |
| `فكرني النهارده الساعة 3 العصر أدفع فاتورة النت` (ref 18:00) | CLARIFY `past` | 15:00 today is before the reference |
| `فكرني يوم الخميس 30 أكتوبر الساعة 8 الصبح أجدد الباسبور` (ref 27 Oct) | CLARIFY `date` | 30 October 2026 is a Friday |
| `فكرني كل يوم الساعة 8 بالليل آخد الدوا` | ABSTAIN `recurring` | complete arguments, still out of scope |
| `فكّر أحمد يجيب الشاحن بكرة` | ABSTAIN `other_person` | the reminder is for someone else |

A full line (ACT reminder):

```json
{"id": "r0001", "family_id": "f0001", "pair_id": "p0001", "text": "فكرني بكرة الساعة ٧ الصبح أشتري عيش", "ref_datetime": "2026-10-07T10:00:00+03:00", "intent": "create_reminder", "route": "ACT", "args": {"datetime": "2026-10-08T07:00:00+03:00", "reminder_text": "أشتري عيش"}, "source": "author"}
```

## 9. Open points

The **[DECIDE]** items in the task contract (weekday rules, `بالليل` after midnight, relative
offsets, template wording) and the reason priority in section 3 must be confirmed when the
contract is frozen at v1.0. Changing them later means relabelling.
