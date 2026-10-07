# ATHAR Chat: draft dataset (for product review)

**Status: assistant-drafted, not the evaluation dataset.** These 300 requests were written by
an AI assistant to show the shape, scope and size of the data. Every item is marked
`source: assistant_draft`. The real dataset is written and reviewed by native Egyptian Arabic
speakers. Draft items never enter the locked test set, and no reported result is based on them.

## Files

| File | For | What it is |
|---|---|---|
| `athar_draft_for_pms.csv` | product managers | one row per request, plain-English columns; opens in Excel or Google Sheets |
| `athar_draft.jsonl` | engineers | the same items in the project's data format (`athar.schema.GoldItem`) |

## What the assistant is expected to do

Each request gets one of three behaviours:

| Behaviour | When | Example reply |
|---|---|---|
| **Act** (125 requests) | one reminder or alarm for the user, with everything it needs | `تمام، هفكرك بـ«أجيب فول وطعمية» يوم الخميس 8 أكتوبر الساعة 8:00 الصبح.` |
| **Ask a question** (125) | supported, but something is missing or unclear | `الساعة 8 الصبح ولا بالليل؟` |
| **Decline politely** (50) | outside what the assistant does | `معلش، أنا أقدر أعمل تذكير أو منبه لمرة واحدة ليك بس، ومقدرش أنفذ الطلب ده.` |

What can be missing (an "Ask" row can list several): which day (25 requests), what time (61),
morning or evening (32), a time that already passed (10), what to remind about (13).

Why requests are declined: deleting (9), editing (9), repeating schedules (9), for someone else
(8), several things at once (8), a different tool such as a timer or the weather (7).

## Coverage

- 178 reminder requests, 72 alarm requests, 50 out-of-scope requests.
- 63 requests (21%) mix Arabic and English (`remind me بكرة الساعة 9 الصبح ...`).
- 130 scenario groups. 100 of them come as a pair: a complete request and the same request
  with one detail removed, so each "Act" request has an "Ask" twin.
- Send times cover mornings, afternoons and late evenings on every weekday. Some requests
  fall on either side of Egypt's switch to winter time on 30 October 2026.

## Limits of this draft

- An assistant wrote it, so the wording is plausible, but it is not how a varied group of
  Egyptians actually type. Native speakers should expect to find phrasing that sounds off.
- The writer knew the system's date and time rules. All 300 expected answers match what
  the system does today. Real data will be harder.
- Names are invented. No real personal data.
