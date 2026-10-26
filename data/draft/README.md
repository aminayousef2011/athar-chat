# ATHAR Chat: draft dataset v2 (project-manager focus)

**Status: assistant-drafted training and development data, never test data.** These 300 requests
were written by an AI assistant, first to show the shape, scope and size of the data and to
dry-run the tooling. Every item is marked `source: assistant_draft`. The planned native
contributor collection did not happen (protocol log 2026-10-10), so the final dataset
(`data/athar.jsonl`) uses them as 261 of the 298 training items and 39 of the 40 dev items. The
locked test split is written entirely by the author, and no draft item enters it. Every trained
system, and every dev-based choice, therefore rests mostly on these drafts
([docs/report.md](../../docs/report.md), sections 3 and 7).

## Who the requests come from

| Domain (`domain` field) | Items | What they are |
|---|---|---|
| `project_management` (**main**) | 210 (70%) | Egyptian project managers about their work: standups, sprint reviews and retros, status reports, deadlines, client and vendor follow-ups, Jira, sign-offs, UAT and go-lives, budgets, timesheets, early alarms for site visits |
| `general` | 90 (30%) | everyday personal requests (bread, medicine, family calls), whole scenario groups kept from draft v1 |

Results are reported per domain, so performance on project-manager requests is visible on its own.

## Files

| File | For | What it is |
|---|---|---|
| `athar_draft_for_pms.csv` | reviewers, project managers | one row per request, plain-English columns, including Domain and the assistant's reply; opens in Excel or Google Sheets |
| `athar_draft.jsonl` | engineers | the same items in the project's data format (`athar.schema.GoldItem`) |
| `split.yaml`, `manifest.json` | engineers | dry-run split (train 260 / dev 40, no test: drafts never enter test) |

## What the assistant is expected to do

| Behaviour | Project management | General | Example reply |
|---|---|---|---|
| **Act** (create it) | 88 | 37 | `تمام، هفكرك بـ«أحضر الـ standup» يوم الخميس 8 أكتوبر الساعة 9:30 الصبح.` |
| **Ask a question** | 88 | 37 | `الساعة 9:30 الصبح ولا بالليل؟` |
| **Decline politely** | 34 | 16 | `معلش، أنا أقدر أعمل تذكير أو منبه لمرة واحدة بس، ومقدرش أنفذ الطلب ده.` |

Replies are gender-neutral (task contract v1.0).

What can be missing in the project-manager "Ask" requests (a request can lack several things):
what time (44), which day (22), morning or evening (22), a time that already passed (7), what to
remind about (4).

Why project-manager requests are declined, mirroring what PMs actually ask for:
- **repeating** (7): "remind me about the standup every day"
- **editing** (7): "move the client-call reminder to 4"
- **for someone else** (6): "remind the team …"
- **deleting** (5)
- **several things at once** (5)
- **another tool** (4): "create a Jira ticket", "book a meeting room"

## Coverage

- Project management: 153 reminders, 23 alarms, 34 out of scope, in 87 scenario groups, 68 of
  them with a complete/incomplete pair. **87% mix Arabic and English**
  (`فكرني أراجع الـ risk register قبل اجتماع الإدارة`), as Egyptian PMs do. This is higher than in
  everyday speech, and native reviewers should check it is realistic.
- General: 46 reminders, 28 alarms, 16 out of scope; 22% code-switched.
- Send times cover the Egyptian work week (Sunday to Thursday) at different times of day, with
  some weekend and late-evening requests. Some requests fall on either side of the switch to
  winter time on 30 October 2026.

## Limits of this draft

- An assistant wrote it, so the wording is plausible, but it is not how a varied group of
  Egyptian project managers actually type. Native speakers should expect phrasing that
  sounds off.
- The writer knew the system's date and time rules, and all 300 expected answers match what
  the system does today. Real data will be harder.
- Names and companies are invented. No real personal or client data.
