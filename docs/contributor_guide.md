# ATHAR Chat: guide for contributors

Thank you for helping. (Most contributors get a **Google Form** link instead of the CSV files below; the
form shows the same cards and rules.) ATHAR Chat is a university NLP project: an assistant that sets **one
reminder or one alarm** from a message written in **Egyptian Arabic**, asks a question when
something is missing, and politely declines anything else. To test it fairly, we need requests
written by **native Egyptian Arabic speakers, in their own words**. Most of the requests come
from **project managers** about their work.

## What you do

You get a few **scenario cards** (`contrib/scenario_cards.csv`) and type, for each one, the
message you would really send to an assistant like this. It takes about 30-40 minutes for
35-40 messages.

Each card gives you:

| Column | Meaning |
|---|---|
| `situation` | what you want (in English, so you are not influenced by someone else's wording) |
| `now_local_en` / `now_local_ar` | imagine you are typing at this moment (it matters for "tomorrow", "tonight", ...) |
| `version_a` | write the full request |
| `version_b` | (some cards) write the same request again, leaving out what it says |

Type your messages in `contrib/collection_sheet.csv` (it opens in Excel or Google Sheets): one
row per card and version. Fill in:

- `text`: your message, exactly as you would type it;
- `contributor`: your contributor code (for example `P03`), the same on every row you write;
- `comment` (optional): anything unclear about the card.

## Rules

1. **Write it yourself, the way you really type.** Egyptian Arabic, short and informal. Mixing
   in English work words (`standup`, `deadline`, `Jira`, `UAT`) is good if you normally do it.
   Typos and Arabic or Western digits are fine.
2. **No AI tools, translators or copying.** Do not use ChatGPT, Gemini, Google Translate or
   similar tools, and do not copy messages from other people or from examples. The whole point
   is real native wording. If a request was not written by you, leave the row empty.
3. **Follow the card, but stay natural.** If a card asks you to leave something out, leave
   it out the way people really do. You don't need to sound like a test.
4. **No private data.** Use invented names (`أحمد`, `منى`, `شركة النور`). Never real phone numbers,
   addresses, client names or company secrets.
5. **Pure Arabizi** (`fakarni bokra`) is out of scope: use Arabic script, with English words if
   you like.

## How your messages are used

- They become the project's evaluation dataset (`data/athar.jsonl`). Researchers label each
  message (what the assistant should do). The labels are the researchers' job, not yours.
- Your contributor code is stored with each message so that quality can be checked. Your name
  is not stored in the dataset.
- The dataset is used for the course project and its report. Tell the project team before you
  start if you do **not** agree to it being shared publicly with the report (for example on
  GitHub) under an open licence. Without that agreement, it stays private
  (`data/raw/private/` is never committed).
- You can withdraw your messages at any time before the dataset is frozen. Ask the project
  team to remove the rows with your code.

---

## ملخص بالعربي

- هتاخد كام **كارت سيناريو**. كل كارت بيوصف موقف (بالإنجليزي، عشان متتأثرش بكلام حد).
- اكتب الرسالة اللي **إنت فعلاً هتبعتها** للمساعد، بالمصري وبطريقتك. لو بتكتب كلمات إنجليزي في الشغل
  (standup، deadline، Jira) عادي جداً.
- `version_a`: اكتب الطلب كامل. `version_b`: نفس الطلب، بس شيل الحاجة اللي الكارت بيقول عليها.
- **ممنوع** ChatGPT أو أي أداة ذكاء اصطناعي أو ترجمة، وممنوع تنقل كلام حد تاني.
- متكتبش بيانات حقيقية: أسماء وأرقام من خيالك.
- اكتب الكود بتاعك (زي `P03`) في عمود `contributor` في كل سطر.

---

## For the project team: from cards to dataset

**Collect wave 1 first.** Only the locked test split (120 items) must be native-written, so
wave 1 (90 cards, 150 requests) covers it with margin. Wave 2 is a backup and extra dev data.
Messages for recruiting, reminders and the instructor: [contrib/messages.md](../contrib/messages.md).

```bash
uv run athar-data cards --out contrib                    # 175 cards (column `wave`), 300 requests
uv run athar-data form-script --cards contrib/scenario_cards.csv --wave 1 --people 5 --out contrib/make_forms.gs
# script.google.com -> New project -> paste make_forms.gs -> Run createForms -> copy the links from the log
# one form + one contributor code per person; afterwards Responses -> Download responses (.csv)
uv run athar-data import-contrib form_export.csv --cards contrib/scenario_cards.csv --out runs/labelling/sheet.csv
# label runs/labelling/sheet.csv with docs/annotation_guide.md (the card plans are not in it)
uv run athar-data build runs/labelling/sheet.csv --cards contrib/scenario_cards.csv --out data/athar.jsonl
uv run athar-data validate data/athar.jsonl --strict     # then agreement, kappa, split (guide section 7)
```

- **Coverage.** The cards follow annotation guide 2.2: 125 pair cards (version A complete,
  version B with planned gaps, so each pair is one ACT and one CLARIFY item), 50 out-of-scope
  cards (every unsupported reason 7-10 times), about 70% project management, reference times on
  every weekday from early morning to late night, on both sides of Cairo's DST change (29 Oct
  2026). Each card is one family.
- **Blind labelling.** The labelling sheet has the request and the reference time, not the
  card's plan. Label what was written: a contributor may have left out more or less than the
  card asked. `build --cards` lists the items whose label differs from the plan, as a check on
  both.
- **Spread contributors over cards.** Give each person cards from both domains and all three
  route plans, so that no system can learn one person's style for one label.
- **Unwritten or unusable rows** are simply left out by `import-contrib`. Re-issue those cards
  to someone else if the coverage targets need them.
