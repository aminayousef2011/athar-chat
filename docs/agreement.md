# Annotation agreement (Ibrahim (second annotator, independent first pass), 2026-10-15)

Items compared: 60

| Field | n | Agreement | Cohen's κ | Reading |
|---|---|---|---|---|
| intent | 60 | 95.0% | 0.916 | almost perfect |
| route | 60 | 100.0% | 1.000 | almost perfect |
| target `date` (present/absent) | 60 | 100.0% | 1.000 | almost perfect |
| target `time` (present/absent) | 60 | 98.3% | 0.946 | almost perfect |
| target `am_pm` (present/absent) | 60 | 100.0% | 1.000 | almost perfect |
| target `past` (present/absent) | 60 | 100.0% | 1.000 | almost perfect |
| target `reminder_text` (present/absent) | 60 | 96.7% | 0.815 | almost perfect |
| unsupported_reason (both unsupported) | 11 | 100.0% | 1.000 | almost perfect |
| target set (both CLARIFY) | 26 | 92.3% | | |
| datetime (both ACT) | 23 | 100.0% | | |
| reminder text (both ACT reminders) | 15 | 100.0% | | |

**intent confusion** (rows: gold, columns: annotator)

| | create_reminder | create_alarm | unsupported |
|---|---|---|---|
| create_reminder | 31 | 3 | 0 |
| create_alarm | 0 | 15 | 0 |
| unsupported | 0 | 0 | 11 |

**route confusion** (rows: gold, columns: annotator)

| | ACT | CLARIFY | ABSTAIN |
|---|---|---|---|
| ACT | 23 | 0 | 0 |
| CLARIFY | 0 | 26 | 0 |
| ABSTAIN | 0 | 0 | 11 |

**Label disagreements** (intent/route/targets)

| id | gold | annotator |
|---|---|---|
| c005-b | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/time |
| c011-b | create_reminder/CLARIFY/past | create_alarm/CLARIFY/past |
| c021-b | create_reminder/CLARIFY/time | create_alarm/CLARIFY/time |
| c121-b | create_reminder/CLARIFY/am_pm | create_alarm/CLARIFY/am_pm |
| c153-a | create_reminder/CLARIFY/time,reminder_text | create_reminder/CLARIFY/time |

## How this was measured

- **Who.** Ibrahim, a native Egyptian Arabic speaker, labelled the 60-item, route-stratified sample
  (`athar-data agreement-sheet`; author-written items only, one per family where possible) on his
  own, from the blind sheet and `docs/annotation_guide.md`, without the gold labels.
- **Which pass.** κ is computed on his **first, independent pass**: the sheet as returned at 15:03 on
  2026-10-15, preserved in git as `sheet_Ibrahim.xlsx` at commit `5fba0c9` (SHA-256
  `b6b7cd3de4761dc9a789e7aa984bbbabb3f1a730715ad638d35d038dabe6ca9b`). Re-running `athar-data kappa` on it (after the date conversion
  below) reproduces every number above. Later revisions of the sheet, made after the
  disagreements were known, are adjudication input only and are not used here.
- **Dates.** He wrote dates day/month, as the task contract does. Excel turned the entries with a day
  of 12 or less into dates with day and month swapped (3/11 → 11 March). The only change made to the
  sheet was converting that column back, mechanically and without reference to the gold labels. All
  23 moments then agree with gold. `athar-data kappa` now reads day/month text and refuses dates
  Excel converted, so this cannot pass silently.
- **No adjudication changes.** The five disagreements are kept as they are, and the gold labels are
  unchanged, so the frozen split (`940beba352ec`) stands. The disagreements are part of the evidence
  that the second annotation was independent.

## Reading the disagreements

| Item | Split | What differs | Reading |
|---|---|---|---|
| c021-b, c121-b | test | «فكرني … اصحى …» labelled alarm, gold reminder | A genuine ambiguity of the contract: "remind me … to wake up" is a reminder by its verb and an alarm by its purpose. Documented as a known limitation; both items stay reminders in gold. |
| c011-b | test | «فكرني … عشان اراجع فرق الميزانية» labelled alarm | Annotator slip (the content is a task, not waking up). |
| c005-b | train | «فكرني بكرة الساعة 8 الصبح بحاجة»: target `time` instead of `reminder_text` | Annotator slip: the time is given; the content (بحاجة) is vague. |
| c153-a | train | «فكرني بـ ابعت لزميلي بخصوص الابديت»: only `time`, gold `time` + `reminder_text` | Borderline: whether «بخصوص الابديت» is specific enough is a judgement call. |

Route agreement is perfect; all intent disagreements are reminder vs alarm on a reminder verb
(«فكرني»), never in-scope vs out-of-scope, so they do not touch the safety-critical decision to
execute or abstain.
