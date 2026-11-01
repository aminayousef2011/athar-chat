# Annotation agreement (Ibrahim (vs the author's labels, before adjudication), 2026-11-01)

Items compared: 247

| Field | n | Agreement | Cohen's κ | Reading |
|---|---|---|---|---|
| intent | 247 | 96.0% | 0.938 | almost perfect |
| route | 247 | 94.3% | 0.913 | almost perfect |
| target `date` (present/absent) | 247 | 89.9% | 0.366 | fair |
| target `time` (present/absent) | 247 | 91.9% | 0.509 | moderate |
| target `am_pm` (present/absent) | 247 | 96.0% | 0.834 | almost perfect |
| target `past` (present/absent) | 247 | 98.8% | 0.908 | almost perfect |
| target `reminder_text` (present/absent) | 247 | 99.6% | 0.963 | almost perfect |
| unsupported_reason (both unsupported) | 57 | 89.5% | 0.870 | almost perfect |
| target set (both CLARIFY) | 91 | 64.8% | | |
| datetime (both ACT) | 85 | 71.8% | | |
| reminder text (both ACT reminders) | 46 | 89.1% | | |

**intent confusion** (rows: gold, columns: annotator)

| | create_reminder | create_alarm | unsupported |
|---|---|---|---|
| create_reminder | 100 | 3 | 6 |
| create_alarm | 1 | 80 | 0 |
| unsupported | 0 | 0 | 57 |

**route confusion** (rows: gold, columns: annotator)

| | ACT | CLARIFY | ABSTAIN |
|---|---|---|---|
| ACT | 85 | 5 | 5 |
| CLARIFY | 3 | 91 | 1 |
| ABSTAIN | 0 | 0 | 57 |

**Label disagreements** (intent/route/targets)

| id | gold | annotator |
|---|---|---|
| c003-b-pas | create_reminder/CLARIFY/am_pm | create_reminder/CLARIFY/date,time |
| c003-b-r2 | create_reminder/CLARIFY/date | create_reminder/CLARIFY/date,time |
| c007-a-w5 | create_alarm/CLARIFY/past | create_alarm/ACT/ |
| c008-b-e1 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/am_pm,reminder_text |
| c008-b-k1 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/date,time,reminder_text |
| c009-b-po1 | create_alarm/CLARIFY/date | create_alarm/CLARIFY/date,time |
| c010-a-c1 | create_reminder/ACT/ | create_reminder/CLARIFY/date |
| c010-b-c1 | create_reminder/CLARIFY/am_pm | create_reminder/CLARIFY/date,am_pm |
| c012-a-h18 | create_reminder/ACT/ | unsupported/ABSTAIN/ |
| c012-b-pas | create_reminder/CLARIFY/reminder_text | create_alarm/CLARIFY/date,time,reminder_text |
| c015-b-7b | create_reminder/CLARIFY/date | create_reminder/CLARIFY/date,time |
| c015-b-pm12 | create_reminder/CLARIFY/date | create_reminder/CLARIFY/date,time |
| c017-a-w5 | create_reminder/ACT/ | create_reminder/CLARIFY/date |
| c017-b-h18 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/past |
| c018-a-e1 | create_reminder/ACT/ | create_reminder/CLARIFY/date |
| c018-b-e1 | create_reminder/CLARIFY/date | create_reminder/CLARIFY/time |
| c020-b-po1 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/date,time,reminder_text |
| c023-b-h18 | create_reminder/CLARIFY/am_pm | create_reminder/CLARIFY/date,time |
| c025-b-d1 | create_alarm/ACT/ | create_alarm/CLARIFY/am_pm |
| c026-b-7b | create_alarm/CLARIFY/date | create_alarm/CLARIFY/date,time |
| c029-b-h18 | create_reminder/ACT/ | create_reminder/CLARIFY/date |
| c030-b-po1 | create_reminder/CLARIFY/date | create_reminder/CLARIFY/am_pm |
| c035-b-b2 | create_reminder/CLARIFY/reminder_text | create_alarm/CLARIFY/reminder_text |
| c038-b-f2 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/time,reminder_text |
| c038-b-pcs1 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/time,reminder_text |
| c042-b-pas | create_reminder/CLARIFY/date | create_reminder/CLARIFY/time |
| c042-b-r2 | create_reminder/CLARIFY/date | create_reminder/CLARIFY/am_pm |
| c047-b-pf1 | create_alarm/CLARIFY/date | create_alarm/CLARIFY/date,time |
| c048-a-7b | create_alarm/CLARIFY/date | create_alarm/ACT/ |
| c052-a-c1 | create_alarm/CLARIFY/date | create_alarm/ACT/ |
| c052-a-h18 | create_reminder/ACT/ | unsupported/ABSTAIN/ |
| c052-b-c1 | create_alarm/CLARIFY/time | create_alarm/CLARIFY/date,time |
| c052-b-h18 | create_reminder/CLARIFY/time | unsupported/ABSTAIN/ |
| c053-b-pcs1 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/date,time,reminder_text |
| c056-b-pas | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/date,time,reminder_text |
| c056-b-r2 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/am_pm,reminder_text |
| c076-b-h18 | create_alarm/CLARIFY/am_pm | create_reminder/CLARIFY/am_pm |
| c090-b-w4 | create_alarm/CLARIFY/am_pm | create_alarm/CLARIFY/date,am_pm |
| c090-b-w5 | create_alarm/CLARIFY/date | create_alarm/CLARIFY/am_pm |
| c091-b-k1 | create_reminder/CLARIFY/time | create_reminder/CLARIFY/date,time |
| c097-b-b2 | create_reminder/CLARIFY/past | create_alarm/CLARIFY/am_pm |
| c097-b-c1 | create_alarm/CLARIFY/time | create_alarm/CLARIFY/date,time |
| c100-a-h18 | create_reminder/ACT/ | unsupported/ABSTAIN/ |
| c100-b-h18 | create_reminder/ACT/ | unsupported/ABSTAIN/ |
| c102-b-d1 | create_reminder/CLARIFY/reminder_text | create_reminder/CLARIFY/date,time,reminder_text |
| c106-b-7b | create_reminder/CLARIFY/date | create_reminder/CLARIFY/date,am_pm |
| c106-b-pm12 | create_reminder/CLARIFY/date | create_reminder/CLARIFY/time |
| c129-a-h18 | create_reminder/ACT/ | unsupported/ABSTAIN/ |
