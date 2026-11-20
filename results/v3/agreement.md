# Annotation agreement (Ibrahim, 2026-11-13)

Items compared: 227

| Field | n | Agreement | Cohen's κ | Reading |
|---|---|---|---|---|
| intent | 227 | 99.6% | 0.993 | almost perfect |
| route | 227 | 97.4% | 0.959 | almost perfect |
| target `date` (present/absent) | 227 | 99.1% | 0.955 | almost perfect |
| target `time` (present/absent) | 227 | 100.0% | 1.000 | almost perfect |
| target `am_pm` (present/absent) | 227 | 99.6% | 0.985 | almost perfect |
| target `past` (present/absent) | 227 | 97.8% | 0.751 | substantial |
| target `reminder_text` (present/absent) | 227 | 100.0% | 1.000 | almost perfect |
| unsupported_reason (both unsupported) | 50 | 100.0% | 1.000 | almost perfect |
| target set (both CLARIFY) | 85 | 98.8% | | |
| datetime (both ACT) | 86 | 95.3% | | |
| reminder text (both ACT reminders) | 51 | 96.1% | | |

**intent confusion** (rows: gold, columns: annotator)

| | create_reminder | create_alarm | unsupported |
|---|---|---|---|
| create_reminder | 104 | 0 | 0 |
| create_alarm | 0 | 72 | 0 |
| unsupported | 1 | 0 | 50 |

**route confusion** (rows: gold, columns: annotator)

| | ACT | CLARIFY | ABSTAIN |
|---|---|---|---|
| ACT | 86 | 0 | 0 |
| CLARIFY | 5 | 85 | 0 |
| ABSTAIN | 0 | 1 | 50 |

**Label disagreements** (intent/route/targets)

| id | gold | annotator |
|---|---|---|
| c017-a-p2 | create_alarm/CLARIFY/past | create_alarm/ACT/ |
| c027-a-ac | create_reminder/CLARIFY/past | create_reminder/ACT/ |
| c027-a-p2 | create_reminder/CLARIFY/past | create_reminder/ACT/ |
| c090-a-ac | create_reminder/CLARIFY/past | create_reminder/ACT/ |
| c090-a-p2 | create_alarm/CLARIFY/past | create_alarm/ACT/ |
| c091-b-ad1 | create_reminder/CLARIFY/date,time | create_reminder/CLARIFY/time |
| c097-a-b1 | create_alarm/ACT/ | create_alarm/ACT/am_pm |
| c144-a-ac | unsupported/ABSTAIN/ | create_reminder/CLARIFY/date |

Ids not in the gold dataset (ignored): ['c017-b-p2', 'c091-a-ad1', 'c097-a-pp1', 'c109-b-p2']
