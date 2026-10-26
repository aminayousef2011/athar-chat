# Data card: ATHAR Chat routing dataset

Status: **split and frozen 2026-10-12** (`data/splits/manifest.json`, manifest
`940beba352ec`). 458 items: 158 written by the author, a native Egyptian Arabic speaker
(`source: author`), and the 300 assistant-drafted items (`source: assistant_draft`), which are
kept out of test. Second-annotator agreement (2026-10-15): intent κ 0.92, route κ 1.00
([agreement.md](agreement.md)).

## Summary

| | |
|---|---|
| Task | route one Egyptian Arabic request to ACT / CLARIFY / ABSTAIN, with tool arguments or clarification targets |
| Size | 458 single-turn requests: 158 author-written + 300 assistant-drafted |
| Language | Egyptian Arabic (Arabic script), with Arabic-English code-switching; Arabizi excluded |
| Users | about 70% project managers about their work (`domain: project_management`), 30% general requests |
| Labels | intent, route, clarification targets, unsupported reason, resolved datetime, reminder text |
| Format | JSONL, one item per line, schema `athar.schema.GoldItem` |
| Splits | by scenario family: train 298, dev 40, **test 120 (50 ACT / 50 CLARIFY / 20 ABSTAIN), all author-written** |
| Authors | Amina Yousef (author, native speaker); second annotator: Ibrahim (native speaker, 60-item agreement sample) |
| Contract | [task_contract.md](task_contract.md) v1.1 (annotation rules frozen at v1.0, 2026-10-07) |
| Licence | _TBD_ |

## Motivation

The dataset tests one claim: whether lightweight text classifiers (word TF-IDF + logistic
regression, with and without character n-grams) route requests more safely than keyword
rules, when everything except intent recognition is shared. Safety means acting only when a
request is supported and complete, asking when it is incomplete, and declining when it is out
of scope. No public Egyptian Arabic dataset labels all three outcomes with tool arguments,
so the requests are written for this project.

## Composition

- **Instances**: a request text, the moment it is sent (`ref_datetime`, `Africa/Cairo`), and
  the label. One instance is one turn; there is no dialogue history.
- **Scope**: two simulated tools, `create_reminder(datetime, reminder_text)` and
  `create_alarm(datetime)`. Out of scope, labelled ABSTAIN: deletion, modification, recurring
  schedules, requests for another person, several actions in one request, other tools.
- **Families and pairs**: paraphrases and variants of one scenario share a `family_id`.
  Complete/incomplete pairs (`pair_id`) differ only in the detail that makes one of them
  CLARIFY.
- **Planned mix**: about 125 ACT, 125 CLARIFY and 50 ABSTAIN. Every clarification target and
  every unsupported reason is covered. About a quarter of the items are code-switched.
  Some items resolve across Cairo's DST change.
- **Domain**: about 70% of the requests come from project managers about their work (standups,
  sprint reviews, status reports, deadlines, client and vendor follow-ups, Jira, sign-offs).
  The rest are general personal requests. Results are reported per domain.
- **Personal data**: none. Names are invented; no real contact details.

## Collection and annotation

- **Author-written items (158)**: written by the author from scenario cards that describe a
  situation in English with no Arabic example wording (`contrib/author_sheet_wave1.xlsx`,
  `contrib/author_sheet_extra_oos.xlsx`), checked with `athar-data check-author`, then labelled
  blind to the cards' plans (`data/author/labelling_sheet.xlsx`). Labels follow what was
  written: 29 items differ from their card's plan, and 19 A/B versions that turned out not to be
  minimal pairs (18 different tools, 1 both complete) were unpaired by `athar-data build`
  without changing their text. A first attempt and a version with texts edited after labelling
  were discarded (protocol section 10). The first attempt is kept for transparency in
  `data/author/discarded/` (it was committed in `5fba0c9`); the edited version is in
  `data/raw/private/`, not committed. Neither is used anywhere.
- **Test construction (found after the test run).** The 120 test texts reuse a small set of
  sentence frames: 52 distinct frames, the most frequent used 10 times, and some frame and purpose
  combinations are implausible («صحيني … 6 بالليل عشان اصحى بدري»). 59 of the 120 share a frame with
  a training item, 51 of them through the author's 37 training items. Read the test as one
  writer's templated phrasing, not a sample of natural requests (report, section 3.2).
- **Assistant-drafted items (300)**: training and development data only; the splitter keeps
  them out of test. The 20 pilot items are not part of the dataset.
- They are written requests, not collected from users, so they are plausible requests, not a
  sample of real traffic. One writer produced the whole test split (protocol 4.2, threats).
- Labels follow the task contract. Gold datetimes are resolved by hand, not by the system.
  Annotators do not run any system while labelling.
- Validation: `athar-data validate --strict` must pass: schema, label consistency, unique ids,
  UTC offsets that match the zone at each instant, pair structure, and no duplicate text
  across families.

## Agreement

| | |
|---|---|
| Sample | 60 author-written items, route-stratified (23 ACT / 26 CLARIFY / 11 ABSTAIN), one per family where possible |
| Second annotation | Ibrahim, native Egyptian Arabic speaker, independent, 2026-10-15 (first pass) |
| Intent κ | **0.916** (95.0% agreement) |
| Route κ | **1.000** (100%) |
| Target-set agreement (both CLARIFY) | 92.3% (24 of 26); per-target κ 0.82-1.00 |
| Unsupported reason (both ABSTAIN) | 100% (11 of 11) |
| Moment / reminder text (both ACT) | 100% (23 of 23) / 100% (15 of 15) |
| Report | [agreement.md](agreement.md) / `docs/agreement.json` (from `athar-data kappa`) |

κ is reported on the annotator's first, independent pass, before any discussion. The split was
made before the second annotation (2026-10-12), as the protocol allowed ("if adjudication changes
labels, the split is rebuilt"). No gold label was changed: the five disagreements are kept and
discussed in [agreement.md](agreement.md). Two of them expose a genuine ambiguity of the contract,
«فكرني … اصحى» ("remind me … to wake up"), which stays a reminder in gold and is listed under
limitations below.

## Splits

Made once by `athar-data split` with [data/split_config.yaml](../data/split_config.yaml)
(seeded; the same file and config give a byte-identical manifest). Whole families go to one
split. The test split must hit 50 / 50 / 20 exactly. Dev takes 40 items with roughly the
route mix of what remains, and train is the rest. `data/splits/manifest.json` stores a SHA-256
per item, the dataset hash, the config and the contract version, and is hashed itself.
`athar-data verify` fails on any later edit. The test split is loaded only by the Phase 4
evaluation (`load_split(..., unlock_test=True)`).

| | Value |
|---|---|
| Manifest SHA-256 | `940beba352ec508c1811f1f7b819ea6a3190fe0b7b91d99b2b5ff1e7c350db50` |
| Dataset SHA-256 | `c2c7cfe4c6ae64613de0b90c00e1c211a89bd133cf0b471cf8bc2fba6420ba37` (`data/athar.jsonl`) |

## Statistics

<!-- stats:start -->
Generated by `athar-data stats` from `data/athar.jsonl`.

458 items in 225 families (family size 1-4), 139 complete/incomplete pairs; request length 2-15 words (median 8).

| | all | train | dev | test |
|---|---|---|---|---|
| route ACT | 186 | 120 | 16 | 50 |
| route CLARIFY | 194 | 127 | 17 | 50 |
| route ABSTAIN | 78 | 51 | 7 | 20 |
| intent create_reminder | 285 | 203 | 20 | 62 |
| intent create_alarm | 95 | 44 | 13 | 38 |
| intent unsupported | 78 | 51 | 7 | 20 |
| target `date` | 34 | 26 | 4 | 4 |
| target `time` | 92 | 62 | 12 | 18 |
| target `am_pm` | 56 | 30 | 3 | 23 |
| target `past` | 21 | 14 | 0 | 7 |
| target `reminder_text` | 26 | 17 | 0 | 9 |
| reason deletion | 10 | 4 | 2 | 4 |
| reason modification | 17 | 13 | 0 | 4 |
| reason recurring | 12 | 5 | 3 | 4 |
| reason other_person | 11 | 9 | 1 | 1 |
| reason multi_action | 12 | 9 | 1 | 2 |
| reason other_tool | 16 | 11 | 0 | 5 |
| reason other | 0 | 0 | 0 | 0 |
| code-switched | 234 | 191 | 19 | 24 |
| domain project_management | 323 | 204 | 29 | 90 |
| domain general | 135 | 94 | 11 | 30 |
| source author | 158 | 37 | 1 | 120 |
| source contributor | 0 | 0 | 0 | 0 |
| source assistant_draft | 300 | 261 | 39 | 0 |
| **items** | 458 | 298 | 40 | 120 |
<!-- stats:end -->

Regenerate with
`uv run athar-data stats data/athar.jsonl --manifest data/splits/manifest.json --card docs/data_card.md`.

## Intended use and limitations

- **Intended use**: comparing intent recognizers inside one fixed routing pipeline, in a
  course project. It is not a training set for a production assistant.
- **Size**: 120 test items give wide confidence intervals. Results are reported with a
  family-level bootstrap.
- **Written, not observed**: the requests reflect what the writers think people type.
  Spelling noise, regional variety and request length may differ from real traffic.
- **Few annotators**: one or two native speakers, mostly from one region. Labels follow the
  contract's conventions (weekday rules, AM/PM periods), which other speakers may read
  differently. They were confirmed explicitly at contract v1.0.
- **Address form**: replies are gender-neutral (contract v1.0); many requests are written in
  the masculine, as people type them.
- **Script**: Arabizi is excluded, though many Egyptian users write it.
- **Reminder or alarm**: «فكرني … اصحى» ("remind me … to wake up") is a reminder by its verb and an
  alarm by its purpose. Gold follows the verb (reminder); the second annotator chose alarm on both
  such items in the agreement sample (c021-b, c121-b; [agreement.md](agreement.md)).

## Maintenance

Frozen after the split. Corrections found later are recorded in an erratum in the report, not
patched into the dataset, so the locked test results remain reproducible.
