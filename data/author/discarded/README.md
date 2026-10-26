# Discarded first attempt at the author-written requests

`author_sheet_first_attempt.xlsx` (150 requests; previously `author_sheet_wave1.xlsx` at the
repository root, committed in `5fba0c9`) is the author's first attempt at the wave-1 requests.

It was discarded before labelling because most requests copied the card's typing moment into the
request («… يوم الحد 18 أكتوبر الساعة 11:35 بالليل»), so a "complete" request asked for the moment
it was sent and could not be ACT. `athar-data check-author` now flags exactly this.

None of its 150 texts appears in `data/athar.jsonl`; no split, model or result uses it. It is kept
for transparency, because it bears on how the final test texts were written: the first attempt
shows varied wording, while the second, used for the test split, reuses a small set of sentence
frames (docs/report.md, section 3.2).
