# Protocol v3: does human-written training data improve safe routing?

**Status: registered 2026-11-06 (gate G1)**, approved by the author. The commit that adds this file
is the registration. At that commit no v3 system exists, no v3 message is labelled, and no system
has been run on the v3 texts. Every later change goes into the deviations log (section 9) with its
date and its timing relative to gates G2-G5.

## 1. Question and motivation

Report section 9.2 (post hoc) found that adding about 215 contributor messages to train+dev raised
TF-IDF classifiers by 0.12-0.14 STS (leave-one-writer-out on the 238 confirmatory natural items).
Report section 9.1 found that MARBERTv2 trained on train only ties the rules on the author's test
but is 0.10 below them on the contributors' messages, mostly on out-of-scope requests. Both results
are exploratory: the first trains on folds of the locked natural set, and the second was run
after both evaluations had been read. This protocol tests them once, on messages no system has
been trained, selected or evaluated on.

## 2. Data

- **v3 set.** The screened messages of the contributors who were *not* drawn in the 2026-10-29
  sample (protocol_natural.md, log 2026-10-29; `data/natural/sample.json`). Counted on 2026-11-06
  from the `*_kept.csv` files in `data/raw/private/contrib/`: **236 messages from 17 writers, all
  confirmatory** (no scenario family in train or dev). The pool's 474 confirmatory messages are the
  natural set's 238 plus these 236. Earlier documents gave 227, which wrongly subtracted the natural
  set's 9 non-confirmatory items.

  | Block | v3 messages | v3 writers |
  |---|---|---|
  | block1 | 26 | AD1, PM1 |
  | block2 | 27 | PM2, R4 |
  | block3 | 45 | B1, PP1, PS1 |
  | block4 | 12 | F1 |
  | block5 | 30 | PO2, R1 |
  | block6 | 14 | C2 |
  | block7 | 16 | PR1 |
  | block8 (waves 2 and 2b, same cards) | 66 | AC, P2, W1, W2, W3 |
  | wave1-block5 | 0 | none (its only writer, H18, is in the natural set) |
- **Confirmatory items.** Messages whose scenario family is in neither train nor dev, as in
  protocol_natural.md: all 236.
- **Scenario overlap with the natural set.** The v3 writers wrote for the same scenario blocks as
  the natural-set writers (2 drawn per block from 3-6). Section 4 removes this overlap from
  training by block.
- **Exposure.** The author read these messages during screening (2026-10-26 to 2026-10-29), before
  any v3 system existed. No label, no system output and no statistic has been computed on them.
  This is disclosed and not otherwise controllable.

## 3. Order of events (gates)

| Gate | Event | Record |
|---|---|---|
| G1 | this protocol committed | commit hash |
| G2 | v3 systems built and frozen (`models_v3/freeze.json`: model hashes, configs, source hashes) | commit hash |
| G3 | every v3 system's predictions on the **unlabelled** v3 texts committed, including the GPU runs of section 6 | commit hash, `environment.json` per run |
| G4 | two independent annotations, joint adjudication recorded per item, gold locked (`athar-data lock`) | manifest SHA-256 |
| G5 | one scoring run (`athar-v3 evaluate`, refuses to overwrite its output) | `results/v3/evaluation.json` |

Nothing may be frozen, re-run or re-selected after G4. A system missing at G3 is not scored.

## 4. Systems

All systems use **core v1.1** (`athar.v2.core`), as in the natural-set evaluation. Hyper-parameters
are fixed; nothing is selected on the v3 set or on the natural set.

| System | Recognizer | Training pool | Role |
|---|---|---|---|
| S1 v1.1 | keyword rules (frozen, `models_v2`) | none | reference |
| **S3-drafted** | S3-gcv settings (C = 1, word 1-2-grams + char 2-5-grams; `models_v2/freeze.json`) | train + dev (338) | baseline for H-a |
| **S3-human** | same settings | train + dev + natural set minus the test block (section 4.1) | H-a |
| **S6-drafted** | MARBERTv2, recipe of `configs/train.yaml` (20 epochs, LR 2e-5, batch 16, warm-up 0.1, class-balanced loss), seeds 0-4 | train + dev (338) | baseline for H-b, H-c |
| **S6-human** | same recipe and seeds | train + dev + natural set minus the test block | H-b, H-c |

### 4.1 Block cross-fitting

The v3 set is split into 8 folds by scenario block (block1-block8). For the v3 messages of block
*b*, the `-human` systems are trained on train + dev + every natural-set message whose scenario
card is **not** among block *b*'s cards. Exclusion is by card, not by block label, because
`wave1-block5` (18 cards, natural-set writer H18) shares card `c012` with block5. No v3 message is
scored by a model that saw a natural-set message on its own scenario. This gives 8 fits per
`-human` system (40 for S6-human over 5 seeds).

`-drafted` systems need no cross-fitting: their pool contains no contributor message.

## 5. Hypotheses and tests

**Primary** (Holm over H-a and H-b, α = 0.05; confirmatory items):
- **H-a:** STS(S3-human) > STS(S3-drafted).
- **H-b:** STS(S6-human) > STS(S6-drafted), each the per-item mean over seeds 0-4.

**Secondary** (reported with unadjusted p):
- **H-c:** on gold-ABSTAIN confirmatory items, ABSTAIN success of S6-human > S6-drafted (per-item
  seed means).

**Estimation** (no test; 95% family-level bootstrap intervals):
- STS(S3-human) − STS(S1 v1.1) and STS(S6-human) − STS(S1 v1.1).
- Reading rule, fixed now: "within 0.10 of the rules" is stated only if the interval's lower bound
  is above −0.10; "not worse than the rules" is never claimed from this design.

**Descriptive:** the GPU systems of section 6, and every system's success by gold route.

### 5.1 Statistics

As protocol_natural.md: paired approximate randomization swapping whole scenario families
(10,000 permutations), family-level percentile bootstrap (10,000 resamples), seeds and level from
`configs/eval.yaml`; exact McNemar as a sensitivity check for single-seed systems. Sensitivity:
writer-clustered tests and leave-one-writer-out.

### 5.2 Minimum size and power

- **Minimum size.** At least 150 confirmatory items from at least 8 writers; otherwise every
  result is descriptive.
- **Power** (`athar.eval.stats.simulate_power`, 500 simulations × 2,000 permutations, α = 0.05).
  The v3 set has 236 messages in 67 scenario families (mean size 3.5), simulated as 67 families
  of 3 (201 items, conservative) and of 4 (268). Success rates are taken from report section 9.2:
  S3-drafted 0.580, S3-human 0.697. The coupling between the two systems is calibrated to the
  observed discordance (34 of 238 items, `discordance` = 0.074) and also run at 0.15.

  | Assumed effect | discordance 0.074 | discordance 0.15 |
  |---|---|---|
  | observed, +0.117 | 1.00 (size 3) / 1.00 (size 4) | 0.98 / 1.00 |
  | half, +0.059 | 0.74 / 0.85 | 0.58 / 0.75 |

  H-b assumes the same effect size, since no estimate for MARBERTv2 with human data exists. A
  margin-0.05 non-inferiority test against S1 would have about 0.21 power (normal approximation),
  which is why section 5 makes no non-inferiority claim.

## 6. GPU systems (descriptive; fair-run of Jais 3)

Run on the unlabelled v3 texts before G4, with outputs committed at G3. Exemplars come from train
and dev only.
- **S8-greedy:** Jais-3-7B tool calling with greedy decoding, inline exemplars and the annotation
  guide's clarification rules in the system prompt. New prompt code in `athar.v3`; the frozen
  `athar.agent` is not modified.
- **S4 Jais-3-7B inline:** 6-shot `prior`, exemplars inline (its best setting on both earlier sets),
  scored on normalized text and on raw text.
- **S9-greedy:** the S9 hybrid computed from S8-greedy's outputs.
- **Label tokenization** (checked 2026-11-06, `scripts/check_label_tokenization.py` →
  `results/posthoc/label_tokenization.json`). S4 tokenizes prompt and label separately. For all
  five S4 models at their pinned revisions (Jais-3-7B, Jais-3-3.7B, Jais 2 8B, Nile-Chat 4B,
  Qwen3 8B), the result is identical to tokenizing the joined text in 120 of 120 request-label pairs
  for each of 0-shot, 6-shot turns and 6-shot inline, so no correction is needed.

## 7. Annotation

- Two native annotators, the author and Ibrahim (the natural-set annotator; route κ 0.913 there),
  label every v3 item independently from blank sheets with the unchanged annotation guide.
- **Adjudication** (fixed now): the items where the two disagree on route, intent, targets, reason,
  datetime or reminder text are discussed by both annotators together. The record lists, per item,
  both labels, the final label and which annotator's label it follows, or that it is a third label.
  An AI assistant may check final labels against the written guide. Its corrections are listed per
  item and must be approved by both annotators.

## 8. What this protocol does not change

H1-H3 (locked test), N1-N3 (natural set) and every locked label and output stay as reported. The
natural set is used here only as training data.

## 9. Deviations log

| Date | Change | Reason | Timing relative to G1-G5 |
|---|---|---|---|
| | | | |
