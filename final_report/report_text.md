ATHAR Chat: Safe Tool Routing for Egyptian Arabic


Amina Yousef


## Abstract


A reminder assistant must decide whether to execute a request (ACT), ask for missing details (CLARIFY) or decline it (ABSTAIN). On a new Egyptian Arabic benchmark (458 requests), we compared keyword rules with TF-IDF logistic-regression intent classifiers in one shared pipeline, scored by Safe Task Success (STS: correct decision and arguments). On 238 messages from 17 new writers, the rules won (STS 0.718 vs 0.513, Holm p = 0.0003). The mostly AI-drafted training data explain much of this gap: on 236 messages from 17 further writers, a pre-registered test showed that adding in-domain human messages raises TF-IDF by +0.089 and MARBERTv2 by +0.088 STS (Holm p ≤ 0.002); retrained TF-IDF matched the rules (0.589 each). The extractor limits every system (0.786 STS with gold intents); four post hoc extractor fixes added 0.047-0.051 STS on the held-out writers, but none is deployable.


## 1. Motivation and objectives


A reminder assistant must not execute a request whose time or content is missing or ambiguous, re-ask for what the user gave, or attempt what it cannot do. Egyptian Arabic makes this harder: messages are informal («اعمللي تذكير بعد بكرة الساعة 9 بالليل عشان اخد الدوا», "set me a reminder the day after tomorrow at 9 pm to take my medicine"), inconsistently spelled and mixed with English. When2Call [1] showed for English that deciding when to call a tool is a problem of its own, and out-of-scope requests are hard for intent classifiers [6]; we know of no Egyptian Arabic dataset labelling this decision. Intended users: teams building dialect assistants with little compute, and researchers who need such a benchmark.


Research questions. (RQ1) When all systems share the same argument extractor and execution policy, does lightweight supervised intent classification improve Safe Task Success over keyword rules? (RQ2) Does human-written instead of AI-drafted training data change the answer? Objectives: (O1) a task contract and safety-aware metric; (O2) a reviewed Egyptian Arabic benchmark split by scenario; (O3) rules vs. TF-IDF classifiers in one pipeline; (O4) a test on writers other than the author; (O5) the errors limiting the pipeline.


## 2. Deliverables


• Repository: https://github.com/aminayousef2011/athar-chat (branch report-tier1), commit 69807223ab25dca8f2cd46b5826b0311cabf4fa8; PDF added by the next commit.


• Data: data/athar.jsonl (458 items; splits data/splits/manifest.json), data/natural/natural.jsonl (247), data/v3/manifest.json (texts private), data card and guides (docs/); consent covers this project, no public licence yet.


• Code, models, outputs: package athar with 402 tests and locked dependencies (uv.lock); frozen classifiers (models/, models_v2/), stored predictions (results/), pinned Hugging Face models (Table 14), a Streamlit demo (app/) and the full technical report (docs/report.md); S6/S7 weights are not stored.


## 3. Methods and implementation


## 3.1 Task contract and metric


The simulated tools are create_reminder(datetime, reminder_text) and create_alarm(datetime); each request has a sending time. The contract resolves explicit dates, today/tomorrow, weekdays with «الجاي» (next), clock times and period words such as «الصبح». ACT needs every argument resolved to one future instant; CLARIFY names the missing slots (date, time, AM/PM, past time, reminder text); ABSTAIN covers deletion, modification, recurrence, other people's items, several actions and other tools. Safe Task Success counts an item only if ABSTAIN is declined, CLARIFY asks for exactly the gold slots, or ACT calls the gold tool at the gold instant with the gold reminder text (exact after normalization, request words and fillers removed); invalid outputs fail.


## 3.2 Data


Table 1. Datasets. Train, dev and test share no scenario family; natural and v3 reuse the test's.

| Set | Items | Families | ACT/CLAR/ABST | Written by | Use |
| train | 298 | 137 | 120 / 127 / 51 | 261 AI-drafted, 37 author | fitting |
| dev | 40 | 17 | 16 / 17 / 7 | 39 AI-drafted, 1 author | selection |
| test (locked) | 120 | 71 | 50 / 50 / 20 | author (native speaker) | evaluation 1 |
| natural (locked) | 238 | 70 | 91 / 88 / 59 | 17 other native speakers | evaluation 2 |
| v3 (locked) | 236 | 67 | 88 / 98 / 50 | 17 further native speakers | evaluation 3 |

The author wrote the test from scenario cards; a second native speaker labelled 60 items independently (intent κ = 0.916, route κ = 1.000). For the natural set, collected after the systems were frozen, contributors wrote without AI tools a complete and an incomplete message per test scenario; a seeded draw of two writers per scenario group from 474 screened messages (34 writers) gave 247, 238 in families absent from train and dev. Two annotators labelled all (intent κ = 0.938, route κ = 0.913; same slots on 65% of 91 shared CLARIFY items, same instant on 72% of 85 shared ACT items); the author adjudicated the 66 disagreements alone. v3 is the other 17 writers' 236 messages, labelled after its systems were frozen; both annotators adjudicated 20 jointly (route κ = 0.959, same slots 99%).


## 3.3 Pipeline and systems


Shared pipeline: message → normalizer → intent recognizer (the only part that differs) → extractor → policy → route. The normalizer unifies alef, ya and taa marbuta variants, strips diacritics and elongation and maps Arabic-Indic digits; a pattern-based extractor parses the supported time expressions; the reminder text is what remains once request framing, times and fillers are removed. The policy sends an unsupported intent to ABSTAIN, a supported one with all slots to ACT, otherwise to CLARIFY, so STS differences come from intent. Intent is isolated because out-of-scope detection is hard [6]; lexical models fit laptop CPUs.


Systems compared. S1: hand-written trigger and out-of-scope lexicons, no training data. S2: word TF-IDF with class-balanced logistic regression [4]; S3: S2 plus character 2-4-grams for spelling variation; both fitted on train, with C and n-grams chosen by dev intent macro-F1 (S2: C = 0.3; S3: C = 0.1). Oracle: gold intents, the pipeline's ceiling. S1 v1.1 adds an extractor fix found in the test errors, so it is compared only on the new-writer sets. Extensions (Appendix A): other recognizers S4-S7, e.g. fine-tuned MARBERTv2 [2] (S6); S8: Jais-3-7B [12] tool calling; S9: S8 checked by the policy (S1 v1.1 intent where S8 made no valid call).


## 3.4 Changes from the proposal


Training data: the proposal planned manually written items only. Recruiting writers through Google Forms failed before the freeze: many invited speakers declined, and 19 of the first form's 21 responses shared one writing pattern (several were identical), pointing to AI-generated or copied answers; they were discarded (10 October). With no time to recruit again, the author's own requests were kept for an all-human test and 261 of 298 training items were AI-drafted (458 items, 34 test pairs; planned: 300, 20). Evaluation: screened contributor messages collected after the freeze became two more test sets. Systems: S1-S3 ran on a laptop CPU as proposed; the open-weight extensions S4-S9 ran on Kaggle T4 GPUs.


## 4. Evaluation and results


## 4.1 Evaluation data and metrics


Evaluation data. All three sets are native-written requests for the two tools covering all routes, incomplete variants and in-domain out-of-scope requests; 79% of natural messages concern project management (by design), so results speak mainly to work reminders. Test intervals span about ±0.09 STS, so the new-writer sets carry the conclusions. Training data is relevant (same tools, routes and contract) but inadequate in quality and coverage: 261 of 298 items are AI-drafted; 64% are code-switched against 20% of the test and 35% of the natural set; 88% of training reminders use «فكرني» and 5% «تذكير», against 50% and 42% in the test. Separation. Splits follow scenario families under a hash-locked manifest; vocabularies and hyper-parameters come from train/dev only; each set was scored once; natural and v3 systems were frozen before labelling. Limitations. The test is one writer's templated phrasing (52 sentence frames; 59 items share a frame with train) by the author of S1's lexicon; natural and v3 test new writers, not new scenarios. C was chosen on 40 mostly drafted dev items (grouped-CV retuning on train+dev: S2 0.546, S3 0.580 on natural).


Metrics. Besides STS: intent and route macro-F1; unsafe execution (a wrong or unneeded call) over all items and over calls; unnecessary clarification (Unn. clar.: CLARIFY on gold ACT); and pairs fully correct. Intervals are family-level bootstraps (10,000 resamples) [8]; paired tests permute families [7], Holm-corrected [9]. Pre-registered hypotheses: on the natural set, rules beat TF-IDF (N1), the extractor fix helps (N2) and the policy helps LLM tool calling (N3); on v3, human training data helps TF-IDF (H-a) and MARBERTv2 (H-b).


## 4.2 Results


Finding 1: rules beat drafted-data classifiers. On new writers (Table 3) rules lead TF-IDF by +0.206 [0.129, 0.287] STS (N1, Holm p = 0.0003; 13 of 17 writers; also per annotator, Table 8). The author's test agrees (Table 2; S2 − S1 = −0.167, S3 − S1 = −0.142; Holm p ≤ 0.005) but alone is inconclusive (frame-clustered Holm p 0.055-0.526, Table 7).


Table 2. Locked test (120 items, 34 pairs). Metrics as defined in §4.1.

| System | STS [95% CI] | Intent F1 | Route F1 | Unsafe (all) | Unsafe (ACT) | Unn. clar. | Pairs |
| oracle (gold intent) | 0.808 [0.725, 0.885] | 1.000 | 0.960 | 0.133 | 0.320 | 0.060 | 0.588 |
| S1 keyword rules | 0.767 [0.678, 0.847] | 0.936 | 0.897 | 0.133 | 0.320 | 0.060 | 0.588 |
| S2 TF-IDF + LR | 0.600 [0.513, 0.686] | 0.738 | 0.797 | 0.200 | 0.511 | 0.060 | 0.382 |
| S3 + char n-grams | 0.625 [0.538, 0.711] | 0.751 | 0.804 | 0.167 | 0.455 | 0.060 | 0.382 |

Table 3. Natural set (238 items, 17 new writers, 64 pairs). Last column: success by gold route.

| System | STS [95% CI] | Intent F1 | Route F1 | Unsafe (all) | Unsafe (ACT) | Unn. clar. | Pairs | ACT/CLAR/ ABST |
| oracle | 0.786 [0.715, 0.851] | 1.000 | 0.955 | 0.134 | 0.337 | 0.044 | 0.578 | 0.69/0.74/1.00 |
| S1 rules | 0.718 [0.648, 0.785] | 0.922 | 0.880 | 0.151 | 0.383 | 0.044 | 0.547 | 0.64/0.72/0.85 |
| S2 TF-IDF | 0.513 [0.444, 0.580] | 0.667 | 0.714 | 0.218 | 0.536 | 0.044 | 0.328 | 0.49/0.50/0.56 |
| S3 + char | 0.513 [0.447, 0.577] | 0.670 | 0.721 | 0.239 | 0.576 | 0.033 | 0.328 | 0.46/0.55/0.54 |

N2 was not confirmed (one item changed). N3 was (+0.102, Holm p = 0.001), but 56 of its 73 gains were Jais-3-7B's missing calls filled by S1 (Table 5). Safety. 19 of S1's 94 natural calls were harmful (wrong tool/time or should not run; test: 3 of 50), 17 more only in wording.


## 4.3 Training data and extractor errors


Table 4. Registered v3 test (236 items, 17 further writers). Human data: natural-set messages added to training.

| System | Training data | STS [95% CI] | Success ACT / CLAR / ABST |
| oracle (gold intent) | – | 0.669 [0.612, 0.727] | 0.62 / 0.54 / 1.00 |
| S1 keyword rules | none | 0.589 [0.526, 0.648] | 0.57 / 0.48 / 0.84 |
| S3 TF-IDF + char | drafted + author | 0.500 [0.429, 0.567] | 0.55 / 0.46 / 0.50 |
| S3 TF-IDF + char | + human data | 0.589 [0.529, 0.650] | 0.59 / 0.52 / 0.72 |
| S6 MARBERTv2 | drafted + author | 0.549 [0.488, 0.610] | 0.60 / 0.52 / 0.50 |
| S6 MARBERTv2 | + human data | 0.637 [0.580, 0.693] | 0.62 / 0.53 / 0.88 |

Finding 2: in-domain human training data closes the gap (Table 4). On v3, adding human-written messages raised S3 by +0.089 [0.037, 0.150] (H-a, Holm p = 0.002) and MARBERTv2 by +0.088 [0.049, 0.134] (H-b, Holm p = 0.0002), mostly on out-of-scope requests. Not just more data: about 215 human messages alone beat 338 drafted/author items (+0.089, p = 0.004, descriptive). Retrained S3 equals the rules (+0.000 [−0.054, 0.058]); retrained MARBERTv2 exceeds them (+0.048 [0.013, 0.091], estimate only). They share v3's scenario cards: the gain is from in-distribution human data, not authorship alone.


Finding 3: the extractor is the ceiling; fixes help all systems. With gold intents STS is 0.786 (natural) and 0.669 (v3): intents explain under half the rules' failures. Four fixes solve 28 of the oracle's 50 natural failures: emphasis words («ابعت الريليز نوت ضروري»), want-verb framing («عايز تفكرني»), vague content («حاجه مهمه») and «الخميس بعد الجاي» (Thursday after next). These post hoc fixes (core v1.2) lifted the oracle on unseen v3 to 0.720 (+0.051 [0.019, 0.086], p = 0.010; 14 fixed, 2 broken); all v3 systems gained 0.047-0.051 (Table 13).


## 5. Conclusion


We delivered a task contract and metric, a 458-item benchmark, a shared pipeline and locked tests by 34 other writers (O1-O4). Training data decided the comparison: classifiers trained on mostly AI-drafted data lost to rules by 0.206 STS; in-domain human data closed the gap. The extractor limits every system (O5: 0.786 with gold intents); post hoc fixes added 0.05 on v3.


## References


[1] Ross, H., Mahabaleshwarkar, A. S. and Suhara, Y. (2025). When2Call: When (not) to Call Tools. NAACL-HLT, 3391-3409.


[2] Abdul-Mageed, M., Elmadany, A. and Nagoudi, E. M. B. (2021). ARBERT & MARBERT: Deep bidirectional transformers for Arabic. ACL.


[3] Dettmers, T., Pagnoni, A., Holtzman, A. and Zettlemoyer, L. (2023). QLoRA: Efficient finetuning of quantized LLMs. NeurIPS.


[4] Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. JMLR 12, 2825-2830.


[5] Sengupta, N. et al. (2023). Jais and Jais-chat: Arabic-centric foundation and instruction-tuned open generative large language models. arXiv:2308.16149.


[6] Larson, S. et al. (2019). An evaluation dataset for intent classification and out-of-scope prediction. EMNLP.


[7] Berg-Kirkpatrick, T., Burkett, D. and Klein, D. (2012). An empirical investigation of statistical significance in NLP. EMNLP.


[8] Field, C. A. and Welsh, A. H. (2007). Bootstrapping clustered data. JRSS B 69(3).


[9] Holm, S. (1979). A simple sequentially rejective multiple test procedure. Scandinavian Journal of Statistics 6(2).


[10] Mosbach, M., Andriushchenko, M. and Klakow, D. (2021). On the stability of fine-tuning BERT. ICLR.


[11] Jais-2-8B-Chat, model card. Hugging Face: inception42/Jais-2-8B-Chat.


[12] Jais-3-7B and Jais-3-3.7B, model cards. Hugging Face: inception42/Jais-3-7B, inception42/Jais-3-3.7B.


[13] Nile-Chat-4B, model card. Hugging Face: MBZUAI-Paris/Nile-Chat-4B.


[14] Yang, A. et al. (2025). Qwen3 technical report. arXiv:2505.09388.


## Appendix A. Additional results


Table 5. Safe Task Success of the extension systems on the three locked sets, with the numeric format each GPU model ran in, or CPU (S4: primary setting, 6 examples with prior calibration; S5/S6 fitted after both evaluations, exploratory). ¹S9 uses core v1.1, whose fix was drawn from the test errors. v3: core v1.1; the GPU rows were frozen before labelling (amendments A5, A7), descriptive.

| System | Format | Locked test (120) | Natural (238) | v3 (236) |
| S1 keyword rules (reference) | CPU | 0.767 | 0.718 | 0.589 |
| S4 Jais 2 8B (6-shot) | nf4-fp32 | 0.758 | 0.618 | – |
| S4 Jais-3-7B (6-shot) | fp16 | 0.550 | 0.664 | – |
| S4 Jais-3-3.7B (6-shot) | fp32 | 0.742 | 0.744 | – |
| S4 Nile-Chat 4B (6-shot) | fp32 | 0.742 | 0.664 | – |
| S4 Qwen3 8B (6-shot) | fp16 | 0.700 | 0.718 | – |
| S5 multilingual-e5-small + LR | CPU | 0.683 | 0.483 | – |
| S6 MARBERTv2 fine-tuned (5 seeds) | CPU | 0.775 ± 0.019 | 0.614 ± 0.021 | – |
| S7 Jais 2 + QLoRA (3 seeds) | nf4-fp32 | 0.786 (0.758 / 0.800 / 0.800) | 0.730 (0.697 / 0.739 / 0.752) | – |
| S8 Jais-3-7B tool calling (3 seeds) | fp16 | 0.725 (0.758 / 0.725 / 0.692) | 0.599 | – |
| S9 S8 + deterministic policy (3 seeds) | fp16 | not run¹ | 0.702 | – |
| S4 Jais-3-7B 6-shot inline, normalized / raw text | fp16 | – | – | 0.581 / 0.597 |
| S8 Jais-3-7B greedy, with exemplars (1 run) | fp16 | – | – | 0.619 |
| S9 from that S8 | fp16 | – | – | 0.657 |

Systems S4-S7. They replace only the intent recognizer; the extractor and policy are shared. S4 prompts an LLM for the intent with 6 examples (Jais 2 8B [5, 11], Jais-3-7B, Jais-3-3.7B [12], Nile-Chat 4B [13], Qwen3 8B [14]). S5 is logistic regression on frozen multilingual-e5-small embeddings. S6 is fine-tuned MARBERTv2 [2] (five seeds, recipe after [10]). S7 is Jais 2 with QLoRA [3] (three seeds). No LLM recognizer beat the rules significantly on any set (Tables 5, 9).


Precision. Kaggle's T4 GPUs have no bf16, so the models ran in different formats: Jais 2 overflowed in fp16 and ran 4-bit, Jais-3-7B and Qwen3 ran in fp16, the smaller models in fp32. Differences between these models may therefore partly reflect precision. Jais-3-7B was released after the freeze but before the test was opened; Jais 2 stayed the pre-registered LLM. In its primary setting it declined 34 in-scope test requests; with the exemplars listed in the user message (a secondary analysis declared before the test was opened) it scored 0.733. As S8 it made no valid call on 11, 12 and 16 of the 120 test items.


Protocol v3 (docs/protocol_v3.md, amendments A1-A9). Registered on 6 November before any v3 system existed; systems and GPU outputs frozen before labelling; gold locked (manifest 750e7847) and scored once (results/v3/evaluation.json, evaluation_gpu.json). H-a and H-b also hold with writers as clusters and in all 17 leave-one-writer-out analyses; the exploratory natural-set analysis agreed (Table 6). GPU systems (descriptive): raw instead of normalized text did not change S4 (+0.017, p = 0.50; same intent on 230 of 236); the policy added +0.038 to greedy Jais 3 with exemplars (p = 0.09). Greedy S8 made no valid call on 8 of 236 items (sampled S8 on the natural set: 23-25 of 238 a seed).


Cost. On a laptop CPU, each final S2/S3 fit takes under 0.3 s, inference under 0.3 ms per item, and peak memory is 141 MB (models/freeze.json).


Out-of-scope requests separate the learned systems on new writers: S6, Jais 2 and Nile-Chat declined only about half of them (0.48, 0.48 and 0.53), against 0.85 for the rules. These requests sit inside the reminder domain (recurring, deleting or changing a reminder), where the rules' explicit out-of-scope verbs decide better.


Table 6. Training on human-written messages: STS (intent accuracy). Natural set: exploratory leave-one-writer-out (each held-out writer's scenarios removed; about 215 contributor messages per fold). v3: registered test on 236 unseen messages, trained per scenario block on the natural set minus that block's cards (H-a: S3, H-b: S6). S2/S3 use grouped-CV settings; S1 is core v1.1.

| Classifier, set | Drafted + author (338) | Contributors only (~215) | Both |
| S2, natural | 0.546 (0.731) | 0.660 (0.870) | 0.672 (0.878) |
| S3, natural | 0.580 (0.773) | 0.681 (0.882) | 0.697 (0.908) |
| S5, natural | 0.492 (0.685) | 0.613 (0.815) | 0.630 (0.836) |
| S1 rules, natural |  |  | 0.723 (0.933) |
| S3, v3 (registered) | 0.500 (0.775) | 0.589 (0.877) | 0.589 (0.894) |
| S6 MARBERTv2, v3 (5 seeds) | 0.549 (0.869) |  | 0.637 (0.951) |
| S1 rules, v3 |  |  | 0.589 (0.898) |

Table 7. Locked test: sensitivity of the rules-vs-classifier differences to the clustering unit (frames: items sharing a sentence frame).

| Clustering, comparison | Clusters | Difference | 95% CI | Holm p |
| family (pre-registered), S2 vs S1 | 71 | −0.167 | [−0.250, −0.084] | 0.002 |
| family (pre-registered), S3 vs S1 | 71 | −0.142 | [−0.222, −0.061] | 0.005 |
| frame A (purpose-clause cut), S2 vs S1 | 52 | −0.167 | [−0.307, −0.037] | 0.096 |
| frame A (purpose-clause cut), S3 vs S1 | 52 | −0.142 | [−0.273, −0.018] | 0.114 |
| frame B (content masked), S2 vs S1 | 54 | −0.167 | [−0.288, −0.050] | 0.055 |
| frame B (content masked), S3 vs S1 | 54 | −0.142 | [−0.252, −0.037] | 0.055 |
| family + frame A (connected), S2 vs S1 | 19 | −0.167 | [−0.289, 0.031] | 0.526 |
| family + frame A (connected), S3 vs S1 | 19 | −0.142 | [−0.244, 0.100] | 0.526 |

Table 8. Natural set: confirmed differences under alternative labels and scoring rules (unadjusted p).

| Labels / rule | n | N1 S1 − S2 | N3 S9 − S8 |
| locked gold (pre-registered) | 238 | +0.206 (p 0.0001) | +0.102 (p 0.0005) |
| author's labels before adjudication | 238 | +0.160 (p 0.0001) | +0.076 (p 0.0011) |
| second annotator's labels (1 invalid dropped) | 237 | +0.186 (p 0.0001) | +0.097 (p 0.0006) |
| locked gold, lenient reminder text | 238 | +0.206 (p 0.0001) | +0.098 (p 0.0006) |
| locked gold without uncovered expressions (14 dropped) | 224 | +0.219 (p 0.0001) | +0.112 (p 0.0003) |

Table 9. Post hoc: LLM intent recognizers minus S1 in STS, same pipeline and stored outputs, family-level paired test and bootstrap as pre-registered, Holm over the six comparisons of each set (scripts/posthoc_llm_vs_rules.py). S7: per-item mean over three seeds. Last column: unsafe execution rate on the natural set.

| Recognizer | Test: Δ [95% CI] | Holm p | Natural: Δ [95% CI] | Holm p | Unsafe |
| S4 Jais 2 8B | −0.008 [−0.043, 0.026] | 1.000 | −0.101 [−0.170, −0.032] | 0.040 | 0.206 |
| S4 Jais-3-7B | −0.217 [−0.294, −0.137] | 0.001 | −0.055 [−0.120, 0.012] | 0.638 | 0.113 |
| S4 Jais-3-3.7B | −0.025 [−0.056, 0.000] | 1.000 | +0.025 [−0.016, 0.070] | 1.000 | 0.134 |
| S4 Nile-Chat 4B | −0.025 [−0.076, 0.025] | 1.000 | −0.055 [−0.118, 0.009] | 0.638 | 0.189 |
| S4 Qwen3 8B | −0.067 [−0.116, −0.018] | 0.107 | +0.000 [−0.056, 0.059] | 1.000 | 0.160 |
| S7 Jais 2 + QLoRA | +0.019 [−0.013, 0.059] | 1.000 | +0.011 [−0.032, 0.057] | 1.000 | 0.148 |
| S1 keyword rules (reference) | STS 0.767 |  | STS 0.718 |  | 0.151 |

## Appendix B. Representative errors and failure types


Table 10. Real outputs of the shared pipeline with the gold intent (oracle). Arabic cells read right to left; the second line of a cell is the reminder text.

| Message | Gold | Pipeline output | Cause |
| عايز تذكير بكرة الساعة 5 العصر علشان احدث سجل المخاطر | ACTاحدث سجل المخاطر | ACTعايز علشان احدث سجل المخاطر | want-verb framing kept (test) |
| اعمللي تذكير لو ينفع | CLARIFY time, reminder text | CLARIFY time | filler read as content (test) |
| فكرني بكرة الساعة 8 الصبح بحاجة محتاجه ضروري | CLARIFY reminder text | ACT | vague content accepted (test) |
| فكرني اعدل الrisk register ضروري بكره الساعه 7 الصبح | ACTاعدل الrisk register | ACTاعدل الrisk register ضروري | emphasis word kept (new writer) |
| محتاج تفكرني اراجع الjira tickets بكره الساعه 10 الصبح قبل ميعاد ال sprint review | ACTاراجع الjira tickets قبل ميعاد ال sprint review | ACTمحتاج اراجع الjira tickets قبل ميعاد ال sprint review | second-person framing (new writer) |

Table 11. Locked test: failures by type (counts of 120 items; from results/error_analysis). S2's 20 failures beyond S1's are all intent errors.

| Failure type | Oracle | S1 | S2 | S3 |
| Wrong reminder text only | 13 | 13 | 12 | 11 |
| Wrong tool | 0 | 0 | 9 | 7 |
| Executed an incomplete request | 3 | 3 | 3 | 2 |
| Asked although complete | 3 | 3 | 3 | 3 |
| Asked for the wrong slots | 4 | 4 | 1 | 1 |
| Clarified with the wrong intent | 0 | 0 | 6 | 6 |
| Asked instead of declining | 0 | 5 | 7 | 5 |
| Declined an in-scope request | 0 | 0 | 7 | 10 |
| All failures | 23 | 28 | 48 | 45 |

Table 12. Locked test: intent precision / recall per class, and STS on code-switched items (test n = 24, natural n = 84; exploratory, small subgroups). S2 labelled the 62 gold reminders as 41 reminder, 15 alarm, 6 unsupported.

| System | Reminder P / R | Alarm P / R | Unsupported P / R | CS test | CS natural |
| oracle (gold intent) | 1.00 / 1.00 | 1.00 / 1.00 | 1.00 / 1.00 | 0.833 | 0.738 |
| S1 keyword rules | 0.95 / 1.00 | 0.95 / 1.00 | 1.00 / 0.75 | 0.750 | 0.667 |
| S2 TF-IDF + LR | 0.87 / 0.66 | 0.70 / 0.97 | 0.65 / 0.65 | 0.708 | 0.488 |
| S3 + char n-grams | 0.91 / 0.65 | 0.73 / 0.97 | 0.60 / 0.75 | 0.750 | 0.488 |

Table 13. Post hoc core v1.2 (src/athar/v2/core12.py, scripts/posthoc_core_v12.py): STS of each system with its frozen intents through core v1.1 and v1.2. The fixes were written from the natural-set oracle failures (in-sample there); v3 was not inspected meanwhile. Paired family-level permutation test and bootstrap, unadjusted p. One natural item breaks because its gold keeps «ضروري» inside the text, unlike the others; the two broken v3 items were not examined, to keep v3 unread. Locked test: no change (0.967 with core v1.1, whose fixes came from its errors).

| Set | System | Core v1.1 | Core v1.2 | Δ [95% CI] | p | Fixed / broken |
| natural (dev.) | oracle | 0.790 | 0.903 | +0.113 [0.061, 0.172] | < 0.001 | 28 / 1 |
| natural (dev.) | S1 rules | 0.723 | 0.836 | +0.113 [0.061, 0.172] | < 0.001 | 28 / 1 |
| v3 (held-out) | oracle | 0.669 | 0.720 | +0.051 [0.019, 0.086] | 0.010 | 14 / 2 |
| v3 (held-out) | S1 rules | 0.589 | 0.640 | +0.051 [0.019, 0.086] | 0.010 | 14 / 2 |
| v3 (held-out) | S3 + human | 0.589 | 0.640 | +0.051 [0.019, 0.086] | 0.010 | 14 / 2 |
| v3 (held-out) | S6 + human (5 seeds) | 0.637 | 0.688 | +0.051 [0.019, 0.086] | 0.010 | 14 / 2 |
| v3 (held-out) | S3 drafted | 0.500 | 0.547 | +0.047 [0.015, 0.082] | 0.018 | 13 / 2 |
| v3 (held-out) | S6 drafted (5 seeds) | 0.549 | 0.599 | +0.050 [0.019, 0.085] | 0.010 | 14 / 2 |

## Appendix C. Reproducing the results


All commands run from the repository root on a CPU; T2 stands for Table 2. The five analysis scripts (T2-T11) rebuild their tables from the stored outputs and reproduce the committed files exactly (Table 4 is read from results/v3/evaluation.json; Table 8 also needs the private labelling sheets in data/raw/private/, which are not shared); the encoder commands retrain S5/S6 (slow on CPU). The GPU systems (S4, S7, S8) were run, and S7 was trained, with notebooks/kaggle_s4_llm.ipynb.


uv sync


uv run pytest                                               # tests


uv run python scripts/error_analysis.py --out results/error_analysis   # T2, T11


uv run python scripts/natural_tables.py                     # T3


uv run python scripts/posthoc_analysis.py --out results/posthoc        # T7


uv run python scripts/posthoc_llm_vs_rules.py               # T9


uv run --with openpyxl python scripts/natural_sensitivity.py           # T8


uv run --extra neural python scripts/posthoc_encoders_human.py encoders # T5


uv run --extra neural python scripts/posthoc_encoders_human.py human   # T6


uv run python scripts/posthoc_core_v12.py   # T13 (needs the private v3 texts)


uv run --extra ui streamlit run app/streamlit_app.py        # demo


Table 14. Pretrained models used (linked), with the pinned Hugging Face revisions.

| System | Hugging Face model | Revision |
| S4, S7 | inception42/Jais-2-8B-Chat | da0e1639cd92b508b24120f7f77f5270a8465dc4 |
| S4, S8, S9 | inception42/Jais-3-7B | 94d69f49b4af702d42fcee122cc06e493a6fe7e1 |
| S4 | inception42/Jais-3-3.7B | 3d5754d5db87cd3a59f0f5b09c7ac39b351ff5ed |
| S4 | MBZUAI-Paris/Nile-Chat-4B | 2a365667659bbd6786c2760d868e37c506d76ce3 |
| S4 | Qwen/Qwen3-8B | b968826d9c46dd6066d109eabc6255188de91218 |
| S5 | intfloat/multilingual-e5-small | 614241f622f53c4eeff9890bdc4f31cfecc418b3 |
| S6 | UBC-NLP/MARBERTv2 | fe88db9db8ccdb0c4e1627495f405c44a5f89066 |

## Appendix D. Use of generative AI


The 300 AI-drafted training and development requests were generated with an AI assistant and are marked source: assistant_draft; none is in either evaluation set. The code, tests and documentation were developed with Claude Code as a programming assistant. During adjudication of the natural set, the assistant checked the author's final labels against the written guide and flagged 24 that broke a rule; the author approved each correction. The core v1.2 fixes were written with the assistant from the natural-set oracle failures, without reading the v3 texts; the v3 scores were computed once and only in aggregate. No message was labelled from scratch by an AI system. The author designed the study, wrote the test requests and their labels, recruited the contributors, ran the experiments and is responsible for every claim in this report.
