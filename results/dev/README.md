# Development results (dev split and native Egyptian set)

Outputs of the development runs, as downloaded from Kaggle or written locally. Nothing here is a test
result: the locked test outputs are in `results/test_results.json` and `results/s4/`.

Which run produced each folder, with its notebook version and flags: `docs/report_supplement.md`, appendix D
("How each result was produced"). Each folder's `environment.json` records the GPUs, library versions,
model revisions, manifest hash and notebook version of that run.

Not copied here:
- `egy_test.jsonl`, the external `Alhasan/egyptian-nlu` test file (CC BY 4.0). The notebook and `athar-ood`
  download it at a pinned revision.
- The S7 LoRA adapters of run 6b (3 × 177 MB). Their SHA-256 are in `docs/jais3.md`, section 13.
