# Test fixtures

| File | Source | Licence | SHA-256 |
|---|---|---|---|
| `jais3_chat_template.jinja` | `chat_template.jinja` of [`inception42/Jais-3-7B`](https://huggingface.co/inception42/Jais-3-7B) at revision `94d69f49b4af702d42fcee122cc06e493a6fe7e1` (identical in `inception42/Jais-3-3.7B` at `3d5754d5db87cd3a59f0f5b09c7ac39b351ff5ed`; first downloaded on 2026-10-15 at `a9097a3e2318` / `e664b9e1c555`, which the Hub's history rewrite of 2026-10-16 replaced with byte-identical content) | Apache-2.0 | `38964d6fc5e411718d88f7a5c1b3664b3d97208c6ac93160b6c9eadc63c14057` |

The template is kept byte-for-byte so the offline tests render prompts exactly as Jais 3 does
(thinking blocks, tool definitions, tool-call formats) without downloading the model.
`tests/test_llm.py` checks its hash; update it only together with the pinned revision.
