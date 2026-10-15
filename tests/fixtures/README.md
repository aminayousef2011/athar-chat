# Test fixtures

| File | Source | Licence | SHA-256 |
|---|---|---|---|
| `jais3_chat_template.jinja` | `chat_template.jinja` of [`inception42/Jais-3-7B`](https://huggingface.co/inception42/Jais-3-7B) at revision `a9097a3e2318aeb9a5840bfee0655469792f5528` (identical in `inception42/Jais-3-3.7B` at `e664b9e1c555f4d6684af844149694b78a862951`) | Apache-2.0 | `38964d6fc5e411718d88f7a5c1b3664b3d97208c6ac93160b6c9eadc63c14057` |

The template is kept byte-for-byte so the offline tests render prompts exactly as Jais 3 does
(thinking blocks, tool definitions, tool-call formats) without downloading the model.
`tests/test_llm.py` checks its hash; update it only together with the pinned revision.
