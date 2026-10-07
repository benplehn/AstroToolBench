# Models for the first comparison

Four profiles are defined in `astrotoolbench.models.catalog`. They cover two
hosted model developers, NVIDIA, and a small checkpoint we can later train.

| Profile | Model | How it runs |
| --- | --- | --- |
| `claude` | `anthropic/claude-sonnet-4.6` | OpenRouter, `OPENROUTER_API_KEY` |
| `gemini` | `google/gemini-2.5-pro` | OpenRouter, `OPENROUTER_API_KEY` |
| `nemotron` | `nvidia/nemotron-3.5-lightning-30b-a3b` | NVIDIA API, `NVIDIA_API_KEY` |
| `qwen` | `Qwen/Qwen3-4B-Instruct-2507` | Local Hugging Face backend |

Claude is the large hosted reference. Gemini gives us a comparison from a
second developer; the stable 2.5 Pro ID avoids starting this baseline on a
preview alias. Nemotron keeps the NVIDIA model already used by the prototype.
Qwen 4B is the smaller open-weight model for later post-training work.

Model pages checked on 2026-10-07 for this selection:
[Claude](https://openrouter.ai/anthropic/claude-sonnet-4.6),
[Gemini](https://openrouter.ai/google/gemini-2.5-pro),
[Nemotron](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
[Qwen](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
These are selection notes, not benchmark results or a claim that these are the
latest models. API access still depends on the account and provider.

The Qwen weights and tokenizer use revision
`cdbee75f17c01a7cc42f958dc650907174af0554`. Hosted APIs do not expose an equivalent
checkpoint hash. Results record both the requested profile and the model ID
returned by the API. Claude and Gemini go through a gateway, so their timing
includes OpenRouter routing and network overhead.

Profiles deliberately ignore generic `LLM_MODEL`, `LLM_BASE_URL` and
`LLM_API_KEY` overrides. They use the exact model, endpoint and provider key
listed above, with process variables taking precedence over an explicit `.env`
file. This keeps a leftover prototype setting from changing the comparison.

```bash
python -m astrotoolbench.eval --list-models
```

Install `.[llm]` for the three hosted profiles, or `.[local]` for Qwen. The local
backend loads the pinned checkpoint once, uses greedy decoding and counts tokens
with its tokenizer. Select `--device cpu`, `cuda` or `mps` explicitly;
`--local-files-only` disables downloads. The local backend currently supports
text conversations without tools, which is what baseline A needs.
