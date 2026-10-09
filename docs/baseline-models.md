# Models for the first comparison

Four profiles in `astrotoolbench.models.catalog`: two big hosted models from
different companies, NVIDIA's model, and a small open model I can fine-tune
later.

| Profile | Model | Runs on |
| --- | --- | --- |
| `claude` | `anthropic/claude-sonnet-4.6` | OpenRouter, `OPENROUTER_API_KEY` |
| `gemini` | `google/gemini-2.5-pro` | OpenRouter, `OPENROUTER_API_KEY` |
| `nemotron` | `nvidia/nemotron-3.5-lightning-30b-a3b` | NVIDIA API, `NVIDIA_API_KEY` |
| `qwen` | `Qwen/Qwen3-4B-Instruct-2507` | local, Hugging Face |

Why these:
- **Claude Sonnet**: the strong hosted reference.
- **Gemini 2.5 Pro**: a second big model, to check the conclusions don't depend
  on one company. I used the stable ID rather than a preview alias.
- **Nemotron 3.5 Lightning**: the NVIDIA model I already used in the prototype.
- **Qwen3 4B**: small open model, the candidate for post-training later.

Model pages I checked on 2026-10-07:
[Claude](https://openrouter.ai/anthropic/claude-sonnet-4.6),
[Gemini](https://openrouter.ai/google/gemini-2.5-pro),
[Nemotron](https://huggingface.co/nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16),
[Qwen](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).

Qwen is pinned to revision `cdbee75f17c01a7cc42f958dc650907174af0554`. Hosted
APIs don't give a checkpoint hash, so results store both the profile and the
model ID returned by the API. Claude and Gemini go through OpenRouter, so their
latency includes the extra hop.

The profiles ignore `LLM_MODEL`, `LLM_BASE_URL` and `LLM_API_KEY` on purpose:
each one always uses the model, endpoint and key in the table (environment
variables win over `.env`). That way an old prototype setting can't silently
change the comparison.

```bash
python -m astrotoolbench.eval --list-models
```

Install `.[llm]` for the hosted models, `.[local]` for Qwen. Qwen is loaded once,
uses greedy decoding and counts tokens with its tokenizer. Pick `--device cpu`,
`cuda` or `mps`; `--local-files-only` blocks downloads. Qwen works for both
conditions, including tool calls in [baseline B](raw-tools-baseline.md).
