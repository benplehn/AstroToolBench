# Tokenizer inspection

I wanted to see what a tool-calling conversation actually looks like once it
goes through a model's chat template, and how many tokens the tool schemas cost.
This script loads only the tokenizer of an open model (no weights, no PyTorch,
no GPU) and renders a task or a recorded conversation.

```bash
python -m pip install ".[tokenizer]"
python scripts/inspect_template.py
```

Default model: `Qwen/Qwen2.5-0.5B-Instruct`, whose
[tokenizer config](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json)
has a chat template supporting tools, tool calls and tool results.

The first run downloads the tokenizer from Hugging Face; after that:

```bash
python scripts/inspect_template.py --local-files-only
```

It prints the tokenizer class, vocabulary size (number of tokens in the
vocabulary, not a prompt length), whether it's a fast tokenizer and has a chat
template, then the rendered text.

## A task prompt

Default task is `period-001`:

```bash
python scripts/inspect_template.py --task period-001 --local-files-only \
  --output results/templates/period-001.txt
```

Existing output files are refused. Messages are built with the same
`build_messages` as the agent, so no answers or tolerances.

Rendering:

```python
tokenizer.apply_chat_template(
    messages,
    tools=TOOLS_C,
    tokenize=False,
    add_generation_prompt=True,
)
```

`tokenize=False` gives text instead of token IDs, and `add_generation_prompt=True`
adds the start of the assistant turn. Qwen puts the tool schemas in the system
message between `<tools>` and `</tools>`, with instructions on how to format
calls.

## A recorded conversation

```bash
python scripts/inspect_template.py \
  --trace docs/examples/nvidia-nemotron-agent.json --local-files-only \
  --output results/templates/nemotron-agent-qwen.txt
```

This reads `messages` and `tools` from the trace. One catch: the API stores tool
arguments as JSON strings, but Transformers expects dicts. If you pass strings,
they end up quoted inside the call. So the script converts them (on a copy, the
trace isn't modified) and errors on invalid arguments.

In the output, Qwen wraps calls in `<tool_call>` and results in `<tool_response>`,
puts tool results inside a `user` message and drops the call IDs (they're still
in the original trace). If the conversation ends with a final answer, no
generation prefix is added; if it ends with a tool result, it is.

See [Transformers tool use](https://huggingface.co/docs/transformers/v4.57.1/chat_extras).

## B vs C token cost

```bash
python scripts/inspect_template.py \
  --trace docs/examples/nvidia-nemotron-agent.json --compare-tools \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-files-only \
  --report results/templates/token-comparison.json
```

This takes only the initial system/user messages from the trace (the C tool
calls can't be reused with B's argument format), and renders them with B
schemas, C schemas and no tools.

| Measure | How |
| --- | --- |
| Schema JSON tokens | tokens of the compact JSON tool list (`ensure_ascii=False`, separators `,` `:`) |
| Prompt tokens | tokens of the full rendered prompt |
| Tool overhead | prompt − same prompt without tools (includes Qwen's tool instructions) |
| Description cost | prompt − same prompt with every `description` removed |

The description cost includes the keys and punctuation too. And B vs C isn't
only descriptions: parameter names and constraints differ as well.

Counting uses `tokenizer.encode(text, add_special_tokens=False)` because the
template already added the special tokens; I check it matches
`apply_chat_template(..., tokenize=True)`.

### Result

The [report](examples/qwen-token-comparison.json) has the Qwen revision, library
versions, hashes and exact messages. Source: the seven-message Nemotron trace for
`multi-001`. The whole conversation rendered with Qwen is **1624 tokens**.

| Initial `multi-001` prompt | B | C |
| --- | ---: | ---: |
| Schema JSON tokens | 350 | 705 |
| Prompt tokens | 845 | 1220 |
| Tool overhead (vs 290 without tools) | 555 | 930 |
| Without descriptions | 720 | 813 |
| Description cost | 125 | 407 |

So C costs **375 more tokens** than B on the first request, and C's descriptions
alone are **407 tokens**. That's for all four tools, even the ones this task
doesn't need, and the schemas are resent at every request, so it adds up over a
conversation.

The 1624 tokens is one rendering of the finished conversation, not the sum of
the three API requests.

Docs: [chat templates](https://huggingface.co/docs/transformers/v4.57.1/chat_templating).

`--model` picks another model or local folder, `--revision` a branch/tag/commit
(use a commit hash to make it reproducible). Remote code is disabled.

Keep in mind this is Qwen's tokenizer: the trace came from Nemotron on NVIDIA's
API, and this tells nothing about the exact template or billed tokens there.
