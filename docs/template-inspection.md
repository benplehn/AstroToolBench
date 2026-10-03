# Chat tokenizer inspection

The inspection command loads the tokenizer assets for an open model and renders
a benchmark prompt or a recorded agent conversation using its chat template.
It does not load model weights or generate responses. The optional dependency
requires neither PyTorch nor a GPU. It includes Transformers for the tokenizer
and Jinja2 for applying the chat template.

```bash
python -m pip install ".[tokenizer]"
python scripts/inspect_template.py
```

The default is `Qwen/Qwen2.5-0.5B-Instruct`. Its
[published tokenizer configuration](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct/blob/main/tokenizer_config.json)
includes a chat template with tool definitions, assistant tool calls and tool
results.

The first run downloads tokenizer files from Hugging Face and caches them
locally. Subsequent runs can use only the cached assets:

```bash
python scripts/inspect_template.py --local-files-only
```

The output identifies the tokenizer class, vocabulary size including added
tokens, fast-tokenizer availability and chat-template availability, then prints
the complete rendered conversation. Vocabulary size is the number of available
token entries, not a prompt token count or the
number of model parameters.

## Render a benchmark prompt

The default task is `period-001`. Select another task or save the raw rendered
text, without the terminal metadata:

```bash
python scripts/inspect_template.py --task period-001 --local-files-only \
  --output results/templates/period-001.txt
```

Existing output files are rejected. The command uses the same `build_messages`
function as the agent: system instructions and the task prompt are included;
reference answers, tolerances and other evaluation fields are excluded.

The rendering operation is:

```python
tokenizer.apply_chat_template(
    messages,
    tools=TOOLS_C,
    tokenize=False,
    add_generation_prompt=True,
)
```

`tokenize=False` returns text rather than numerical token IDs.
`add_generation_prompt=True` appends the opening of an assistant message, where
the model would begin its response. The Qwen template places tool schemas inside
the system message, between `<tools>` and `</tools>`, and adds instructions for
representing future tool requests. This operation describes available tools;
it does not execute tools or generate new responses.

## Inspect a recorded conversation

```bash
python scripts/inspect_template.py \
  --trace docs/examples/nvidia-nemotron-agent.json --local-files-only \
  --output results/templates/nemotron-agent-qwen.txt
```

The trace mode reads the recorded `messages` and `tools`, without loading the
current benchmark task or executing calls. In a copy of the messages, it converts
API JSON argument strings into dictionaries, as required by Transformers. Passing
the original strings directly would encode arguments as quoted JSON strings
inside the call instead of objects. Invalid or non-object arguments are rejected
with an explicit error; the source trace is never changed.

The default example contains two successive calls and their results. Qwen wraps
assistant requests in `<tool_call>` and tool outputs in `<tool_response>` blocks.
This template places tool responses inside a `user` message and omits call IDs
from the rendered text; the original API trace retains the `tool` roles and IDs.
The completed final answer is included without adding another assistant prefix.
Histories ending in a tool result receive a generation prefix for the next turn.

See [Transformers tool use](https://huggingface.co/docs/transformers/v4.57.1/chat_extras)
for the dictionary argument format and history serialization conventions.

## Compare token costs

```bash
python scripts/inspect_template.py \
  --trace docs/examples/nvidia-nemotron-agent.json --compare-tools \
  --revision 7ae557604adf67be50417f59c2c2f167def9a775 --local-files-only \
  --report results/templates/token-comparison.json
```

The comparison uses only the initial system/user messages of the trace, with an
assistant generation prefix. Actual C calls are excluded from the B comparison
because they use different argument conventions. Each condition uses the same
messages and tokenizer, varying only the tool schemas.

| Measurement | Definition |
| --- | --- |
| Schema JSON tokens | Tokens in the compact JSON tool list (`ensure_ascii=False`, separators `,` and `:`). |
| Prompt tokens | Tokens in the complete initial prompt as rendered by the model's template. |
| Tool overhead | Prompt tokens minus the same prompt without tools; includes the template's tool instructions. |
| Description field cost | Full prompt minus the same prompt after removing every `description` field, including function and parameter descriptions. |

The description measurement includes keys and JSON punctuation, as well as
description text and token-boundary effects. The difference between B and C also
includes parameter names, inputs and schema constraints; it must not be attributed
entirely to descriptions.

Token counting uses `tokenizer.encode(text, add_special_tokens=False)` because
the rendered text already includes template control tokens. A check compares
these IDs to `apply_chat_template(..., tokenize=True)` for the same conversation.
Token counts are not character counts, and independently tokenized fragments
need not add up to the token count of their concatenation.

### Recorded measurement

The [inspection report](examples/qwen-token-comparison.json) records the pinned
Qwen revision, library versions, source/template/schema hashes and exact messages.
The source is the existing seven-message NVIDIA trace for `multi-001`; applying
the Qwen template to the complete conversation produces **1624 tokens**.

| Initial `multi-001` prompt measurement | B | C |
| --- | ---: | ---: |
| Compact schema JSON tokens | 350 | 705 |
| Complete prompt tokens | 845 | 1220 |
| Tool overhead versus the 290-token prompt without tools | 555 | 930 |
| Prompt tokens without description fields | 720 | 813 |
| Description field cost | 125 | 407 |

C adds **375 prompt tokens** relative to B on this initial request. Removing all
C description fields saves **407 tokens** with other schema fields unchanged.
These measurements describe the complete set of four tool definitions, including
tools not used by this particular task. Definitions are included in each model
request of the agent, so their input-token cost recurs as the conversation grows;
exact totals must be counted in each full request because token boundaries vary.

The 1624-token inspection is one rendering of the completed conversation. It is
not the sum of input/output usage from the original provider's three requests.

See Hugging Face's [chat template documentation](https://huggingface.co/docs/transformers/v4.57.1/chat_templating)
for message formatting and generation prompts.

Use `--model` to select another repository or local directory and `--revision`
to select a branch, tag or commit hash. For reproducible inspections, use a commit
hash rather than the moving default branch. Remote custom Python code is disabled.

This Qwen tokenizer describes Qwen's input representation. The source trace was
generated by NVIDIA, but this is a local reserialization, not evidence of the
exact template or billable token counts used by the hosted NVIDIA model.
