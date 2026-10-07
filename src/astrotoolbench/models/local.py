"""Hugging Face text generation for the local no-tools baseline."""

from collections.abc import Sequence
import re

from .base import BackendError, Message, ModelBackend, ModelResponse, TokenUsage, ToolDefinition


class HuggingFaceBackend(ModelBackend):
    def __init__(
        self, model: str, *, revision: str, max_tokens: int = 4096,
        device: str = "cpu", local_files_only: bool = False,
    ):
        if not model.strip() or not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
            raise ValueError("A model ID and a full Hugging Face commit revision are required.")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer.")
        if device not in {"cpu", "cuda", "mps"}:
            raise ValueError("device must be cpu, cuda or mps.")
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError('Install local model dependencies with pip install ".[local]".') from error

        self.model = model
        self.revision = revision
        self.max_tokens = max_tokens
        self._torch = torch
        options = {"revision": revision, "local_files_only": local_files_only, "trust_remote_code": False}
        # Load once, outside the per-task generation timer used by the runner.
        self._tokenizer = AutoTokenizer.from_pretrained(model, **options)
        self._model = AutoModelForCausalLM.from_pretrained(
            model, torch_dtype=torch.float32 if device == "cpu" else torch.float16, **options,
        ).to(device)
        self._model.eval()

    def generate(
        self, messages: Sequence[Message], tools: Sequence[ToolDefinition] | None = None,
    ) -> ModelResponse:
        if tools or any(message.role == "tool" or message.tool_calls for message in messages):
            raise ValueError("The local text backend supports conversations without tools.")
        if not messages:
            raise ValueError("At least one message is required.")
        conversation = [{"role": message.role, "content": message.content} for message in messages]
        try:
            text = self._tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=True)
            inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
            input_tokens = inputs["input_ids"].shape[-1]
            with self._torch.inference_mode():
                output = self._model.generate(
                    **inputs, max_new_tokens=self.max_tokens, do_sample=False,
                    num_beams=1, num_return_sequences=1,
                )
            generated = output[0][input_tokens:]
            eos = self._model.generation_config.eos_token_id
            eos_ids = eos if isinstance(eos, list) else [eos]
            stopped = len(generated) > 0 and generated[-1].item() in eos_ids
            return ModelResponse(
                model=self.model, content=self._tokenizer.decode(generated, skip_special_tokens=True),
                finish_reason="length" if len(generated) >= self.max_tokens and not stopped else "stop",
                usage=TokenUsage(input_tokens=input_tokens, output_tokens=len(generated)),
            )
        except (RuntimeError, ValueError, TypeError) as error:
            raise BackendError("Local model generation failed.") from error
