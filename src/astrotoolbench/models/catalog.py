"""The four models selected for the first baseline comparison."""

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Literal

from .base import ModelBackend


@dataclass(frozen=True)
class ModelProfile:
    name: str
    model: str
    backend: Literal["api", "local"]
    base_url: str | None = None
    api_key_env: str | None = None
    revision: str | None = None


MODEL_PROFILES = (
    ModelProfile("claude", "anthropic/claude-sonnet-4.6", "api",
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ModelProfile("gemini", "google/gemini-2.5-pro", "api",
                 "https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    ModelProfile("nemotron", "nvidia/nemotron-3.5-lightning-30b-a3b", "api",
                 "https://integrate.api.nvidia.com/v1", "NVIDIA_API_KEY"),
    ModelProfile("qwen", "Qwen/Qwen3-4B-Instruct-2507", "local",
                 revision="cdbee75f17c01a7cc42f958dc650907174af0554"),
)


def get_profile(name: str) -> ModelProfile:
    for profile in MODEL_PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"Unknown model profile: {name}. Choose " + ", ".join(p.name for p in MODEL_PROFILES))


def create_backend(
    profile: ModelProfile, *, env_file: str | Path | None = None,
    max_tokens: int = 4096, timeout_s: float = 60.0,
    device: str = "cpu", local_files_only: bool = False,
) -> ModelBackend:
    if profile.backend == "local":
        from .local import HuggingFaceBackend
        return HuggingFaceBackend(
            profile.model, revision=profile.revision, max_tokens=max_tokens,
            device=device, local_files_only=local_files_only,
        )
    from atb.client import ClientSettings
    from dotenv import dotenv_values
    from .api import OpenAIBackend

    values = dict(dotenv_values(env_file)) if env_file is not None else {}
    values.update(os.environ)
    key = (values.get(profile.api_key_env) or "").strip()
    if not key:
        raise ValueError(f"Missing API key. Set {profile.api_key_env} for {profile.name}.")
    # Explicit profiles must not inherit another endpoint or a generic API key.
    settings = ClientSettings(profile.base_url, profile.model, key, timeout_s, max_retries=0)
    return OpenAIBackend(settings, temperature=0.0, max_tokens=max_tokens)
