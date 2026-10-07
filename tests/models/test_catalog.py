"""Model selection and credentials stay explicit across experiments."""

from dataclasses import asdict
from unittest.mock import MagicMock

import pytest

from astrotoolbench.models.catalog import MODEL_PROFILES, create_backend, get_profile


def test_four_distinct_models_and_pinned_local_weights():
    assert len(MODEL_PROFILES) == len({profile.model for profile in MODEL_PROFILES}) == 4
    assert {profile.name for profile in MODEL_PROFILES} == {"claude", "gemini", "nemotron", "qwen"}
    assert len(get_profile("qwen").revision) == 40
    assert all("api_key" not in asdict(profile) for profile in MODEL_PROFILES)


def test_unknown_profile_is_not_an_implicit_model_fallback():
    with pytest.raises(ValueError, match="Unknown model profile"):
        get_profile("unlisted-model")


def test_local_factory_passes_the_pinned_revision_without_api_settings(monkeypatch):
    import astrotoolbench.models.local as local
    constructor = MagicMock()
    monkeypatch.setattr(local, "HuggingFaceBackend", constructor)
    create_backend(get_profile("qwen"), device="mps", local_files_only=True, max_tokens=128)
    constructor.assert_called_once_with(
        "Qwen/Qwen3-4B-Instruct-2507", revision=get_profile("qwen").revision,
        max_tokens=128, device="mps", local_files_only=True,
    )


@pytest.fixture
def api_constructor(monkeypatch):
    pytest.importorskip("openai")
    pytest.importorskip("dotenv")
    import astrotoolbench.models.api as api
    constructor = MagicMock()
    monkeypatch.setattr(api, "OpenAIBackend", constructor)
    for key in ("OPENROUTER_API_KEY", "NVIDIA_API_KEY", "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
    return constructor


@pytest.mark.parametrize("name,key_name", [("claude", "OPENROUTER_API_KEY"), ("gemini", "OPENROUTER_API_KEY"),
                                          ("nemotron", "NVIDIA_API_KEY")])
def test_profile_cannot_be_overridden_by_generic_provider_variables(api_constructor, monkeypatch, name, key_name):
    monkeypatch.setenv(key_name, "matching-key")
    monkeypatch.setenv("LLM_API_KEY", "another-provider-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://another-provider.example/v1")
    monkeypatch.setenv("LLM_MODEL", "another-model")
    profile = get_profile(name)
    create_backend(profile)
    settings = api_constructor.call_args.args[0]
    assert (settings.base_url, settings.model, settings.api_key) == (profile.base_url, profile.model, "matching-key")
    assert settings.max_retries == 0
    assert "matching-key" not in repr(settings)


def test_explicit_env_file_and_process_precedence(api_constructor, monkeypatch, tmp_path):
    file = tmp_path / ".env"
    file.write_text("NVIDIA_API_KEY=file-key\n", encoding="utf-8")
    monkeypatch.setenv("NVIDIA_API_KEY", "process-key")
    create_backend(get_profile("nemotron"), env_file=file, max_tokens=128, timeout_s=12)
    settings = api_constructor.call_args.args[0]
    assert settings.api_key == "process-key"
    assert settings.timeout_s == 12
    assert api_constructor.call_args.kwargs == {"temperature": 0.0, "max_tokens": 128}


def test_missing_key_does_not_borrow_generic_or_other_provider_key(api_constructor, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "unrelated-key")
    monkeypatch.setenv("NVIDIA_API_KEY", "unrelated-key")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        create_backend(get_profile("claude"))
    api_constructor.assert_not_called()
