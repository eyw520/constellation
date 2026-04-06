from enum import StrEnum

from pydantic import BaseModel


class LLMModel(StrEnum):
    GPT_4O = "gpt-4o-2024-11-20"
    GPT_4_1 = "gpt-4.1-2025-04-14"
    CLAUDE_OPUS_4_1 = "claude-opus-4-1-20250805"
    CLAUDE_OPUS_4 = "claude-opus-4-20250514"
    CLAUDE_SONNET_4 = "claude-sonnet-4-20250514"


class ModelProvider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


def get_provider(model: LLMModel) -> ModelProvider:
    if model.startswith("gpt") or model.startswith("o1") or model.startswith("o3"):
        return ModelProvider.OPENAI
    elif model.startswith("claude"):
        return ModelProvider.ANTHROPIC
    raise ValueError(f"Unknown model provider for: {model}")


def is_openai_model(model: LLMModel) -> bool:
    return get_provider(model) == ModelProvider.OPENAI


def is_anthropic_model(model: LLMModel) -> bool:
    return get_provider(model) == ModelProvider.ANTHROPIC


class LLMConfig(BaseModel):
    model: LLMModel = LLMModel.GPT_4_1
    temperature: float = 0.7
    max_tokens: int = 4096

    model_config = {"frozen": True}
