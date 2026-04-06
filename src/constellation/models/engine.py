from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag

from constellation.models.llm import LLMModel
from constellation.models.task_tag import TaskTag
from constellation.models.tool import ToolConfig


class GateConfig(BaseModel):
    prompt: str
    model: LLMModel | None = None
    num_turns: int = 3

    model_config = {"frozen": True}


class SyncEngineConfig(BaseModel):
    type: Literal["sync"] = "sync"
    name: str | None = None
    synthetic_tool_name: str | None = None
    gate: GateConfig | None = None
    system_prompt: str
    user_prompt_template: str
    num_turns: int = 5
    output_choices: list[str]
    task_tag_mapping: dict[str, TaskTag | None]
    model: LLMModel = LLMModel.GPT_4_1
    tools: list[ToolConfig] = Field(default_factory=list)

    model_config = {"frozen": True}


class AsyncEngineConfig(BaseModel):
    type: Literal["async"] = "async"
    name: str | None = None
    system_prompt: str
    user_prompt_template: str
    num_turns: int = 5
    model: LLMModel = LLMModel.GPT_4_1
    tools: list[ToolConfig] = Field(default_factory=list)

    model_config = {"frozen": True}


def _engine_discriminator(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("type", "sync"))
    return str(getattr(value, "type", "sync"))


EngineConfig = Annotated[
    Annotated[SyncEngineConfig, Tag("sync")] | Annotated[AsyncEngineConfig, Tag("async")],
    Discriminator(_engine_discriminator),
]
