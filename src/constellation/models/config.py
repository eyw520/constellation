from pydantic import BaseModel, Field

from constellation.models.engine import EngineConfig
from constellation.models.llm import LLMConfig
from constellation.models.mcp import MCPServerConfig
from constellation.models.tool import ToolConfig


class SessionConfig(BaseModel):
    timeout_seconds: float = 300.0
    max_turns: int = 100

    model_config = {"frozen": True}


class InitiationConfig(BaseModel):
    enabled: bool = False
    greeting: str | None = None

    model_config = {"extra": "forbid"}


class AgentConfig(BaseModel):
    prompt: str | None = None
    initiation: InitiationConfig = Field(default_factory=InitiationConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    session: SessionConfig = Field(default_factory=SessionConfig)
    tools: list[ToolConfig] = Field(default_factory=list)
    engines: list[EngineConfig] = Field(default_factory=list)
    mcp_servers: list[MCPServerConfig] = Field(default_factory=list)

    model_config = {"extra": "forbid"}
