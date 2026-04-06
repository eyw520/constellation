from typing import Literal

from pydantic import BaseModel, Field


class StdioTransport(BaseModel):
    type: Literal["stdio"] = "stdio"
    command: str = Field(..., description="Command to execute")
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)

    model_config = {"frozen": True}


class MCPServerConfig(BaseModel):
    name: str = Field(..., description="Unique server identifier")
    transport: StdioTransport
    timeout_seconds: float = Field(default=30.0, ge=1.0, le=300.0)

    model_config = {"frozen": True}


class MCPHandler(BaseModel):
    type: Literal["mcp"] = "mcp"
    server: str
    tool: str
    input_mapping: dict[str, str] | None = None
    output_mapping: dict[str, str] | None = None

    model_config = {"frozen": True}
