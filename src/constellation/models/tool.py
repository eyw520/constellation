from typing import Any

from pydantic import BaseModel


class ToolConfig(BaseModel):
    type: str
    name: str | None = None
    description: str | None = None
    input_schema: dict[str, Any] | None = None
    handler: dict[str, Any] | None = None
    config: dict[str, Any] = {}

    model_config = {"frozen": True}
