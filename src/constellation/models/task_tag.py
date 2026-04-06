from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, Tag


class TaskTagResult(BaseModel):
    type: Literal["result"] = "result"
    task_name: str
    result: str
    engine_name: str | None = None
    synthetic_tool_name: str | None = None

    model_config = {"frozen": True}


class TaskTagInvocation(BaseModel):
    type: Literal["invocation"] = "invocation"
    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class TaskTagDisable(BaseModel):
    type: Literal["disable"] = "disable"
    engine_name: str | None = None

    model_config = {"frozen": True}


TaskTag = Annotated[
    Annotated[TaskTagResult, Tag("result")]
    | Annotated[TaskTagInvocation, Tag("invocation")]
    | Annotated[TaskTagDisable, Tag("disable")],
    Discriminator("type"),
]
