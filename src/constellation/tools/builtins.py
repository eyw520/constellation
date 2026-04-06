from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from constellation.services.llm.types import LLMTool


class GetCurrentTimeInput(BaseModel):
    timezone: str = Field(default="local", description="Timezone (currently only 'local' supported)")


class EndConversationInput(BaseModel):
    reason: str = Field(default="user_request", description="Reason for ending the conversation")


def get_current_time(input_data: dict[str, Any]) -> dict[str, Any]:
    now = datetime.now()
    return {
        "current_time": now.strftime("%Y-%m-%d %H:%M:%S"),
        "timezone": "local",
    }


def end_conversation(input_data: dict[str, Any]) -> dict[str, Any]:
    return {
        "end_conversation": True,
        "reason": input_data.get("reason", "user_request"),
    }


GET_CURRENT_TIME_TOOL = LLMTool(
    name="get_current_time",
    description="Get the current date and time",
    input_schema=GetCurrentTimeInput,
)

END_CONVERSATION_TOOL = LLMTool(
    name="end_conversation",
    description="End the current conversation gracefully",
    input_schema=EndConversationInput,
)

BUILTIN_TOOLS: dict[str, tuple[LLMTool, Any]] = {
    "get_current_time": (GET_CURRENT_TIME_TOOL, get_current_time),
    "end_conversation": (END_CONVERSATION_TOOL, end_conversation),
}
