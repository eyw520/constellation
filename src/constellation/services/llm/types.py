from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class LLMMessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class SystemMessage(BaseModel):
    text: str

    def to_openai(self) -> dict[str, Any]:
        return {"role": "system", "content": self.text}

    def to_anthropic(self) -> dict[str, Any]:
        return {"type": "text", "text": self.text, "cache_control": {"type": "ephemeral"}}


class TextMessage(BaseModel):
    role: LLMMessageRole
    content: str

    def to_openai(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": self.content}

    def to_anthropic(self) -> dict[str, Any]:
        return {"role": self.role.value, "content": self.content}


class ToolUseMessage(BaseModel):
    id: str
    name: str
    input: dict[str, Any]


class ToolResultMessage(BaseModel):
    tool_use_id: str
    content: str


class AssistantMessageWithTools(BaseModel):
    tool_calls: list[ToolUseMessage]

    def to_openai(self) -> dict[str, Any]:
        return {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": str(tc.input)},
                }
                for tc in self.tool_calls
            ],
        }

    def to_anthropic(self) -> dict[str, Any]:
        content = [
            {"type": "tool_use", "id": tc.id, "name": tc.name, "input": tc.input}
            for tc in self.tool_calls
        ]
        return {"role": "assistant", "content": content}


class UserMessageWithToolResults(BaseModel):
    tool_results: list[ToolResultMessage]

    def to_openai(self) -> list[dict[str, Any]]:
        return [
            {"role": "tool", "tool_call_id": tr.tool_use_id, "content": tr.content} for tr in self.tool_results
        ]

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": tr.tool_use_id, "content": tr.content}
                for tr in self.tool_results
            ],
        }


LLMMessage = SystemMessage | TextMessage | AssistantMessageWithTools | UserMessageWithToolResults
ConversationMessage = TextMessage | AssistantMessageWithTools | UserMessageWithToolResults


class LLMTool(BaseModel):
    name: str
    description: str
    input_schema: type[BaseModel]

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema.model_json_schema(),
            },
        }

    def to_anthropic(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema.model_json_schema(),
        }


class ToolCallDict(BaseModel):
    id: str
    name: str
    input: dict[str, Any] | str


class ToolResultDict(BaseModel):
    tool_call_id: str
    content: str
