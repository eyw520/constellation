import asyncio
from collections.abc import Callable, Coroutine
import inspect
from typing import Any

from constellation.services.llm.types import LLMTool


ToolHandler = Callable[..., dict[str, Any] | Coroutine[Any, Any, dict[str, Any]]]


class ToolExecutionError(Exception):
    def __init__(self, tool_name: str, message: str, is_validation_error: bool = False):
        self.tool_name = tool_name
        self.is_validation_error = is_validation_error
        super().__init__(f"Tool '{tool_name}' error: {message}")


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, LLMTool] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, tool: LLMTool, handler: ToolHandler) -> None:
        self._tools[tool.name] = tool
        self._handlers[tool.name] = handler

    def get_tools(self) -> list[LLMTool]:
        return list(self._tools.values())

    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get_handler(self, tool_name: str) -> ToolHandler | None:
        return self._handlers.get(tool_name)

    def _validate_input(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        if tool_name not in self._handlers:
            raise ToolExecutionError(tool_name, f"Unknown tool: {tool_name}")

        tool = self._tools[tool_name]
        try:
            validated = tool.input_schema(**tool_input)
            return validated.model_dump()
        except Exception as e:
            raise ToolExecutionError(tool_name, f"Input validation failed: {e}", is_validation_error=True)

    def execute(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        validated_input = self._validate_input(tool_name, tool_input)

        try:
            result = self._handlers[tool_name](validated_input)
            if asyncio.iscoroutine(result):
                raise ToolExecutionError(
                    tool_name,
                    "Handler is async but execute() was called. Use execute_async() instead.",
                )
            return result  # type: ignore
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(tool_name, str(e))

    async def execute_async(self, tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
        validated_input = self._validate_input(tool_name, tool_input)

        try:
            handler = self._handlers[tool_name]
            if inspect.iscoroutinefunction(handler):
                return await handler(validated_input)
            else:
                return handler(validated_input)  # type: ignore
        except ToolExecutionError:
            raise
        except Exception as e:
            raise ToolExecutionError(tool_name, str(e))
