from collections.abc import Sequence
import logging
from typing import TYPE_CHECKING, Any

from pydantic import create_model

from constellation.models.mcp import MCPHandler
from constellation.models.tool import ToolConfig
from constellation.services.llm.types import LLMTool
from constellation.tools.builtins import BUILTIN_TOOLS
from constellation.tools.registry import ToolRegistry


if TYPE_CHECKING:
    from constellation.services.mcp.manager import MCPServerManager

logger = logging.getLogger(__name__)


def create_input_model_from_schema(tool_type: str, input_schema: dict[str, Any]) -> type:
    model_name = f"{tool_type.title().replace('_', '')}Input"
    properties = input_schema.get("properties", {})
    required = set(input_schema.get("required", []))

    field_definitions: dict[str, Any] = {}
    for name, prop in properties.items():
        field_type = str
        if prop.get("type") == "integer":
            field_type = int
        elif prop.get("type") == "number":
            field_type = float
        elif prop.get("type") == "boolean":
            field_type = bool
        elif prop.get("type") == "array":
            field_type = list
        elif prop.get("type") == "object":
            field_type = dict

        if name in required:
            field_definitions[name] = (field_type, ...)
        else:
            field_definitions[name] = (field_type | None, None)

    return create_model(model_name, **field_definitions)


class ToolFactory:
    @staticmethod
    def create_tool_registry(
        tool_configs: Sequence[ToolConfig] | None,
        mcp_manager: "MCPServerManager | None" = None,
    ) -> ToolRegistry:
        registry = ToolRegistry()

        if not tool_configs:
            return registry

        for config in tool_configs:
            if config.type == "builtin":
                if not config.name:
                    raise ValueError("Builtin tool config missing 'name' field")
                if config.name not in BUILTIN_TOOLS:
                    available = list(BUILTIN_TOOLS.keys())
                    raise ValueError(f"Unknown builtin tool: {config.name}. Available: {available}")
                tool, handler = BUILTIN_TOOLS[config.name]
                registry.register(tool, handler)

            elif config.type in BUILTIN_TOOLS:
                tool, handler = BUILTIN_TOOLS[config.type]
                registry.register(tool, handler)

            elif config.handler is not None and ToolFactory._is_mcp_handler(config.handler):
                if mcp_manager is None:
                    raise ValueError(f"Tool '{config.type}' has MCP handler but no MCPServerManager")
                tool, handler = ToolFactory._create_mcp_tool(config, mcp_manager)
                registry.register(tool, handler)

            else:
                available = list(BUILTIN_TOOLS.keys())
                raise ValueError(f"Unknown tool type: {config.type}. Available: {available}")

        return registry

    @staticmethod
    def _is_mcp_handler(handler: dict[str, Any]) -> bool:
        return handler.get("type") == "mcp"

    @staticmethod
    def _create_mcp_tool(
        config: ToolConfig,
        mcp_manager: "MCPServerManager",
    ) -> tuple[LLMTool, Any]:
        handler_data = config.handler
        if not handler_data:
            raise ValueError(f"Tool '{config.type}' missing handler")

        handler = MCPHandler(**handler_data)
        mcp_tool_info = mcp_manager.get_tool_info(handler.server, handler.tool)

        input_schema = config.input_schema or mcp_tool_info.get("inputSchema", {})
        description = config.description or mcp_tool_info.get("description", "")

        input_model = create_input_model_from_schema(config.type, input_schema)

        tool = LLMTool(
            name=config.type,
            description=description,
            input_schema=input_model,
        )

        server_name = handler.server
        tool_name = handler.tool
        input_mapping = handler.input_mapping
        output_mapping = handler.output_mapping

        async def mcp_handler(tool_input: dict[str, Any]) -> dict[str, Any]:
            if input_mapping:
                mapped_input = {mcp_key: tool_input.get(input_key) for mcp_key, input_key in input_mapping.items()}
            else:
                mapped_input = tool_input

            result = await mcp_manager.call_tool(server_name, tool_name, mapped_input)

            if output_mapping:
                return {output_key: result.get(mcp_key) for output_key, mcp_key in output_mapping.items()}

            return result

        logger.debug(f"Created MCP tool '{config.type}' -> {handler.server}:{handler.tool}")
        return tool, mcp_handler
