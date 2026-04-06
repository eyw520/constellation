import itertools
import json
import logging
from typing import Any

from pydantic import TypeAdapter

from constellation.models.mcp import MCPServerConfig
from constellation.services.mcp.transport import MCPTransportBase, MCPTransportError, StdioMCPTransport
from constellation.services.mcp.types import (
    JSONRPCRequest,
    MCPInitializeResult,
    MCPToolCallResult,
    MCPToolInfo,
    MCPToolsListResult,
)


logger = logging.getLogger(__name__)


class MCPClientError(Exception):
    pass


_initialize_result_adapter = TypeAdapter(MCPInitializeResult)
_tools_list_result_adapter = TypeAdapter(MCPToolsListResult)
_tool_call_result_adapter = TypeAdapter(MCPToolCallResult)

MCP_PROTOCOL_VERSION = "2024-11-05"


class MCPClient:
    def __init__(self, config: MCPServerConfig):
        self._config = config
        self._transport: MCPTransportBase | None = None
        self._request_id_counter = itertools.count(1)
        self._server_capabilities: MCPInitializeResult | None = None
        self._available_tools: dict[str, MCPToolInfo] = {}

    @property
    def name(self) -> str:
        return self._config.name

    @property
    def is_connected(self) -> bool:
        return self._transport is not None

    @property
    def available_tools(self) -> dict[str, MCPToolInfo]:
        return self._available_tools

    async def connect(self) -> None:
        if self._transport is not None:
            return

        self._transport = StdioMCPTransport(self._config.transport, self._config.timeout_seconds)

        try:
            await self._transport.connect()
            await self._initialize()
            await self._discover_tools()
        except MCPTransportError:
            await self.close()
            raise
        except Exception as e:
            await self.close()
            raise MCPClientError(f"Failed to initialize MCP client: {e}")

    async def _initialize(self) -> None:
        response = await self._send_request(
            "initialize",
            {
                "protocolVersion": MCP_PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "constellation", "version": "1.0.0"},
            },
        )

        if response.error:
            raise MCPClientError(f"Initialize failed: {response.error.message}")

        self._server_capabilities = _initialize_result_adapter.validate_python(response.result)
        await self._send_notification("notifications/initialized", {})

    async def _discover_tools(self) -> None:
        response = await self._send_request("tools/list", {})

        if response.error:
            raise MCPClientError(f"Failed to list tools: {response.error.message}")

        result = _tools_list_result_adapter.validate_python(response.result)
        self._available_tools = {tool.name: tool for tool in result.tools}

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if self._transport is None:
            raise MCPClientError("Client not connected")

        if tool_name not in self._available_tools:
            available = list(self._available_tools.keys())
            raise MCPClientError(f"Tool '{tool_name}' not found. Available: {available}")

        response = await self._send_request(
            "tools/call",
            {"name": tool_name, "arguments": arguments},
        )

        if response.error:
            raise MCPClientError(f"Tool call failed: {response.error.message}")

        result = _tool_call_result_adapter.validate_python(response.result)

        if result.isError:
            error_text = self._extract_text_content(result.content)
            raise MCPClientError(f"Tool returned error: {error_text}")

        return self._process_tool_result(result.content)

    def _extract_text_content(self, content: list[dict[str, Any]]) -> str:
        texts = [item.get("text", "") for item in content if item.get("type") == "text"]
        return "\n".join(texts) if texts else str(content)

    def _process_tool_result(self, content: list[dict[str, Any]]) -> dict[str, Any]:
        if len(content) == 1:
            item = content[0]
            if item.get("type") == "text":
                text = item.get("text", "")
                try:
                    return json.loads(text)
                except (json.JSONDecodeError, TypeError):
                    return {"result": text}
            return item
        return {"content": content}

    async def _send_request(self, method: str, params: dict[str, Any]) -> Any:
        if self._transport is None:
            raise MCPClientError("Client not connected")

        request = JSONRPCRequest(
            id=next(self._request_id_counter),
            method=method,
            params=params,
        )
        return await self._transport.send_request(request)

    async def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        if self._transport is None:
            raise MCPClientError("Client not connected")

        notification = JSONRPCRequest(method=method, params=params)
        await self._transport.send_request(notification)

    async def close(self) -> None:
        if self._transport:
            try:
                await self._transport.close()
            except Exception as e:
                logger.warning(f"Failed to close MCP transport for {self.name}: {e}")
            self._transport = None
            self._available_tools = {}
            self._server_capabilities = None

    async def __aenter__(self) -> "MCPClient":
        await self.connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
