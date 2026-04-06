import logging
from typing import Any

from constellation.models.mcp import MCPServerConfig
from constellation.services.mcp.client import MCPClient, MCPClientError


logger = logging.getLogger(__name__)


class MCPServerManagerError(Exception):
    pass


class MCPServerManager:
    def __init__(self, configs: list[MCPServerConfig]):
        self._configs = {config.name: config for config in configs}
        self._clients: dict[str, MCPClient] = {}
        self._started = False

    @property
    def server_names(self) -> list[str]:
        return list(self._configs.keys())

    @property
    def connected_servers(self) -> list[str]:
        return list(self._clients.keys())

    async def start(self) -> None:
        if self._started:
            return

        for name, config in self._configs.items():
            try:
                client = MCPClient(config)
                await client.connect()
                self._clients[name] = client
                tool_count = len(client.available_tools)
                logger.info(f"Connected to MCP server '{name}' with {tool_count} tools")
            except Exception as e:
                logger.warning(f"Failed to connect to MCP server '{name}': {e}")

        self._started = True

    async def stop(self) -> None:
        for name, client in self._clients.items():
            try:
                await client.close()
                logger.debug(f"Disconnected from MCP server '{name}'")
            except Exception as e:
                logger.warning(f"Error closing MCP server '{name}': {e}")

        self._clients.clear()
        self._started = False

    def get_client(self, server_name: str) -> MCPClient:
        if server_name not in self._clients:
            if server_name in self._configs:
                raise MCPServerManagerError(f"MCP server '{server_name}' is configured but not connected")
            raise MCPServerManagerError(f"MCP server '{server_name}' not found")
        return self._clients[server_name]

    def get_tool_info(self, server_name: str, tool_name: str) -> dict[str, Any]:
        client = self.get_client(server_name)
        if tool_name not in client.available_tools:
            available = list(client.available_tools.keys())
            raise MCPServerManagerError(f"Tool '{tool_name}' not found on '{server_name}'. Available: {available}")
        tool_info = client.available_tools[tool_name]
        return {
            "name": tool_info.name,
            "description": tool_info.description,
            "inputSchema": tool_info.inputSchema,
        }

    def list_all_tools(self) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for name, client in self._clients.items():
            result[name] = [
                {"name": t.name, "description": t.description, "inputSchema": t.inputSchema}
                for t in client.available_tools.values()
            ]
        return result

    async def call_tool(self, server_name: str, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        client = self.get_client(server_name)
        try:
            return await client.call_tool(tool_name, arguments)
        except MCPClientError as e:
            raise MCPServerManagerError(f"Tool call failed on '{server_name}': {e}")

    async def __aenter__(self) -> "MCPServerManager":
        await self.start()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.stop()
