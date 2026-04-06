from typing import Any

from pydantic import BaseModel


class JSONRPCError(BaseModel):
    code: int
    message: str
    data: Any | None = None


class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: int | None = None
    method: str
    params: dict[str, Any] = {}


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: int | None = None
    result: Any | None = None
    error: JSONRPCError | None = None


class MCPToolInfo(BaseModel):
    name: str
    description: str | None = None
    inputSchema: dict[str, Any] = {}


class MCPInitializeResult(BaseModel):
    protocolVersion: str
    capabilities: dict[str, Any] = {}
    serverInfo: dict[str, Any] = {}


class MCPToolsListResult(BaseModel):
    tools: list[MCPToolInfo]


class MCPToolCallResult(BaseModel):
    content: list[dict[str, Any]]
    isError: bool = False
