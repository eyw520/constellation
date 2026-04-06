from abc import ABC, abstractmethod
import asyncio
import json
import logging
import os

from pydantic import TypeAdapter

from constellation.models.mcp import StdioTransport
from constellation.services.mcp.types import JSONRPCRequest, JSONRPCResponse


logger = logging.getLogger(__name__)


class MCPTransportError(Exception):
    pass


class MCPTransportBase(ABC):
    @abstractmethod
    async def connect(self) -> None:
        pass

    @abstractmethod
    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        pass

    @abstractmethod
    async def close(self) -> None:
        pass


_response_adapter = TypeAdapter(JSONRPCResponse)


class StdioMCPTransport(MCPTransportBase):
    def __init__(self, config: StdioTransport, timeout: float = 30.0):
        self._config = config
        self._timeout = timeout
        self._process: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def connect(self) -> None:
        env = os.environ.copy()
        env.update(self._config.env)

        args = [self._config.command, *self._config.args]
        try:
            self._process = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
        except FileNotFoundError:
            raise MCPTransportError(f"Command not found: {self._config.command}")
        except Exception as e:
            raise MCPTransportError(f"Failed to start MCP server: {e}")

    async def send_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        if self._process is None or self._process.stdin is None or self._process.stdout is None:
            raise MCPTransportError("Transport not connected")

        async with self._lock:
            request_line = request.model_dump_json() + "\n"

            try:
                self._process.stdin.write(request_line.encode())
                await self._process.stdin.drain()
            except Exception as e:
                raise MCPTransportError(f"Failed to send request: {e}")

            try:
                response_line = await asyncio.wait_for(
                    self._process.stdout.readline(),
                    timeout=self._timeout,
                )
            except TimeoutError:
                raise MCPTransportError(f"Request timed out after {self._timeout}s")

            if not response_line:
                stderr_output = ""
                if self._process.stderr:
                    try:
                        stderr_output = (await self._process.stderr.read()).decode()
                    except Exception:
                        pass
                raise MCPTransportError(f"MCP server closed connection. Stderr: {stderr_output}")

            try:
                response_data = json.loads(response_line.decode())
                return _response_adapter.validate_python(response_data)
            except json.JSONDecodeError as e:
                raise MCPTransportError(f"Invalid JSON response: {e}")

    async def close(self) -> None:
        if self._process:
            try:
                self._process.terminate()
                await asyncio.wait_for(self._process.wait(), timeout=5.0)
            except TimeoutError:
                try:
                    self._process.kill()
                    await self._process.wait()
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"Error closing MCP transport: {e}")
                try:
                    self._process.kill()
                except Exception:
                    pass
            finally:
                self._process = None
