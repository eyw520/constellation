import logging

from constellation.engines.async_engine import AsyncEngine
from constellation.engines.registry import EngineRegistry
from constellation.engines.sync_engine import SyncEngine
from constellation.models.config import AgentConfig
from constellation.models.llm import LLMConfig
from constellation.services.mcp.manager import MCPServerManager
from constellation.tools.factory import ToolFactory
from constellation.tools.registry import ToolRegistry


logger = logging.getLogger(__name__)

DEFAULT_INITIATION_SIGNAL = "[[sig: Conversation begins. Greet the user.]]"


class Agent:
    def __init__(self, config: AgentConfig):
        self.config = config
        self._sync_engines: list[SyncEngine] = EngineRegistry.create_sync_engines(config.engines)
        self._async_engines: list[AsyncEngine] = EngineRegistry.create_async_engines(config.engines)
        self._mcp_manager: MCPServerManager | None = (
            MCPServerManager(config.mcp_servers) if config.mcp_servers else None
        )
        self._tool_registry: ToolRegistry | None = None

    async def start(self) -> None:
        if self._mcp_manager:
            await self._mcp_manager.start()

        self._tool_registry = ToolFactory.create_tool_registry(
            self.config.tools,
            mcp_manager=self._mcp_manager,
        )

        num_sync = len(self._sync_engines)
        num_async = len(self._async_engines)
        num_tools = len(self._tool_registry.get_tools())
        logger.info(f"Agent started with {num_sync} sync, {num_async} async engines, {num_tools} tools")

    async def stop(self) -> None:
        if self._mcp_manager:
            await self._mcp_manager.stop()

    def get_sync_engines(self) -> list[SyncEngine]:
        return self._sync_engines

    def get_async_engines(self) -> list[AsyncEngine]:
        return self._async_engines

    def get_tool_registry(self) -> ToolRegistry:
        if self._tool_registry is None:
            raise RuntimeError("Agent not started. Call await agent.start() first.")
        return self._tool_registry

    def get_mcp_manager(self) -> MCPServerManager | None:
        return self._mcp_manager

    def get_llm_config(self) -> LLMConfig:
        return self.config.llm

    def get_system_context(self) -> str:
        return self.config.prompt or ""

    def should_initiate(self) -> bool:
        return self.config.initiation.enabled

    def get_initiation_greeting(self) -> str | None:
        return self.config.initiation.greeting

    def get_initiation_signal(self) -> str:
        return DEFAULT_INITIATION_SIGNAL
