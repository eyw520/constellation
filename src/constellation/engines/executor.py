import asyncio
from threading import Thread
import time
from typing import TYPE_CHECKING

from constellation.engines.async_engine import AsyncEngine, EngineConversationTurn
from constellation.engines.sync_engine import SyncEngine
from constellation.logger import LOGGER
from constellation.models.task_tag import TaskTag, TaskTagDisable, TaskTagInvocation, TaskTagResult
from constellation.services.llm.types import (
    AssistantMessageWithTools,
    ToolResultMessage,
    ToolUseMessage,
    UserMessageWithToolResults,
)


if TYPE_CHECKING:
    from constellation.core.context import ContextManager
    from constellation.tools.registry import ToolRegistry

TaskTagMessages = tuple[AssistantMessageWithTools, UserMessageWithToolResults]


class EngineExecutor:
    def __init__(
        self,
        sync_engines: list[SyncEngine],
        async_engines: list[AsyncEngine],
        tool_registry: "ToolRegistry",
        context_manager: "ContextManager | None" = None,
    ):
        self._sync_engines = sync_engines
        self._async_engines = async_engines
        self._tool_registry = tool_registry
        self._context = context_manager

    def _get_engine_name(self, engine: SyncEngine | AsyncEngine, index: int) -> str:
        if isinstance(engine, SyncEngine):
            if engine.config.name:
                return engine.config.name
            choices = engine.config.output_choices
            return f"sync_engine_{index}_{'-'.join(choices[:2]).lower()}"
        return f"async_engine_{index}"

    def run_sync_engines(
        self,
        user_message: str,
        history: list[EngineConversationTurn],
    ) -> list[TaskTag]:
        if not self._sync_engines:
            return []

        all_tags: list[TaskTag] = []

        for i, engine in enumerate(self._sync_engines):
            engine_name = self._get_engine_name(engine, i)

            if self._context and self._context.is_engine_disabled(engine_name):
                LOGGER.debug(f"Skipping disabled engine: {engine_name}")
                continue

            start_time = time.time()

            try:
                tags = engine.process(user_message, history)
                duration_ms = int((time.time() - start_time) * 1000)
                LOGGER.debug(f"Engine {engine_name} completed in {duration_ms}ms with {len(tags)} tags")
                all_tags.extend(tags)
            except Exception as e:
                LOGGER.error(f"Sync engine {engine_name} failed: {e}")

        return all_tags

    def run_async_engines_background(
        self,
        user_message: str,
        history: list[EngineConversationTurn],
    ) -> None:
        if not self._async_engines:
            return

        async def run_all() -> None:
            tasks = []
            for i, engine in enumerate(self._async_engines):
                tasks.append(self._run_single_async_engine(engine, i, user_message, history))
            await asyncio.gather(*tasks, return_exceptions=True)

        Thread(target=lambda: asyncio.run(run_all()), daemon=True).start()

    async def _run_single_async_engine(
        self,
        engine: AsyncEngine,
        index: int,
        user_message: str,
        history: list[EngineConversationTurn],
    ) -> None:
        engine_name = self._get_engine_name(engine, index)
        start_time = time.time()
        try:
            await engine.process(user_message, history)
            duration_ms = int((time.time() - start_time) * 1000)
            LOGGER.info(f"Async engine {engine_name} completed in {duration_ms}ms")
        except Exception as e:
            LOGGER.error(f"Async engine {engine_name} failed: {e}")

    def process_task_tags(
        self,
        tags: list[TaskTag],
        user_message: str,
    ) -> TaskTagMessages | None:
        if not tags:
            return None

        results: list[TaskTagResult] = []

        for tag in tags:
            if isinstance(tag, TaskTagResult):
                results.append(tag)
            elif isinstance(tag, TaskTagInvocation):
                self._execute_engine_tool(tag)
            elif isinstance(tag, TaskTagDisable):
                if tag.engine_name and self._context:
                    self._context.disable_engine(tag.engine_name)
                    LOGGER.info(f"Engine disabled: {tag.engine_name}")

        return self._create_task_tag_messages(results, user_message)

    def _execute_engine_tool(self, tag: TaskTagInvocation) -> None:
        try:
            result = self._tool_registry.execute(tag.tool_name, tag.tool_input)
            LOGGER.info(f"TaskTag invoked {tag.tool_name}: {result}")

            if self._context:
                self._context.add_tool_result(tag.tool_name, result)

        except Exception as e:
            LOGGER.error(f"TaskTag invocation {tag.tool_name} failed: {e}")
            if self._context:
                self._context.add_tool_result(tag.tool_name, {"error": str(e)})

    def _create_task_tag_messages(
        self,
        results: list[TaskTagResult],
        user_message: str,
    ) -> TaskTagMessages | None:
        if not results:
            return None

        tool_use_messages: list[ToolUseMessage] = []
        tool_result_messages: list[ToolResultMessage] = []

        for result in results:
            engine_name = result.engine_name or result.task_name
            tool_name = result.synthetic_tool_name or "engine_task_result"
            tool_call_id = f"{engine_name[:15]}-{time.time_ns() % 1000000000}"

            tool_use_messages.append(
                ToolUseMessage(
                    id=tool_call_id,
                    name=tool_name,
                    input={},
                )
            )

            tool_result_messages.append(
                ToolResultMessage(
                    tool_use_id=tool_call_id,
                    content=result.result,
                )
            )

        return (
            AssistantMessageWithTools(tool_calls=tool_use_messages),
            UserMessageWithToolResults(tool_results=tool_result_messages),
        )
