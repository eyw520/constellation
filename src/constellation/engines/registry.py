from constellation.engines.async_engine import AsyncEngine
from constellation.engines.sync_engine import SyncEngine
from constellation.models.engine import AsyncEngineConfig, EngineConfig, SyncEngineConfig


class EngineRegistry:
    @classmethod
    def create_sync_engines(cls, configs: list[EngineConfig]) -> list[SyncEngine]:
        engines: list[SyncEngine] = []
        for config in configs:
            if isinstance(config, SyncEngineConfig):
                engines.append(SyncEngine(config))
        return engines

    @classmethod
    def create_async_engines(cls, configs: list[EngineConfig]) -> list[AsyncEngine]:
        engines: list[AsyncEngine] = []
        for config in configs:
            if isinstance(config, AsyncEngineConfig):
                engines.append(AsyncEngine(config))
        return engines
