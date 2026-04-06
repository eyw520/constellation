from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from constellation.core.agent import Agent


VOICE_MODALITY_INSTRUCTIONS = """\
Your responses will be passed through text-to-speech and must be spoken and natural. \
Answers must be concise and conversational. DO NOT include lists or bullets in your answer.\
"""

STATIC_CONTEXT_PREFIX = """\
You are an AI voice assistant.\
"""

DYNAMIC_CONTEXT_TEMPLATE = """
{modality_instructions}
The conversation began at {starting_date_time}. The current date time is {current_date_time}. \
The conversation begins below:\n
"""


@dataclass
class DynamicContextData:
    modality_instructions: str = ""
    starting_date_time: datetime | None = None
    current_date_time: datetime | None = None
    tool_results: dict[str, Any] = field(default_factory=dict)

    def to_template_vars(self) -> dict[str, str]:
        return {
            "modality_instructions": self.modality_instructions,
            "starting_date_time": (
                self.starting_date_time.strftime("%Y-%m-%d %H:%M:%S") if self.starting_date_time else "unknown"
            ),
            "current_date_time": (
                self.current_date_time.strftime("%Y-%m-%d %H:%M:%S") if self.current_date_time else "unknown"
            ),
        }


@dataclass
class ContextState:
    context_static: str
    context_dynamic: str
    dynamic_context: DynamicContextData


class ContextBuilder:
    def __init__(self, agent: "Agent"):
        self.agent = agent
        self.context = agent.get_system_context()
        self.modality_instructions = VOICE_MODALITY_INSTRUCTIONS

    def build_state(self) -> ContextState:
        static_parts = [STATIC_CONTEXT_PREFIX]
        if self.context:
            static_parts.append(self.context)

        return ContextState(
            context_static="\n\n".join(static_parts),
            context_dynamic=DYNAMIC_CONTEXT_TEMPLATE,
            dynamic_context=DynamicContextData(modality_instructions=self.modality_instructions),
        )


class ContextManager:
    def __init__(self, agent: "Agent"):
        self.context_builder = ContextBuilder(agent=agent)
        self.state = self.context_builder.build_state()
        self._static_context_cached: str | None = None
        self._disabled_engines: set[str] = set()
        self._initialize()

    def _initialize(self) -> None:
        now = datetime.now()
        self.state.dynamic_context.starting_date_time = now
        self.state.dynamic_context.current_date_time = now
        self._static_context_cached = self.state.context_static

    def get_context(self) -> tuple[str, str]:
        self.state.dynamic_context.current_date_time = datetime.now()
        context_dynamic = self.state.context_dynamic.format(**self.state.dynamic_context.to_template_vars())
        return self._static_context_cached or "", context_dynamic

    def add_tool_result(self, tool_name: str, result: dict[str, Any]) -> None:
        self.state.dynamic_context.tool_results[tool_name] = result

    def disable_engine(self, engine_name: str) -> None:
        self._disabled_engines.add(engine_name)

    def is_engine_disabled(self, engine_name: str) -> bool:
        return engine_name in self._disabled_engines

    @property
    def disabled_engines(self) -> set[str]:
        return self._disabled_engines.copy()
