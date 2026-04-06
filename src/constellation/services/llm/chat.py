from collections.abc import Sequence
from typing import Any

from constellation.core.turn import ConversationTurn, TurnRole, TurnState
from constellation.models.llm import LLMModel
from constellation.services.llm.service import LLMService
from constellation.services.llm.types import (
    AssistantMessageWithTools,
    ConversationMessage,
    LLMMessage,
    LLMMessageRole,
    LLMTool,
    SystemMessage,
    TextMessage,
    ToolCallDict,
    ToolResultDict,
    ToolResultMessage,
    ToolUseMessage,
    UserMessageWithToolResults,
)


DEFAULT_CONTEXT = "You are a helpful AI assistant. Be concise and friendly in your responses."


class ChatLLM(LLMService):
    def __init__(self, model: LLMModel = LLMModel.GPT_4_1):
        super().__init__(model)
        self.system_messages: list[SystemMessage] = [SystemMessage(text=DEFAULT_CONTEXT)]
        self.history: list[ConversationMessage] = []
        self.turn_history: list[ConversationTurn] = []

    def update_context(self, context_static: str, context_dynamic: str) -> None:
        self.system_messages = [
            SystemMessage(text=context_static),
            SystemMessage(text=context_dynamic),
        ]

    def add_to_turn_history(self, turn: ConversationTurn) -> None:
        if turn.role == TurnRole.ASSISTANT:
            self.turn_history.append(turn)
        elif turn.role == TurnRole.USER:
            last_turn = self.turn_history[-1] if self.turn_history else None
            if last_turn and last_turn.role == turn.role and last_turn.state == TurnState.COMPLETED:
                last_turn.content = f"{last_turn.content} {turn.content}".replace("[NO RESPONSE] ", "")
                last_turn.time_generated = turn.time_generated
            else:
                self.turn_history.append(turn)
        self._sync_history()

    def _sync_history(self) -> None:
        completed_turns = [t for t in self.turn_history if t.state != TurnState.CANCELED and t.content]
        self.history = [
            TextMessage(role=LLMMessageRole(t.role.value), content=t.content or "") for t in completed_turns
        ]

    def add_tool_exchange(self, tool_calls: list[ToolCallDict], tool_results: list[ToolResultDict]) -> None:
        tool_use_messages = [
            ToolUseMessage(
                id=tc.id,
                name=tc.name,
                input=tc.input if isinstance(tc.input, dict) else {},
            )
            for tc in tool_calls
        ]
        self.history.append(AssistantMessageWithTools(tool_calls=tool_use_messages))

        tool_result_messages = [
            ToolResultMessage(tool_use_id=tr.tool_call_id, content=tr.content) for tr in tool_results
        ]
        self.history.append(UserMessageWithToolResults(tool_results=tool_result_messages))

    def add_task_tag_exchange(
        self,
        assistant_msg: AssistantMessageWithTools,
        user_msg: UserMessageWithToolResults,
    ) -> None:
        self.history.append(assistant_msg)
        self.history.append(user_msg)

    def get_turn_history_string(self) -> str:
        return "\n".join(f"{t.role.value}: {t.content}" for t in self.turn_history if t.state != TurnState.CANCELED)

    def check_state(self, turn_id: str) -> TurnState | None:
        for turn in self.turn_history:
            if turn.id == turn_id:
                return turn.state
        return None

    def try_mark_turn_in_flight(self, turn_id: str) -> None:
        for turn in reversed(self.turn_history):
            if turn.id == turn_id and turn.state == TurnState.GENERATED:
                try:
                    turn.mark_in_flight()
                except ValueError:
                    pass
                break
        self._sync_history()

    def try_mark_turn_complete(self, turn_id: str) -> bool:
        for turn in reversed(self.turn_history):
            if turn.id == turn_id:
                if turn.state == TurnState.IN_FLIGHT:
                    try:
                        turn.complete()
                        self._sync_history()
                        return True
                    except ValueError:
                        pass
                elif turn.state == TurnState.CANCELED:
                    return False
                break
        self._sync_history()
        return False

    def cancel_pending_turns(self) -> None:
        for turn in reversed(self.turn_history):
            if turn.state not in [TurnState.COMPLETED, TurnState.CANCELED]:
                try:
                    turn.cancel()
                except ValueError:
                    pass
        self._sync_history()

    def query(self, stream: bool = True, tools: list[LLMTool] | None = None) -> Any:
        messages: Sequence[LLMMessage] = [*self.system_messages, *self.history]
        if stream:
            return self.get_stream(messages, tools)
        else:
            return self.get_response(messages, tools)

    def get_tools_list(self, tool_registry: Any) -> list[LLMTool]:
        return tool_registry.get_tools()
