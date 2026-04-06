from enum import StrEnum
import time
from typing import Any
from uuid import uuid4


class TurnRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class TurnState(StrEnum):
    GENERATED = "generated"
    IN_FLIGHT = "in_flight"
    COMPLETED = "completed"
    CANCELED = "canceled"


class ConversationTurn:
    def __init__(
        self,
        role: TurnRole,
        state: TurnState,
        content: str | None = None,
        source: list[Any] | None = None,
    ):
        self.id = uuid4().hex[:12]
        self.role = role
        self.state = state
        self.content = content
        self.source = source
        self.time_generated = time.time()

    def mark_in_flight(self) -> None:
        if self.state != TurnState.GENERATED:
            raise ValueError(f"Cannot mark {self.state} turn as in_flight")
        self.state = TurnState.IN_FLIGHT

    def complete(self) -> None:
        if self.state != TurnState.IN_FLIGHT:
            raise ValueError(f"Cannot complete turn in {self.state} state")
        self.state = TurnState.COMPLETED

    def cancel(self) -> None:
        if self.state in [TurnState.COMPLETED, TurnState.CANCELED]:
            raise ValueError(f"Cannot cancel turn in {self.state} state")
        self.state = TurnState.CANCELED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role.value,
            "state": self.state.value,
            "content": self.content,
            "time_generated": self.time_generated,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConversationTurn":
        turn = cls(
            role=TurnRole(data["role"]),
            state=TurnState(data["state"]),
            content=data.get("content"),
        )
        turn.id = data["id"]
        turn.time_generated = data.get("time_generated", time.time())
        return turn
