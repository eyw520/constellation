from enum import Enum
import logging
from typing import Annotated, Any

from pydantic import BaseModel, Field

from constellation.models.engine import SyncEngineConfig
from constellation.models.task_tag import TaskTag, TaskTagDisable, TaskTagResult
from constellation.services.llm.service import LLMService


logger = logging.getLogger(__name__)

EngineConversationTurn = dict[str, str]


class GateResponse(BaseModel):
    result: bool = Field(description="Whether the gate condition is satisfied")


def _create_output_enum(choices: list[str]) -> type[Enum]:
    return Enum("OutputEnum", {choice: choice for choice in choices})  # type: ignore


def _create_response_model(output_enum: type[Enum]) -> type[BaseModel]:
    class ResponseModel(BaseModel):
        result: Annotated[output_enum, Field(description="The classification result")]  # type: ignore

    return ResponseModel


class SyncEngine:
    def __init__(self, config: SyncEngineConfig):
        self.config = config
        self._output_enum = _create_output_enum(config.output_choices)
        self._response_model = _create_response_model(self._output_enum)

    def _format_turns(self, history: list[EngineConversationTurn], num_turns: int | None = None) -> str:
        limit = num_turns if num_turns is not None else self.config.num_turns
        recent_turns = history[-limit:] if history else []
        formatted = []
        for turn in recent_turns:
            role = turn["role"].capitalize()
            content = turn["content"]
            formatted.append(f"{role}: {content}")
        return "\n".join(formatted)

    def _build_prompt(self, user_message: str, history: list[EngineConversationTurn]) -> str:
        turns_str = self._format_turns(history)
        if user_message:
            turns_str += f"\nUser: {user_message}"
        return self.config.user_prompt_template.format(turns=turns_str)

    def _evaluate_gate(
        self,
        user_message: str,
        history: list[EngineConversationTurn],
    ) -> bool:
        gate = self.config.gate
        if gate is None:
            return True

        model = gate.model or self.config.model
        llm_service = LLMService(model=model)

        turns_str = self._format_turns(history, num_turns=gate.num_turns)
        if user_message:
            turns_str += f"\nUser: {user_message}"

        full_prompt = f"{gate.prompt}\n\nConversation:\n{turns_str}"

        try:
            response: GateResponse = llm_service.generate_structured_output(
                prompt=full_prompt,
                response_type=GateResponse,
            )
            logger.debug(f"SyncEngine gate evaluated: {response.result}")
            return response.result
        except Exception as e:
            logger.error(f"SyncEngine gate evaluation failed: {e}")
            return True

    def process(
        self,
        user_message: str,
        history: list[EngineConversationTurn] | None = None,
    ) -> list[TaskTag]:
        history = history or []

        try:
            if not self._evaluate_gate(user_message, history):
                logger.debug("SyncEngine gate returned false, skipping classification")
                return []

            llm_service = LLMService(model=self.config.model)
            user_prompt = self._build_prompt(user_message, history)
            full_prompt = f"{self.config.system_prompt}\n\n{user_prompt}"

            response: Any = llm_service.generate_structured_output(
                prompt=full_prompt,
                response_type=self._response_model,
            )

            result_value = response.result.value
            task_tag = self.config.task_tag_mapping.get(result_value)

            if task_tag is None:
                logger.debug(f"SyncEngine classified as '{result_value}' -> silent (null mapping)")
                return []

            logger.debug(f"SyncEngine classified as '{result_value}' -> {task_tag}")

            if isinstance(task_tag, TaskTagResult):
                task_tag = task_tag.model_copy(
                    update={
                        "engine_name": self.config.name,
                        "synthetic_tool_name": self.config.synthetic_tool_name,
                    }
                )
            elif isinstance(task_tag, TaskTagDisable):
                task_tag = task_tag.model_copy(update={"engine_name": self.config.name})

            return [task_tag]

        except Exception as e:
            logger.error(f"SyncEngine processing failed: {e}")
            return []
