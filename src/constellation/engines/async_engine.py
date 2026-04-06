import logging

from constellation.models.engine import AsyncEngineConfig
from constellation.services.llm.service import LLMService


logger = logging.getLogger(__name__)

EngineConversationTurn = dict[str, str]


class AsyncEngine:
    def __init__(self, config: AsyncEngineConfig):
        self.config = config

    def _format_turns(self, history: list[EngineConversationTurn]) -> str:
        limit = self.config.num_turns
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

    async def process(
        self,
        user_message: str,
        history: list[EngineConversationTurn] | None = None,
    ) -> None:
        history = history or []

        try:
            llm_service = LLMService(model=self.config.model)
            user_prompt = self._build_prompt(user_message, history)
            full_prompt = f"{self.config.system_prompt}\n\n{user_prompt}"

            from constellation.services.llm.types import LLMMessageRole, TextMessage

            messages = [TextMessage(role=LLMMessageRole.USER, content=full_prompt)]
            response = llm_service.get_response(messages, tools=None)

            logger.info(f"AsyncEngine completed: {str(response)[:100]}...")

        except Exception as e:
            logger.error(f"AsyncEngine processing failed: {e}")
