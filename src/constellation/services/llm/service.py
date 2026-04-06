from collections.abc import Generator, Sequence
from concurrent.futures import CancelledError, Future, ThreadPoolExecutor
import json
from queue import Empty, Queue
from threading import Lock
from typing import Any, TypeVar

from anthropic import Anthropic
from anthropic._types import NOT_GIVEN
from openai import OpenAI
from pydantic import BaseModel

from constellation.models.llm import LLMModel, is_anthropic_model, is_openai_model
from constellation.services.llm.types import (
    LLMMessage,
    LLMMessageRole,
    LLMTool,
    SystemMessage,
    TextMessage,
)
from constellation.settings import SETTINGS


T = TypeVar("T", bound=BaseModel)

MAX_TOKENS = 4096
NUM_RETRIES = 3


class InferenceCancelledException(Exception):
    pass


def get_client(model: LLMModel) -> OpenAI | Anthropic:
    if is_openai_model(model):
        return OpenAI(api_key=SETTINGS.require_openai_api_key())
    elif is_anthropic_model(model):
        return Anthropic(api_key=SETTINGS.require_anthropic_api_key())
    else:
        raise ValueError(f"Unsupported model: {model}")


def convert_messages_to_openai(messages: Sequence[LLMMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for msg in messages:
        openai_msg = msg.to_openai()
        if isinstance(openai_msg, list):
            result.extend(openai_msg)
        else:
            result.append(openai_msg)
    return result


def convert_messages_to_anthropic(
    messages: Sequence[LLMMessage],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    system_messages: list[dict[str, Any]] = []
    conversation_messages: list[dict[str, Any]] = []

    for msg in messages:
        if isinstance(msg, SystemMessage):
            system_messages.append(msg.to_anthropic())
        else:
            conversation_messages.append(msg.to_anthropic())

    return system_messages, conversation_messages


def convert_tools_to_openai(tools: list[LLMTool] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [tool.to_openai() for tool in tools]


def convert_tools_to_anthropic(tools: list[LLMTool] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [tool.to_anthropic() for tool in tools]


class LLMService:
    def __init__(self, model: LLMModel = LLMModel.GPT_4_1):
        self.model = model
        self.client = get_client(model)
        self.max_tokens = MAX_TOKENS
        self.stop_tokens: list[str] | None = None
        self._request_completed_sentinel = object()
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._futures_lock = Lock()
        self._inference_futures: list[Future[Any]] = []

    def cancel_inference(self) -> None:
        with self._futures_lock:
            for future in self._inference_futures:
                future.cancel()
            self._inference_futures = []

    def _run_generator_loop(
        self,
        messages: Sequence[LLMMessage],
        stream: bool,
        tools: list[LLMTool] | None,
    ) -> Generator[Any, None, None]:
        output_queue: Queue[Any] = Queue()

        def wrapped_func() -> None:
            try:
                if stream:
                    self._get_stream(messages, output_queue, tools)
                else:
                    self._get_response(messages, output_queue, tools)
            except Exception as e:
                output_queue.put(e)

        future = self._executor.submit(wrapped_func)
        with self._futures_lock:
            self._inference_futures.append(future)

        try:
            while True:
                finished = False
                try:
                    output = output_queue.get(timeout=0.1)
                    if isinstance(output, Exception):
                        raise RuntimeError("LLM inference failed") from output
                    if output is self._request_completed_sentinel:
                        finished = True
                        break
                    yield output
                except CancelledError as e:
                    raise InferenceCancelledException() from e
                except Empty:
                    pass
                finally:
                    with self._futures_lock:
                        if future not in self._inference_futures:
                            raise InferenceCancelledException()
                        elif finished:
                            self._inference_futures.remove(future)
                    if finished:
                        break
        except GeneratorExit:
            with self._futures_lock:
                if future in self._inference_futures:
                    self._inference_futures.remove(future)
            raise

    def _get_stream(
        self,
        messages: Sequence[LLMMessage],
        output_queue: Queue[Any],
        tools: list[LLMTool] | None,
    ) -> None:
        if is_openai_model(self.model):
            self._get_openai_stream(messages, output_queue, tools)
        elif is_anthropic_model(self.model):
            self._get_anthropic_stream(messages, output_queue, tools)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _get_openai_stream(
        self,
        messages: Sequence[LLMMessage],
        output_queue: Queue[Any],
        tools: list[LLMTool] | None,
    ) -> None:
        if not isinstance(self.client, OpenAI):
            raise TypeError("Expected OpenAI client")

        openai_messages = convert_messages_to_openai(messages)
        openai_tools = convert_tools_to_openai(tools)

        stream = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,  # type: ignore
            max_tokens=self.max_tokens,
            stop=self.stop_tokens if self.stop_tokens else None,
            tools=openai_tools,  # type: ignore
            stream=True,
        )

        tool_calls: dict[int, dict[str, Any]] = {}

        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta is None:
                continue

            if delta.content:
                output_queue.put(("content", delta.content))

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls:
                        tool_calls[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls[idx]["arguments"] += tc.function.arguments

        if tool_calls:
            for tc_data in tool_calls.values():
                try:
                    args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                except json.JSONDecodeError:
                    args = {}
                output_queue.put(
                    (
                        "tool",
                        {"id": tc_data["id"], "name": tc_data["name"], "input": args},
                    )
                )

        output_queue.put(self._request_completed_sentinel)

    def _get_anthropic_stream(
        self,
        messages: Sequence[LLMMessage],
        output_queue: Queue[Any],
        tools: list[LLMTool] | None,
    ) -> None:
        if not isinstance(self.client, Anthropic):
            raise TypeError("Expected Anthropic client")

        system_messages, conversation_messages = convert_messages_to_anthropic(messages)
        anthropic_tools = convert_tools_to_anthropic(tools)

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_messages,  # type: ignore
            messages=conversation_messages,  # type: ignore
            tools=anthropic_tools or NOT_GIVEN,  # type: ignore
        ) as stream:
            current_tool: dict[str, Any] | None = None

            for event in stream:
                if event.type == "content_block_start":
                    if hasattr(event.content_block, "type"):
                        if event.content_block.type == "tool_use":
                            current_tool = {
                                "id": event.content_block.id,
                                "name": event.content_block.name,
                                "input": "",
                            }
                elif event.type == "content_block_delta":
                    if hasattr(event.delta, "text"):
                        output_queue.put(("content", event.delta.text))  # type: ignore[union-attr]
                    elif hasattr(event.delta, "partial_json"):
                        if current_tool:
                            current_tool["input"] += event.delta.partial_json  # type: ignore[union-attr]
                elif event.type == "content_block_stop":
                    if current_tool:
                        try:
                            args = json.loads(current_tool["input"]) if current_tool["input"] else {}
                        except json.JSONDecodeError:
                            args = {}
                        output_queue.put(
                            (
                                "tool",
                                {"id": current_tool["id"], "name": current_tool["name"], "input": args},
                            )
                        )
                        current_tool = None

        output_queue.put(self._request_completed_sentinel)

    def _get_response(
        self,
        messages: Sequence[LLMMessage],
        output_queue: Queue[Any],
        tools: list[LLMTool] | None,
    ) -> None:
        if is_openai_model(self.model):
            self._get_openai_response(messages, output_queue, tools)
        elif is_anthropic_model(self.model):
            self._get_anthropic_response(messages, output_queue, tools)
        else:
            raise ValueError(f"Unsupported model: {self.model}")

    def _get_openai_response(
        self,
        messages: Sequence[LLMMessage],
        output_queue: Queue[Any],
        tools: list[LLMTool] | None,
    ) -> None:
        if not isinstance(self.client, OpenAI):
            raise TypeError("Expected OpenAI client")

        openai_messages = convert_messages_to_openai(messages)
        openai_tools = convert_tools_to_openai(tools)

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=openai_messages,  # type: ignore
            max_tokens=self.max_tokens,
            stop=self.stop_tokens if self.stop_tokens else None,
            tools=openai_tools,  # type: ignore
        )

        choice = completion.choices[0]
        if choice.message.content:
            output_queue.put(choice.message.content)
        elif choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}  # type: ignore[union-attr]
                except json.JSONDecodeError:
                    args = {}
                output_queue.put(
                    (
                        "tool",
                        {"id": tc.id, "name": tc.function.name, "input": args},  # type: ignore[union-attr]
                    )
                )

        output_queue.put(self._request_completed_sentinel)

    def _get_anthropic_response(
        self,
        messages: Sequence[LLMMessage],
        output_queue: Queue[Any],
        tools: list[LLMTool] | None,
    ) -> None:
        if not isinstance(self.client, Anthropic):
            raise TypeError("Expected Anthropic client")

        system_messages, conversation_messages = convert_messages_to_anthropic(messages)
        anthropic_tools = convert_tools_to_anthropic(tools)

        message = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system_messages,  # type: ignore
            messages=conversation_messages,  # type: ignore
            tools=anthropic_tools or NOT_GIVEN,  # type: ignore
        )

        for block in message.content:
            if block.type == "text":
                output_queue.put(block.text)
            elif block.type == "tool_use":
                output_queue.put(
                    (
                        "tool",
                        {"id": block.id, "name": block.name, "input": block.input},
                    )
                )

        output_queue.put(self._request_completed_sentinel)

    def get_response(self, messages: Sequence[LLMMessage], tools: list[LLMTool] | None = None) -> Any:
        for output in self._run_generator_loop(messages, stream=False, tools=tools):
            return output
        raise RuntimeError("No response from LLM")

    def get_stream(
        self, messages: Sequence[LLMMessage], tools: list[LLMTool] | None = None
    ) -> Generator[Any, None, None]:
        yield from self._run_generator_loop(messages, stream=True, tools=tools)

    def generate_structured_output(
        self,
        prompt: str,
        response_type: type[T],
    ) -> T:
        messages = [TextMessage(role=LLMMessageRole.USER, content=prompt)]

        if is_openai_model(self.model):
            if not isinstance(self.client, OpenAI):
                raise TypeError("Expected OpenAI client")
            openai_messages = convert_messages_to_openai(messages)
            result = self.client.beta.chat.completions.parse(
                model=self.model,
                messages=openai_messages,  # type: ignore
                max_tokens=self.max_tokens,
                response_format=response_type,
            )
            parsed = result.choices[0].message.parsed
            if parsed is None:
                raise ValueError(f"Failed to parse response for {response_type.__name__}")
            return parsed

        elif is_anthropic_model(self.model):
            tool = LLMTool(
                name="build_response",
                description="Build the structured response",
                input_schema=response_type,
            )
            response = self.get_response(messages, [tool])
            if isinstance(response, tuple) and response[0] == "tool":
                return response_type(**response[1]["input"])
            raise ValueError("Expected tool response from Anthropic")

        raise ValueError(f"Unsupported model: {self.model}")
