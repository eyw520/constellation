import asyncio
from collections.abc import Generator
import json
import queue
import threading
import time
from typing import Any
from uuid import uuid4

from constellation.audio.input import MicrophoneInput
from constellation.audio.output import SpeakerOutput
from constellation.core.agent import Agent
from constellation.core.context import ContextManager
from constellation.core.turn import ConversationTurn, TurnRole, TurnState
from constellation.engines.executor import EngineExecutor
from constellation.logger import LOGGER
from constellation.services.asr.deepgram import DeepgramASR
from constellation.services.llm.chat import ChatLLM
from constellation.services.llm.service import InferenceCancelledException
from constellation.services.llm.types import ToolCallDict, ToolResultDict
from constellation.services.tts.openai import OpenAITTS
from constellation.services.vad.webrtc import VADState, WebRTCVAD


class VoiceSession:
    def __init__(self, agent: Agent):
        self.agent = agent
        self.context_manager: ContextManager | None = None
        self.llm: ChatLLM | None = None
        self.engine_executor: EngineExecutor | None = None

        self.mic_input = MicrophoneInput()
        self.speaker_output = SpeakerOutput()
        self.asr = DeepgramASR()
        self.tts = OpenAITTS()
        self.vad = WebRTCVAD()

        self._running = False
        self._processing_thread: threading.Thread | None = None
        self._transcript_queue: queue.Queue[str] = queue.Queue()

        self._last_vad_state = VADState.SILENCE
        self._speech_end_time: float | None = None
        self._speech_end_delay = 0.6

    async def start(self) -> None:
        await self.agent.start()

        self.context_manager = ContextManager(self.agent)
        self.llm = ChatLLM(model=self.agent.get_llm_config().model)
        self.engine_executor = EngineExecutor(
            sync_engines=self.agent.get_sync_engines(),
            async_engines=self.agent.get_async_engines(),
            tool_registry=self.agent.get_tool_registry(),
            context_manager=self.context_manager,
        )

        context_static, context_dynamic = self.context_manager.get_context()
        self.llm.update_context(context_static, context_dynamic)

        self.mic_input.subscribe(self.asr)
        self.mic_input.subscribe(self._on_audio_for_vad)

        self.asr.setup()
        self.mic_input.start()
        self.speaker_output.start()

        self._running = True
        self._processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._processing_thread.start()

        LOGGER.info("Voice session started")

    async def stop(self) -> None:
        self._running = False

        self.mic_input.stop()
        self.speaker_output.stop()
        self.asr.stop()

        if self._processing_thread:
            self._processing_thread.join(timeout=2.0)

        await self.agent.stop()
        LOGGER.info("Voice session stopped")

    def _on_audio_for_vad(self, audio: bytes) -> None:
        current_state = self.vad.process(audio)

        if self._last_vad_state == VADState.SPEECH and current_state == VADState.SILENCE:
            self._speech_end_time = time.time()
        elif current_state == VADState.SPEECH:
            self._speech_end_time = None

            if self.speaker_output.is_playing():
                LOGGER.info("User speaking - interrupting playback")
                self._interrupt()

        self._last_vad_state = current_state

    def _interrupt(self) -> None:
        self.speaker_output.interrupt()
        if self.llm:
            self.llm.cancel_pending_turns()
            self.llm.cancel_inference()

    def _processing_loop(self) -> None:
        while self._running:
            try:
                transcript = self.asr.poll_transcript()

                if transcript:
                    if self._speech_end_time and (time.time() - self._speech_end_time) < self._speech_end_delay:
                        time.sleep(self._speech_end_delay)
                        additional = self.asr.poll_transcript()
                        if additional:
                            transcript = f"{transcript} {additional}"

                    self._process_transcript(transcript)

                time.sleep(0.05)
            except Exception as e:
                LOGGER.error(f"Processing loop error: {e}", exc_info=True)

    def _process_transcript(self, transcript: str) -> None:
        if not self.llm or not self.context_manager or not self.engine_executor:
            return

        LOGGER.info(f"User: {transcript}")
        print(f"\nUser: {transcript}")

        self._interrupt()

        user_turn = ConversationTurn(role=TurnRole.USER, state=TurnState.COMPLETED, content=transcript)
        self.llm.add_to_turn_history(user_turn)

        history = self._get_engine_history()
        task_tags = self.engine_executor.run_sync_engines(transcript, history)
        synthetic_messages = self.engine_executor.process_task_tags(task_tags, transcript)
        self.engine_executor.run_async_engines_background(transcript, history)

        if synthetic_messages:
            assistant_msg, user_tool_result = synthetic_messages
            self.llm.add_task_tag_exchange(assistant_msg, user_tool_result)

        context_static, context_dynamic = self.context_manager.get_context()
        self.llm.update_context(context_static, context_dynamic)

        self._process_llm_turn()

    def _get_engine_history(self) -> list[dict[str, str]]:
        if not self.llm:
            return []
        return [
            {"role": turn.role.value, "content": turn.content or ""}
            for turn in self.llm.turn_history
            if turn.state == TurnState.COMPLETED and turn.content
        ]

    def _process_llm_turn(self) -> None:
        if not self.llm:
            return

        tools = self.llm.get_tools_list(self.agent.get_tool_registry())
        llm_output = self.llm.query(stream=True, tools=tools)

        content_tokens: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        def create_tts_stream() -> Generator[str, None, None]:
            try:
                for type_, token in llm_output:
                    if type_ == "content":
                        content_tokens.append(token)
                        yield token
                    elif type_ == "tool":
                        tool_calls.append(token)
            except InferenceCancelledException:
                LOGGER.info("LLM inference cancelled")

        tts_stream = create_tts_stream()

        assistant_turn = ConversationTurn(
            role=TurnRole.ASSISTANT,
            state=TurnState.GENERATED,
            content=None,
        )
        self.llm.add_to_turn_history(assistant_turn)

        try:
            audio_stream = self.tts.run(tts_stream, input_streaming=True)
            self.speaker_output.play_stream(audio_stream)

            content = "".join(content_tokens)
            if content:
                assistant_turn.content = content
                LOGGER.info(f"Assistant: {content[:100]}...")
                print(f"Assistant: {content}")

            self.llm.try_mark_turn_in_flight(assistant_turn.id)
            self.llm.try_mark_turn_complete(assistant_turn.id)

        except InferenceCancelledException:
            LOGGER.info("Response interrupted")
            return

        if tool_calls:
            tool_results = self._handle_tool_calls(tool_calls)
            tool_call_dicts = [ToolCallDict(**tc) for tc in tool_calls]
            tool_result_dicts = [ToolResultDict(**tr) for tr in tool_results]
            self.llm.add_tool_exchange(tool_call_dicts, tool_result_dicts)
            self._process_llm_turn()

    def _handle_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        tool_registry = self.agent.get_tool_registry()

        for tool_call in tool_calls:
            tool_name = tool_call.get("name")
            tool_call_id = tool_call.get("id", str(uuid4()))
            tool_input = tool_call.get("input", {})

            if not tool_name:
                continue

            LOGGER.info(f"Tool call: {tool_name}({tool_input})")
            print(f"Tool: {tool_name}({tool_input})")

            try:
                if asyncio.iscoroutinefunction(tool_registry.get_handler(tool_name)):
                    result = asyncio.run(tool_registry.execute_async(tool_name, tool_input))
                else:
                    result = tool_registry.execute(tool_name, tool_input)

                result_str = json.dumps(result) if isinstance(result, dict) else str(result)
                LOGGER.info(f"Tool result: {tool_name} -> {result_str[:100]}...")
                print(f"Tool Result: {result_str[:200]}")

                results.append({"tool_call_id": tool_call_id, "content": result_str})

                if self.context_manager:
                    self.context_manager.add_tool_result(tool_name, result)

                if isinstance(result, dict) and result.get("end_conversation"):
                    LOGGER.info(f"End conversation signal: {result.get('reason')}")
                    self._running = False

            except Exception as e:
                LOGGER.error(f"Tool {tool_name} failed: {e}")
                error_content = json.dumps({"error": str(e)})
                results.append({"tool_call_id": tool_call_id, "content": error_content})

        return results

    def run_initiation(self) -> None:
        if not self.agent.should_initiate():
            return

        signal = self.agent.get_initiation_signal()
        LOGGER.info(f"Initiating conversation with signal: {signal}")

        threading.Thread(
            target=lambda: self._process_transcript(signal),
            daemon=True,
        ).start()

    def is_running(self) -> bool:
        return self._running

    def toggle_mute(self) -> bool:
        return self.mic_input.toggle_mute()

    def is_muted(self) -> bool:
        return self.mic_input.is_muted()
