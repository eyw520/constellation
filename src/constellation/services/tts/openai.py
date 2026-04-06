import audioop
from collections.abc import Generator, Iterator
import re

from openai import OpenAI

from constellation.logger import LOGGER
from constellation.settings import SETTINGS


SENTENCE_ENDINGS = re.compile(r"[.!?]+\s*")
MIN_SENTENCE_LENGTH = 10


class OpenAITTS:
    MODEL = "tts-1"
    VOICE = "shimmer"

    def __init__(self) -> None:
        self.client = OpenAI(api_key=SETTINGS.require_openai_api_key())
        self.total_characters = 0

    def stop(self) -> None:
        pass

    def _synthesize(self, text: str) -> Generator[bytes, None, None]:
        self.total_characters += len(text)

        response_context = self.client.audio.speech.with_streaming_response.create(
            input=text,
            model=self.MODEL,
            voice=self.VOICE,  # type: ignore
            response_format="pcm",
        )

        with response_context as response:
            for chunk in response.iter_bytes(chunk_size=1024):
                converted = self._convert_sample_rate(chunk, 24000, 24000)
                yield converted

    def _convert_sample_rate(self, chunk: bytes, from_rate: int, to_rate: int) -> bytes:
        if from_rate == to_rate:
            return chunk
        return audioop.ratecv(chunk, 2, 1, from_rate, to_rate, None)[0]

    def _buffer_sentences(self, text_stream: Iterator[str]) -> Iterator[str]:
        buffer = ""

        for token in text_stream:
            buffer += token

            while True:
                match = SENTENCE_ENDINGS.search(buffer)
                if match and match.end() >= MIN_SENTENCE_LENGTH:
                    sentence = buffer[: match.end()].strip()
                    buffer = buffer[match.end() :]
                    if sentence:
                        yield sentence
                else:
                    break

        if buffer.strip():
            yield buffer.strip()

    def synthesize(self, text: str) -> Generator[bytes, None, None]:
        yield from self._synthesize(text)

    def synthesize_stream(self, text_stream: Iterator[str]) -> Generator[bytes, None, None]:
        try:
            for sentence in self._buffer_sentences(text_stream):
                yield from self._synthesize(sentence)
        except Exception as e:
            LOGGER.error(f"TTS synthesis failed: {e}")

    def run(
        self,
        tts_input: str | Iterator[str],
        input_streaming: bool = False,
    ) -> Generator[bytes, None, None]:
        if input_streaming and not isinstance(tts_input, str):
            return self.synthesize_stream(tts_input)
        elif isinstance(tts_input, str):
            return self.synthesize(tts_input)
        else:
            raise ValueError("Invalid TTS input")
