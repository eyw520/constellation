from typing import Any

from deepgram import DeepgramClient, LiveResultResponse, LiveTranscriptionEvents
from deepgram.clients.listen.v1.websocket.options import LiveOptions

from constellation.audio.broadcaster import AudioSubscriber
from constellation.logger import LOGGER
from constellation.settings import SETTINGS


SAMPLE_RATE = 16000
CHANNELS = 1


class DeepgramASR(AudioSubscriber):
    MODEL = "nova-3"

    def __init__(self) -> None:
        self.deepgram = DeepgramClient(api_key=SETTINGS.require_deepgram_api_key())
        self.dg_connection: Any = None

        self.transcript = ""
        self.transcript_updated = False
        self.has_partial_transcript = False
        self.unfinalized_transcript = ""

        self.buffer = bytearray(0)
        self.frame_count = 0
        self.num_buffer_frames = 5

    def setup(self) -> None:
        if self.dg_connection is not None:
            return

        def transcript_handler(
            _client: Any, result: LiveResultResponse, **_kwargs: Any
        ) -> None:
            self._handle_transcript(result)

        def close_handler(_client: Any, close: Any, **_kwargs: Any) -> None:
            LOGGER.debug(f"Deepgram ASR connection closed: {close}")
            self.dg_connection = None

        def error_handler(_client: Any, error: Any, **_kwargs: Any) -> None:
            LOGGER.error(f"Deepgram ASR error: {error}")

        try:
            options = LiveOptions(
                punctuate=True,
                model=self.MODEL,
                language="multi",
                interim_results=True,
                sample_rate=SAMPLE_RATE,
                channels=CHANNELS,
                encoding="linear16",
            )

            self.dg_connection = self.deepgram.listen.websocket.v("1")

            self.dg_connection.on(LiveTranscriptionEvents.Transcript, transcript_handler)
            self.dg_connection.on(LiveTranscriptionEvents.Close, close_handler)
            self.dg_connection.on(LiveTranscriptionEvents.Error, error_handler)

            if not self.dg_connection.start(options):
                raise RuntimeError("Failed to start Deepgram connection")

            LOGGER.info("Deepgram ASR connection established")
        except Exception as e:
            LOGGER.error(f"Deepgram ASR setup failed: {e}")
            self.dg_connection = None
            raise

    def stop(self) -> None:
        if self.dg_connection is None:
            return

        try:
            self.dg_connection.finish()
            LOGGER.info("Deepgram ASR connection closed")
        except Exception as e:
            LOGGER.error(f"ASR module stop failed: {e}")
        finally:
            self.dg_connection = None

    def on_audio(self, data: bytes) -> None:
        self.buffer += data
        self.frame_count += 1

        if self.dg_connection is None:
            self.setup()

        if self.dg_connection and self.frame_count >= self.num_buffer_frames:
            try:
                self.dg_connection.send(bytes(self.buffer))
            except Exception as e:
                LOGGER.error(f"Deepgram ASR send failed: {e}")
            finally:
                self.buffer = bytearray(0)
                self.frame_count = 0

    def _handle_transcript(self, result: LiveResultResponse) -> None:
        if not result.channel or not result.channel.alternatives:
            return

        chunk = result.channel.alternatives[0].transcript
        if chunk:
            if result.is_final:
                self.transcript_updated = True
                self.has_partial_transcript = False
                self.transcript = self.transcript + " " + chunk if self.transcript else chunk
                self.unfinalized_transcript = ""
            else:
                self.unfinalized_transcript = chunk
                self.has_partial_transcript = True

    def poll_transcript(self) -> str | None:
        if self.transcript_updated:
            self.transcript_updated = False
            transcript = self.transcript.strip()
            self.transcript = ""
            return transcript
        return None

    def reset(self) -> None:
        self.transcript = ""
        self.transcript_updated = False
        self.has_partial_transcript = False
        self.unfinalized_transcript = ""
