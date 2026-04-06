import logging
from threading import Thread
from typing import Any

from deepgram import DeepgramClient
from deepgram.clients.listen.v1.websocket.response import LiveResultResponse

from constellation.audio.broadcaster import AudioSubscriber
from constellation.settings import SETTINGS


logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHANNELS = 1


class DeepgramASR(AudioSubscriber):
    MODEL = "nova-3"

    def __init__(self) -> None:
        self.deepgram = DeepgramClient(api_key=SETTINGS.require_deepgram_api_key())
        self.dg_connection: Any = None
        self.dg_context: Any = None
        self.dg_listening_thread: Thread | None = None

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

        def message_handler(data: Any) -> None:
            if isinstance(data, LiveResultResponse):
                self._handle_transcript(data)

        def close_handler(data: Any) -> None:
            logger.debug(f"Deepgram ASR connection closed: {data}")
            self.dg_connection = None

        def error_handler(data: Any) -> None:
            logger.error(f"Deepgram ASR error: {data}")

        try:
            connection_params = {
                "punctuate": "true",
                "model": self.MODEL,
                "language": "multi",
                "interim_results": "true",
                "sample_rate": str(SAMPLE_RATE),
                "channels": str(CHANNELS),
                "encoding": "linear16",
            }

            self.dg_context = self.deepgram.listen.websocket.v("1").open(**connection_params)
            self.dg_connection = self.dg_context.__enter__()

            self.dg_connection.on("Results", message_handler)
            self.dg_connection.on("Close", close_handler)
            self.dg_connection.on("Error", error_handler)

            logger.info("Deepgram ASR connection established")
        except Exception as e:
            logger.error(f"Deepgram ASR setup failed: {e}")
            self.dg_connection = None
            self.dg_context = None
            raise

    def stop(self) -> None:
        if self.dg_context is None:
            return

        try:
            self.dg_context.__exit__(None, None, None)
            logger.info("Deepgram ASR connection closed")
        except Exception as e:
            logger.error(f"ASR module stop failed: {e}")
        finally:
            self.dg_connection = None
            self.dg_context = None

    def on_audio(self, data: bytes) -> None:
        self.buffer += data
        self.frame_count += 1

        if self.dg_connection is None:
            self.setup()

        if self.dg_connection and self.frame_count >= self.num_buffer_frames:
            try:
                self.dg_connection.send(bytes(self.buffer))
            except Exception as e:
                logger.error(f"Deepgram ASR send failed: {e}")
            finally:
                self.buffer = bytearray(0)
                self.frame_count = 0

    def _handle_transcript(self, result: LiveResultResponse) -> None:
        if not result.channel.alternatives:
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
