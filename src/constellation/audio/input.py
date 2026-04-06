import queue
import threading
from typing import Any

import numpy as np
import sounddevice as sd

from constellation.audio.broadcaster import AudioBroadcaster
from constellation.logger import LOGGER


SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_MS = 20
CHUNK_SAMPLES = int(SAMPLE_RATE * CHUNK_MS / 1000)


class MicrophoneInput:
    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        channels: int = CHANNELS,
        chunk_ms: int = CHUNK_MS,
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_samples = int(sample_rate * chunk_ms / 1000)
        self.broadcaster = AudioBroadcaster()

        self._stream: sd.InputStream | None = None
        self._running = False
        self._muted = False
        self._thread: threading.Thread | None = None
        self._audio_queue: queue.Queue[bytes] = queue.Queue()

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        if status:
            LOGGER.warning(f"Audio input status: {status}")

        audio_bytes = (indata * 32767).astype(np.int16).tobytes()
        self._audio_queue.put(audio_bytes)

    def _broadcast_thread(self) -> None:
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=0.1)
                if self._muted:
                    audio = bytes(len(audio))
                self.broadcaster.broadcast(audio)
            except queue.Empty:
                continue
            except Exception as e:
                LOGGER.error(f"Broadcast error: {e}")

    def start(self) -> None:
        if self._running:
            return

        self._running = True

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
            blocksize=self.chunk_samples,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._broadcast_thread, daemon=True)
        self._thread.start()

        LOGGER.info(f"Microphone input started (sample_rate={self.sample_rate}, channels={self.channels})")

    def stop(self) -> None:
        self._running = False

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        self.broadcaster.clear()
        LOGGER.info("Microphone input stopped")

    def subscribe(self, subscriber: Any) -> None:
        self.broadcaster.subscribe(subscriber)

    def unsubscribe(self, subscriber: Any) -> None:
        self.broadcaster.unsubscribe(subscriber)

    def toggle_mute(self) -> bool:
        self._muted = not self._muted
        LOGGER.info(f"Microphone {'muted' if self._muted else 'unmuted'}")
        return self._muted

    def is_muted(self) -> bool:
        return self._muted
