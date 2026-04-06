from collections.abc import Generator
import queue
import threading

import numpy as np
import sounddevice as sd

from constellation.logger import LOGGER


OUTPUT_SAMPLE_RATE = 24000
CHANNELS = 1


class SpeakerOutput:
    def __init__(self, sample_rate: int = OUTPUT_SAMPLE_RATE, channels: int = CHANNELS):
        self.sample_rate = sample_rate
        self.channels = channels

        self._stream: sd.OutputStream | None = None
        self._audio_queue: queue.Queue[bytes | None] = queue.Queue()
        self._running = False
        self._thread: threading.Thread | None = None
        self._interrupted = threading.Event()

    def _playback_thread(self) -> None:
        while self._running:
            try:
                audio = self._audio_queue.get(timeout=0.1)
                if audio is None:
                    continue

                if self._interrupted.is_set():
                    continue

                audio_array = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32767.0

                if self._stream and not self._interrupted.is_set():
                    self._stream.write(audio_array)

            except queue.Empty:
                continue
            except Exception as e:
                if self._running:
                    LOGGER.error(f"Playback error: {e}")

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._interrupted.clear()

        self._stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=np.float32,
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._playback_thread, daemon=True)
        self._thread.start()

        LOGGER.info(f"Speaker output started (sample_rate={self.sample_rate})")

    def stop(self) -> None:
        self._running = False
        self._interrupted.set()

        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None

        self._clear_queue()
        LOGGER.info("Speaker output stopped")

    def play(self, audio: bytes) -> None:
        if not self._interrupted.is_set():
            self._audio_queue.put(audio)

    def play_stream(self, audio_stream: Generator[bytes, None, None]) -> None:
        for chunk in audio_stream:
            if self._interrupted.is_set():
                break
            self.play(chunk)

    def interrupt(self) -> None:
        self._interrupted.set()
        self._clear_queue()

        if self._stream:
            try:
                self._stream.stop()
                self._stream.start()
            except Exception as e:
                LOGGER.warning(f"Error restarting stream on interrupt: {e}")

        self._interrupted.clear()

    def _clear_queue(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    def is_playing(self) -> bool:
        return not self._audio_queue.empty()
