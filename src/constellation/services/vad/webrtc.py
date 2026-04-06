from collections import deque
from enum import IntEnum
import struct

from constellation.logger import LOGGER


class VADState(IntEnum):
    SILENCE = 0
    SPEECH = 1


class WebRTCVAD:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.02):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.state = VADState.SILENCE

        self._speech_frames = 0
        self._silence_frames = 0
        self._speech_threshold = 3
        self._silence_threshold = 15

        self._energy_history: deque[float] = deque(maxlen=50)
        self._adaptive_threshold = threshold

    def _calculate_rms(self, audio: bytes) -> float:
        try:
            num_samples = len(audio) // 2
            if num_samples == 0:
                return 0.0

            samples = struct.unpack(f"<{num_samples}h", audio)
            sum_squares = sum(s * s for s in samples)
            rms = (sum_squares / num_samples) ** 0.5
            return rms / 32768.0
        except Exception as e:
            LOGGER.error(f"RMS calculation failed: {e}")
            return 0.0

    def _update_adaptive_threshold(self, energy: float) -> None:
        self._energy_history.append(energy)
        if len(self._energy_history) >= 10:
            avg_energy = sum(self._energy_history) / len(self._energy_history)
            self._adaptive_threshold = max(self.threshold, avg_energy * 1.5)

    def process(self, audio: bytes) -> VADState:
        try:
            energy = self._calculate_rms(audio)
            self._update_adaptive_threshold(energy)

            is_speech = energy > self._adaptive_threshold

            if is_speech:
                self._speech_frames += 1
                self._silence_frames = 0
                if self._speech_frames >= self._speech_threshold:
                    self.state = VADState.SPEECH
            else:
                self._silence_frames += 1
                self._speech_frames = 0
                if self._silence_frames >= self._silence_threshold:
                    self.state = VADState.SILENCE

            return self.state
        except Exception as e:
            LOGGER.error(f"VAD processing failed: {e}")
            return self.state

    def reset(self) -> None:
        self.state = VADState.SILENCE
        self._speech_frames = 0
        self._silence_frames = 0
        self._energy_history.clear()

    def is_speaking(self) -> bool:
        return self.state == VADState.SPEECH
