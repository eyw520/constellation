from collections.abc import Callable
from typing import Protocol, runtime_checkable

from constellation.logger import LOGGER


@runtime_checkable
class AudioSubscriber(Protocol):
    def on_audio(self, data: bytes) -> None: ...


AudioCallback = Callable[[bytes], None]


class AudioBroadcaster:
    def __init__(self) -> None:
        self._subscribers: list[AudioSubscriber | AudioCallback] = []

    def subscribe(self, subscriber: AudioSubscriber | AudioCallback) -> None:
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: AudioSubscriber | AudioCallback) -> None:
        try:
            self._subscribers.remove(subscriber)
        except ValueError:
            pass

    def broadcast(self, data: bytes) -> None:
        for subscriber in self._subscribers:
            try:
                if callable(subscriber) and not isinstance(subscriber, AudioSubscriber):
                    subscriber(data)
                else:
                    subscriber.on_audio(data)
            except Exception as e:
                LOGGER.error(f"Audio subscriber failed: {e}")

    def clear(self) -> None:
        self._subscribers.clear()
