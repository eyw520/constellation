from typing import Any, Generic, TypeVar

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


T = TypeVar("T")


class LazySettings(Generic[T]):
    """Lazy initialization wrapper for settings classes.

    Defers instantiation until first attribute access, allowing settings
    to be imported without immediately requiring environment variables.
    """

    def __init__(self, settings_class: type[T]) -> None:
        object.__setattr__(self, "_settings_class", settings_class)
        object.__setattr__(self, "_instance", None)

    def _get_instance(self) -> T:
        instance = object.__getattribute__(self, "_instance")
        if instance is None:
            settings_class = object.__getattribute__(self, "_settings_class")
            instance = settings_class()
            object.__setattr__(self, "_instance", instance)
        return instance

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get_instance(), name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._get_instance(), name, value)


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    OPENAI_API_KEY: str | None = Field(default=None)
    ANTHROPIC_API_KEY: str | None = Field(default=None)
    DEEPGRAM_API_KEY: str | None = Field(default=None)

    def require_openai_api_key(self) -> str:
        if not self.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return self.OPENAI_API_KEY

    def require_anthropic_api_key(self) -> str:
        if not self.ANTHROPIC_API_KEY:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return self.ANTHROPIC_API_KEY

    def require_deepgram_api_key(self) -> str:
        if not self.DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY environment variable is required")
        return self.DEEPGRAM_API_KEY


SETTINGS: Settings = LazySettings(Settings)  # type: ignore
