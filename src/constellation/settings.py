from functools import lru_cache
import os


class Settings:
    @property
    def openai_api_key(self) -> str | None:
        return os.getenv("OPENAI_API_KEY")

    @property
    def anthropic_api_key(self) -> str | None:
        return os.getenv("ANTHROPIC_API_KEY")

    @property
    def deepgram_api_key(self) -> str | None:
        return os.getenv("DEEPGRAM_API_KEY")

    def require_openai_api_key(self) -> str:
        key = self.openai_api_key
        if not key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        return key

    def require_anthropic_api_key(self) -> str:
        key = self.anthropic_api_key
        if not key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        return key

    def require_deepgram_api_key(self) -> str:
        key = self.deepgram_api_key
        if not key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is required")
        return key


@lru_cache
def get_settings() -> Settings:
    return Settings()


SETTINGS = get_settings()
