from pathlib import Path
from typing import Any

import yaml

from constellation.logger import LOGGER
from constellation.models.config import AgentConfig


def load_agent_config(path: str | Path) -> AgentConfig:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Empty agent config: {path}")

    config = AgentConfig(**data)
    LOGGER.info(f"Loaded agent config from {path}")
    return config


def load_agent_config_from_dict(data: dict[str, Any]) -> AgentConfig:
    return AgentConfig(**data)
