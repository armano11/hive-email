import logging
import os
import yaml
from pathlib import Path
from typing import Any
from dotenv import load_dotenv


def setup_logging(config: dict[str, Any]) -> None:
    level = config.get("logging", {}).get("level", "INFO")
    fmt = config.get("logging", {}).get(
        "format", "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    )
    logging.basicConfig(level=getattr(logging, level), format=fmt)


def load_config(path: str = "configs/config.yaml") -> dict[str, Any]:
    with open(path) as f:
        return yaml.safe_load(f)


def load_env() -> None:
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
    load_dotenv()


def get_api_key(provider: str) -> str | None:
    key_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "gemini": "GOOGLE_API_KEY",
        "nvidia": "NVIDIA_API_KEY",
    }
    return os.getenv(key_map.get(provider, ""))
