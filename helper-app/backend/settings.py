"""Tiny JSON-file settings store. Only secret stored today is the
OpenAI API key; the file structure stays open for future fields."""
from __future__ import annotations

import json
import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("HELPER_DATA_DIR", "/data"))
SETTINGS_PATH = DATA_DIR / "settings.json"


def _ensure():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not SETTINGS_PATH.exists():
        SETTINGS_PATH.write_text(json.dumps({}))


def load() -> dict:
    _ensure()
    try:
        return json.loads(SETTINGS_PATH.read_text() or "{}")
    except json.JSONDecodeError:
        return {}


def save(updates: dict) -> dict:
    _ensure()
    current = load()
    current.update(updates)
    SETTINGS_PATH.write_text(json.dumps(current, indent=2))
    return current


def get_openai_key() -> str | None:
    # Env var wins if set (allows ops to inject without touching disk)
    return os.environ.get("OPENAI_API_KEY") or load().get("openai_api_key")
