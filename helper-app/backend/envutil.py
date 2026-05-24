"""Case-insensitive env var lookup.

Lets users write `cloudflare_api_token=...` or `Cloudflare_Api_Token=...`
in their .env file and have it picked up regardless of case.
"""
from __future__ import annotations

import os
from typing import Optional


# Common aliases — if the canonical name isn't found, also try these.
# Lets users use the same key names as the main Streamlit secrets.toml.
ALIASES = {
    "OPENAI_API_KEY": ["OPENAI_KEY", "OPENAI", "OpenAI_key", "openai_key"],
}


def _strip_value(v: Optional[str]) -> Optional[str]:
    """Trim whitespace + surrounding quotes (handles `KEY = "value"` style)."""
    if v is None:
        return None
    v = v.strip()
    if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
        v = v[1:-1].strip()
    return v or None


def getenv_ci(name: str, default: Optional[str] = None) -> Optional[str]:
    """Look up env var by name, ignoring case. Falls back to known aliases."""
    candidates = [name] + ALIASES.get(name, [])
    lower_map = {k.lower(): v for k, v in os.environ.items()}
    for cand in candidates:
        if cand in os.environ:
            r = _strip_value(os.environ[cand])
            if r:
                return r
        r = _strip_value(lower_map.get(cand.lower()))
        if r:
            return r
    return default
