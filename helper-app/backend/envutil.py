"""Case-insensitive env var lookup.

Lets users write `cloudflare_api_token=...` or `Cloudflare_Api_Token=...`
in their .env file and have it picked up regardless of case.
"""
from __future__ import annotations

import os
from typing import Optional


def getenv_ci(name: str, default: Optional[str] = None) -> Optional[str]:
    """Look up env var by name, ignoring case."""
    # Fast path: exact match
    if name in os.environ:
        return os.environ[name]
    target = name.lower()
    for k, v in os.environ.items():
        if k.lower() == target:
            return v
    return default
