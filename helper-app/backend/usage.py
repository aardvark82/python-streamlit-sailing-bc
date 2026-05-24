"""Local CF KV usage tracking.

Cloudflare doesn't expose real-time KV usage via a simple REST endpoint
(the analytics GraphQL needs a separate Account Analytics:Read scope and
lags by minutes). So we instrument our own writes/reads/lists and
persist daily + monthly totals to /data/usage.json.

Numbers shown are 'usage consumed by this helper-app' — which is the
overwhelming majority of traffic since the Streamlit UI is only used
interactively.
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

import pytz

from .envutil import getenv_ci

VAN_TZ = pytz.timezone("America/Vancouver")

# Cloudflare Workers KV — Free plan limits (per docs, as of 2024-2025)
FREE_LIMITS = {
    "reads_per_day":   100_000,
    "writes_per_day":  1_000,
    "deletes_per_day": 1_000,
    "lists_per_day":   1_000,
    "storage_gb":      1,
}

_DATA_DIR = Path(getenv_ci("HELPER_DATA_DIR", "/data"))
_USAGE_PATH = _DATA_DIR / "usage.json"
_lock = threading.Lock()


def _load() -> dict:
    if not _USAGE_PATH.exists():
        return {"daily": {}, "monthly": {}}
    try:
        return json.loads(_USAGE_PATH.read_text() or "{}") or {"daily": {}, "monthly": {}}
    except json.JSONDecodeError:
        return {"daily": {}, "monthly": {}}


def _save(d: dict):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _USAGE_PATH.write_text(json.dumps(d, indent=2))


def _keys_today_month():
    now = datetime.now(VAN_TZ)
    return now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")


def record(op: str, n: int = 1):
    """op ∈ {'read', 'write', 'list', 'delete'}"""
    field = f"{op}s"
    day_key, month_key = _keys_today_month()
    with _lock:
        d = _load()
        d.setdefault("daily", {}).setdefault(day_key, {})[field] = \
            d["daily"][day_key].get(field, 0) + n
        d.setdefault("monthly", {}).setdefault(month_key, {})[field] = \
            d["monthly"][month_key].get(field, 0) + n
        _save(d)


def snapshot() -> dict:
    """Return today, this-month, and limit-comparison %."""
    day_key, month_key = _keys_today_month()
    d = _load()
    today = d.get("daily", {}).get(day_key, {})
    month = d.get("monthly", {}).get(month_key, {})

    def pct(used: int, limit: int) -> float | None:
        if not limit:
            return None
        return round(100.0 * used / limit, 1)

    return {
        "as_of": datetime.now(VAN_TZ).isoformat(timespec="seconds"),
        "today": {
            "date": day_key,
            "reads":  today.get("reads", 0),
            "writes": today.get("writes", 0),
            "lists":  today.get("lists", 0),
            "deletes": today.get("deletes", 0),
            "reads_pct":  pct(today.get("reads", 0),  FREE_LIMITS["reads_per_day"]),
            "writes_pct": pct(today.get("writes", 0), FREE_LIMITS["writes_per_day"]),
            "lists_pct":  pct(today.get("lists", 0),  FREE_LIMITS["lists_per_day"]),
        },
        "this_month": {
            "month": month_key,
            "reads":  month.get("reads", 0),
            "writes": month.get("writes", 0),
            "lists":  month.get("lists", 0),
            "deletes": month.get("deletes", 0),
        },
        "limits": FREE_LIMITS,
        "note": "Locally instrumented — counts CF KV operations issued by this helper-app only (not Streamlit app traffic).",
    }
