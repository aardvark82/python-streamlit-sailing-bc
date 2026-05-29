"""OpenAI call ledger.

Records every OpenAI request the helper-app makes (currently just marine
forecast parsing) with tokens, computed cost, and a human reason.
Persists to /data/openai_log.json. Powers the Settings → OpenAI Log panel.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import pytz

from .envutil import getenv_ci

VAN_TZ = pytz.timezone("America/Vancouver")
_PATH = Path(getenv_ci("HELPER_DATA_DIR", "/data")) / "openai_log.json"
_lock = threading.Lock()
_MAX_ENTRIES = 1000

# USD per 1M tokens. Update if you switch models.
PRICING = {
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
    "gpt-4o":     {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}


def _load() -> list:
    if not _PATH.exists():
        return []
    try:
        return json.loads(_PATH.read_text() or "[]")
    except json.JSONDecodeError:
        return []


def _save(entries: list):
    _PATH.parent.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(entries[-_MAX_ENTRIES:], indent=2))


def cost_for(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    p = PRICING.get(model, PRICING["gpt-5-mini"])
    return prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"]


def record(reason: str, model: str, prompt_tokens: int, completion_tokens: int):
    cost = cost_for(model, prompt_tokens, completion_tokens)
    entry = {
        "ts": datetime.now(VAN_TZ).isoformat(timespec="seconds"),
        "reason": reason,
        "model": model,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
        "cost_usd": round(cost, 6),
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        _save(entries)
    return entry


def snapshot(limit: int = 100) -> dict:
    with _lock:
        entries = _load()

    total_tokens = sum(e.get("total_tokens", 0) for e in entries)
    total_cost = sum(e.get("cost_usd", 0.0) for e in entries)

    # Monthly projection from the observed span
    est_monthly = None
    if entries:
        try:
            first = datetime.fromisoformat(entries[0]["ts"])
            now = datetime.now(VAN_TZ)
            span_days = max((now - first).total_seconds() / 86400.0, 1 / 24.0)  # ≥1h
            est_monthly = total_cost / span_days * 30.0
        except Exception:
            est_monthly = None

    # Aggregate by reason
    by_reason = {}
    for e in entries:
        r = e.get("reason", "unknown")
        b = by_reason.setdefault(r, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        b["calls"] += 1
        b["tokens"] += e.get("total_tokens", 0)
        b["cost_usd"] += e.get("cost_usd", 0.0)
    for b in by_reason.values():
        b["cost_usd"] = round(b["cost_usd"], 4)

    return {
        "total_calls": len(entries),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "est_monthly_cost_usd": round(est_monthly, 4) if est_monthly is not None else None,
        "by_reason": by_reason,
        "entries": list(reversed(entries))[:limit],  # newest first
    }
