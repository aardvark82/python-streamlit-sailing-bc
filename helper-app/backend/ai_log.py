"""Unified AI call ledger.

Records every AI request the helper-app makes — OpenAI or Ollama —
with provider, source-data label, model, tokens, processing time, and
cost (Ollama = $0). Persists to /data/ai_log.json. Backwards-compatible
with the older /data/openai_log.json on first read.
"""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

import pytz

from .envutil import getenv_ci

VAN_TZ = pytz.timezone("America/Vancouver")
_DATA_DIR = Path(getenv_ci("HELPER_DATA_DIR", "/data"))
_PATH = _DATA_DIR / "ai_log.json"
_LEGACY_PATH = _DATA_DIR / "openai_log.json"
_lock = threading.Lock()
_MAX_ENTRIES = 1000

# USD per 1M tokens (OpenAI only — local models are $0).
PRICING = {
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
    "gpt-4o":     {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}


def _load() -> list:
    if _PATH.exists():
        try:
            return json.loads(_PATH.read_text() or "[]")
        except json.JSONDecodeError:
            pass
    # First run after upgrade — migrate any legacy openai_log.json entries
    if _LEGACY_PATH.exists():
        try:
            legacy = json.loads(_LEGACY_PATH.read_text() or "[]")
            for e in legacy:
                e.setdefault("provider", "openai")
                e.setdefault("source_data", e.get("reason", "—"))
                e.setdefault("elapsed_sec", None)
            return legacy
        except Exception:
            pass
    return []


def _save(entries: list):
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _PATH.write_text(json.dumps(entries[-_MAX_ENTRIES:], indent=2))


def cost_for(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
    if provider != "openai":
        return 0.0
    p = PRICING.get(model, PRICING["gpt-5-mini"])
    return prompt_tokens / 1e6 * p["in"] + completion_tokens / 1e6 * p["out"]


def record(*, reason: str, provider: str, model: str,
           prompt_tokens: int, completion_tokens: int,
           elapsed_sec: float | None = None,
           source_data: str | None = None):
    """Append one AI call. source_data = human label for what was processed
    (e.g. 'Marine forecast HTML', 'Wave table HTML'). reason groups calls
    (e.g. 'forecast parsing (Howe Sound)')."""
    cost = cost_for(provider, model, prompt_tokens, completion_tokens)
    entry = {
        "ts": datetime.now(VAN_TZ).isoformat(timespec="seconds"),
        "reason": reason,
        "provider": provider,
        "model": model,
        "source_data": source_data or reason,
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "total_tokens": int(prompt_tokens + completion_tokens),
        "elapsed_sec": round(float(elapsed_sec), 2) if elapsed_sec is not None else None,
        "cost_usd": round(cost, 6),
    }
    with _lock:
        entries = _load()
        entries.append(entry)
        _save(entries)
    return entry


def reset() -> int:
    with _lock:
        n = len(_load())
        _save([])
        # Also clear legacy file so it doesn't get re-migrated
        try:
            if _LEGACY_PATH.exists():
                _LEGACY_PATH.unlink()
        except OSError:
            pass
    return n


def snapshot(limit: int = 200) -> dict:
    with _lock:
        entries = _load()

    total_tokens = sum(e.get("total_tokens", 0) for e in entries)
    total_cost = sum(e.get("cost_usd", 0.0) for e in entries)

    # Monthly projection from observed span
    est_monthly = None
    if entries:
        try:
            first = datetime.fromisoformat(entries[0]["ts"])
            now = datetime.now(VAN_TZ)
            span_days = max((now - first).total_seconds() / 86400.0, 1 / 24.0)
            est_monthly = total_cost / span_days * 30.0
        except Exception:
            est_monthly = None

    # Aggregates
    by_reason = {}
    by_provider = {}
    for e in entries:
        r = e.get("reason", "unknown")
        b = by_reason.setdefault(r, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        b["calls"] += 1
        b["tokens"] += e.get("total_tokens", 0)
        b["cost_usd"] += e.get("cost_usd", 0.0)

        p = e.get("provider", "openai")
        bp = by_provider.setdefault(p, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        bp["calls"] += 1
        bp["tokens"] += e.get("total_tokens", 0)
        bp["cost_usd"] += e.get("cost_usd", 0.0)
    for d in (by_reason, by_provider):
        for b in d.values():
            b["cost_usd"] = round(b["cost_usd"], 4)

    return {
        "total_calls": len(entries),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "est_monthly_cost_usd": round(est_monthly, 4) if est_monthly is not None else None,
        "by_reason": by_reason,
        "by_provider": by_provider,
        "entries": list(reversed(entries))[:limit],
    }
