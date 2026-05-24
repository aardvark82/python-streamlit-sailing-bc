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

import calendar
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


def _project_day(used: int, hours_elapsed: float) -> int:
    """Linear extrapolation to end-of-day. Floor at ~30 min of data to
    avoid wild numbers right after midnight."""
    h = max(0.5, hours_elapsed)
    return int(round(used * 24.0 / h))


def _project_month(used: int, days_elapsed: float, days_in_month: int) -> int:
    d = max(0.25, days_elapsed)
    return int(round(used * days_in_month / d))


def snapshot() -> dict:
    """Return today, this-month, projections, and limit-comparison %."""
    now = datetime.now(VAN_TZ)
    day_key, month_key = now.strftime("%Y-%m-%d"), now.strftime("%Y-%m")
    d = _load()
    today = d.get("daily", {}).get(day_key, {})
    month = d.get("monthly", {}).get(month_key, {})

    # Elapsed within current windows (Vancouver)
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    hours_today = (now - midnight).total_seconds() / 3600.0
    days_in_month = calendar.monthrange(now.year, now.month)[1]
    days_elapsed = (now.day - 1) + (hours_today / 24.0)

    # Free plan has no published monthly cap on Workers KV — derive an
    # implied monthly ceiling from the daily limit so projections have
    # something to compare against. (writes 1000/day → ~30k/mo etc.)
    monthly_limits = {
        "reads_per_month":   FREE_LIMITS["reads_per_day"]   * days_in_month,
        "writes_per_month":  FREE_LIMITS["writes_per_day"]  * days_in_month,
        "lists_per_month":   FREE_LIMITS["lists_per_day"]   * days_in_month,
    }

    def pct(used: int, limit: int) -> float | None:
        return round(100.0 * used / limit, 1) if limit else None

    def build(today_field: str, daily_limit: int, monthly_limit: int) -> dict:
        used_today = today.get(today_field, 0)
        used_month = month.get(today_field, 0)
        proj_day = _project_day(used_today, hours_today)
        proj_month = _project_month(used_month, days_elapsed, days_in_month)
        return {
            "today_used": used_today,
            "today_projected": proj_day,
            "today_limit": daily_limit,
            "today_pct": pct(used_today, daily_limit),
            "today_projected_pct": pct(proj_day, daily_limit),
            "today_will_exceed": proj_day > daily_limit,
            "month_used": used_month,
            "month_projected": proj_month,
            "month_limit": monthly_limit,
            "month_pct": pct(used_month, monthly_limit),
            "month_projected_pct": pct(proj_month, monthly_limit),
            "month_will_exceed": proj_month > monthly_limit,
        }

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "today_date": day_key,
        "month": month_key,
        "hours_into_day": round(hours_today, 1),
        "days_into_month": round(days_elapsed, 2),
        "days_in_month": days_in_month,
        "reads":  build("reads",  FREE_LIMITS["reads_per_day"],  monthly_limits["reads_per_month"]),
        "writes": build("writes", FREE_LIMITS["writes_per_day"], monthly_limits["writes_per_month"]),
        "lists":  build("lists",  FREE_LIMITS["lists_per_day"],  monthly_limits["lists_per_month"]),
        "limits_daily": FREE_LIMITS,
        "limits_monthly": monthly_limits,
        "note": "Locally instrumented (helper-app only). Projection = linear extrapolation of current rate to end-of-day / end-of-month.",
    }
