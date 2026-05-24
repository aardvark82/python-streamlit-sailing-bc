"""Trend analysis: average wind/wave over last N days for three time-of-day
windows (morning 8-12, afternoon 12-16, evening 16-20).

Useful for answering "when is the typical best time to go out" — a low
wind average in the morning window over a 14-day window suggests calmer
mornings are the norm.
"""
from __future__ import annotations

from statistics import mean
from typing import Iterable

WINDOWS = {
    "morning":   (8, 12),
    "afternoon": (12, 16),
    "evening":   (16, 20),
}


def _bucket(hour: int) -> str | None:
    for name, (lo, hi) in WINDOWS.items():
        if lo <= hour < hi:
            return name
    return None


def summarize(readings: Iterable[dict]) -> dict:
    """Given history rows (timestamp tz-aware), produce per-window stats."""
    buckets: dict[str, dict[str, list]] = {
        name: {"wind": [], "wave": []} for name in WINDOWS
    }
    for r in readings:
        ts = r["timestamp"]
        b = _bucket(ts.hour)
        if not b:
            continue
        if r.get("wind_speed") is not None:
            buckets[b]["wind"].append(float(r["wind_speed"]))
        if r.get("wave_height") is not None:
            buckets[b]["wave"].append(float(r["wave_height"]))

    out = {}
    for name, vals in buckets.items():
        out[name] = {
            "samples": len(vals["wind"]),
            "wind_avg_kts": round(mean(vals["wind"]), 1) if vals["wind"] else None,
            "wind_max_kts": max(vals["wind"]) if vals["wind"] else None,
            "wave_avg_cm": round(mean(vals["wave"]) * 100, 0) if vals["wave"] else None,
            "wave_max_cm": round(max(vals["wave"]) * 100, 0) if vals["wave"] else None,
        }
    # Best window = lowest wind avg (among windows with samples)
    rated = [(n, s["wind_avg_kts"]) for n, s in out.items() if s["wind_avg_kts"] is not None]
    out["best_window"] = min(rated, key=lambda x: x[1])[0] if rated else None
    return out
