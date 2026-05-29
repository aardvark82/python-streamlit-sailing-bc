"""Alexa custom-skill endpoint.

Speaks current wind/wave for English Bay + Howe Sound (Pam Rocks),
a 3-hour wind trend projection, and an overall Go/No-Go verdict.
All data comes from the local SQLite store (instant, no KV reads).

Go/No-Go thresholds mirror the main app's fetch_gonogo.py so the
voice verdict agrees with the Streamlit dashboard.
"""
from __future__ import annotations

import logging
from datetime import timedelta

import numpy as np

from . import db

log = logging.getLogger("helper.alexa")

# Thresholds (knots / meters) — copied from fetch_gonogo.py
WIND_GO = 10
WIND_CAUTION = 15
WAVE_GO = 0.51
WAVE_CAUTION = 0.75

ENGLISH_BAY = "46304"
HOWE_SOUND = "WAS"      # Pam Rocks = Howe Sound proxy

_ORDER = {"go": 0, "caution": 1, "nogo": 2}


def _latest(buoy_id):
    rows = db.read_history(buoy_id, days_back=1) or db.read_history(buoy_id, days_back=3)
    return rows[-1] if rows else None


def _wind_trend_3h(buoy_id):
    """Linear-fit wind over the last 3h, project 3h ahead.
    Returns (projected_kts, slope_kts_per_h) or None."""
    rows = db.read_history(buoy_id, days_back=1)
    rows = [r for r in rows if r.get("wind_speed") is not None]
    if len(rows) < 3:
        return None
    last_t = rows[-1]["timestamp"]
    recent = [r for r in rows if r["timestamp"] >= last_t - timedelta(hours=3)]
    if len(recent) < 3:
        return None
    t0 = recent[0]["timestamp"]
    xs = np.array([(r["timestamp"] - t0).total_seconds() / 3600.0 for r in recent])
    ys = np.array([float(r["wind_speed"]) for r in recent])
    if xs.std() == 0:
        return None
    slope, intercept = np.polyfit(xs, ys, 1)
    proj = slope * (xs[-1] + 3) + intercept
    return max(0.0, float(proj)), float(slope)


def _verdict(wind, wave_m):
    status = "go"
    if wind is not None:
        if wind > WIND_CAUTION:
            return "nogo"
        if wind > WIND_GO:
            status = "caution"
    if wave_m is not None:
        if wave_m > WAVE_CAUTION:
            return "nogo"
        if wave_m > WAVE_GO and status == "go":
            status = "caution"
    return status


def build_speech() -> str:
    eb = _latest(ENGLISH_BAY)
    hs = _latest(HOWE_SOUND)
    parts = ["Here are the current sailing conditions."]
    statuses = []

    if eb and eb.get("wind_speed") is not None:
        s = f"English Bay: wind {round(eb['wind_speed'])} knots"
        if eb.get("wave_height") is not None:
            s += f", waves {round(eb['wave_height'] * 100)} centimeters"
        parts.append(s + ".")
        statuses.append(_verdict(eb["wind_speed"], eb.get("wave_height")))
    else:
        parts.append("English Bay data is currently unavailable.")

    if hs and hs.get("wind_speed") is not None:
        parts.append(f"Howe Sound at Pam Rocks: wind {round(hs['wind_speed'])} knots.")
        statuses.append(_verdict(hs["wind_speed"], hs.get("wave_height")))
    else:
        parts.append("Howe Sound data is currently unavailable.")

    tr = _wind_trend_3h(ENGLISH_BAY)
    if tr:
        proj, slope = tr
        if slope > 0.3:
            parts.append(f"Over the next 3 hours, wind is expected to rise to about {round(proj)} knots.")
        elif slope < -0.3:
            parts.append(f"Over the next 3 hours, wind is expected to ease to about {round(proj)} knots.")
        else:
            parts.append(f"Over the next 3 hours, wind should hold near {round(proj)} knots.")

    if statuses:
        overall = max(statuses, key=lambda s: _ORDER[s])
        verdict_text = {
            "go": "Go. Conditions look good for the boat.",
            "caution": "Caution. Conditions are marginal — check carefully.",
            "nogo": "No go. Conditions are not recommended.",
        }[overall]
        parts.append("Overall status: " + verdict_text)

    return " ".join(parts)


def handle_request(body: dict) -> dict:
    """Return an Alexa response envelope. We treat LaunchRequest and any
    IntentRequest identically — just speak the conditions briefing."""
    speech = build_speech()
    return {
        "version": "1.0",
        "response": {
            "outputSpeech": {"type": "PlainText", "text": speech},
            "card": {"type": "Simple", "title": "Sailing BC", "content": speech},
            "shouldEndSession": True,
        },
    }
