"""Alexa custom-skill endpoint.

Speaks current wind/wave for English Bay + Howe Sound (Pam Rocks),
the marine-forecast outlook for later today, and an overall Go/No-Go
verdict. Current conditions come from local SQLite; the outlook +
verdict come from the OpenAI-parsed weather.gc.ca marine forecast.

Go/No-Go thresholds mirror the main app's fetch_gonogo.py so the
voice verdict agrees with the Streamlit dashboard.
"""
from __future__ import annotations

import logging

from . import db, forecast

log = logging.getLogger("helper.alexa")

# Thresholds (knots / meters) — copied from fetch_gonogo.py
WIND_GO = 10
WIND_CAUTION = 15
WAVE_GO = 0.51
WAVE_CAUTION = 0.75

ENGLISH_BAY = "46304"
HOWE_SOUND = "WAS"      # Pam Rocks = Howe Sound proxy

_ORDER = {"go": 0, "caution": 1, "nogo": 2}

# Speech-friendly direction (handles both compass abbreviations from buoys
# and full '…ERLY' words from the parsed forecast).
_DIR_SPEAK = {
    "N": "northerly", "NNE": "north-northeasterly", "NE": "northeasterly",
    "ENE": "east-northeasterly", "E": "easterly", "ESE": "east-southeasterly",
    "SE": "southeasterly", "SSE": "south-southeasterly", "S": "southerly",
    "SSW": "south-southwesterly", "SW": "southwesterly", "WSW": "west-southwesterly",
    "W": "westerly", "WNW": "west-northwesterly", "NW": "northwesterly",
    "NNW": "north-northwesterly",
}


def _speak_dir(d):
    if not d:
        return ""
    d = d.strip().upper()
    if d in ("", "N/A", "VARIABLE", "V"):
        return ""
    return _DIR_SPEAK.get(d, d.lower())


def _latest(buoy_id):
    rows = db.read_history(buoy_id, days_back=1) or db.read_history(buoy_id, days_back=3)
    return rows[-1] if rows else None


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

    hs_status = None
    if hs and hs.get("wind_speed") is not None:
        parts.append(f"Howe Sound at Pam Rocks: wind {round(hs['wind_speed'])} knots.")
        hs_status = _verdict(hs["wind_speed"], hs.get("wave_height"))
    else:
        parts.append("Howe Sound data is currently unavailable.")

    # Forecast-driven outlook + verdict (later-in-day conditions from the
    # weather.gc.ca marine forecast, parsed by OpenAI). CACHED-ONLY here:
    # Alexa enforces an ~8s response budget and a cold OpenAI parse can
    # blow past it. The hourly cron warms the cache; if it isn't warm we
    # fall back to the current-conditions verdict so we never time out.
    fc = forecast.gonogo_from_forecast("howe_sound", allow_fetch=False)
    if fc.get("error"):
        # Forecast not cached yet → use live buoy verdicts so the briefing
        # still ends with a Go/No-Go.
        if hs_status:
            statuses.append(hs_status)
    else:
        if fc.get("driving_period") and fc.get("driving_wind_kts") is not None:
            period = (fc["driving_period"] or "").strip().lower()
            dir_phrase = _speak_dir(fc.get("driving_dir"))
            dir_phrase = f" {dir_phrase}" if dir_phrase else ""
            when = "" if period in ("", "now") else f", {period}"
            parts.append(f"Howe Sound forecast: peak wind {round(fc['driving_wind_kts'])} knots"
                         f"{dir_phrase}{when}.")
        if fc.get("strong_wind_warning"):
            parts.append("A strong wind warning is in effect.")
        elif fc.get("wind_warning"):
            parts.append("A wind warning is in effect.")
        statuses.append(fc["status"])

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
