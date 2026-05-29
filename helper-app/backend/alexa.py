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


def _gather() -> dict:
    """Collect everything once — shared by speech + APL so they never
    disagree."""
    eb = _latest(ENGLISH_BAY)
    hs = _latest(HOWE_SOUND)
    statuses = []

    eb_d = None
    if eb and eb.get("wind_speed") is not None:
        eb_d = {"wind": round(eb["wind_speed"]),
                "wave_cm": round(eb["wave_height"] * 100) if eb.get("wave_height") is not None else None}
        statuses.append(_verdict(eb["wind_speed"], eb.get("wave_height")))

    hs_d = None
    hs_status = None
    if hs and hs.get("wind_speed") is not None:
        hs_d = {"wind": round(hs["wind_speed"])}
        hs_status = _verdict(hs["wind_speed"], hs.get("wave_height"))

    fc = forecast.gonogo_from_forecast("howe_sound", allow_fetch=False)
    fc_d = None
    if fc.get("error"):
        if hs_status:
            statuses.append(hs_status)
    else:
        fc_d = {
            "status": fc["status"],
            "wind": round(fc["driving_wind_kts"]) if fc.get("driving_wind_kts") is not None else None,
            "period": (fc.get("driving_period") or "").strip(),
            "dir": fc.get("driving_dir") or "",
            "wind_warning": fc.get("wind_warning"),
            "strong_wind_warning": fc.get("strong_wind_warning"),
        }
        statuses.append(fc["status"])

    overall = max(statuses, key=lambda s: _ORDER[s]) if statuses else None
    return {"english_bay": eb_d, "pam_rocks": hs_d, "forecast": fc_d, "overall": overall}


def build_speech(data: dict | None = None) -> str:
    data = data or _gather()
    parts = ["Here are the current sailing conditions."]

    eb = data["english_bay"]
    if eb:
        s = f"English Bay: wind {eb['wind']} knots"
        if eb["wave_cm"] is not None:
            s += f", waves {eb['wave_cm']} centimeters"
        parts.append(s + ".")
    else:
        parts.append("English Bay data is currently unavailable.")

    hs = data["pam_rocks"]
    if hs:
        parts.append(f"Howe Sound at Pam Rocks: wind {hs['wind']} knots.")
    else:
        parts.append("Howe Sound data is currently unavailable.")

    fc = data["forecast"]
    if fc and fc["wind"] is not None:
        period = (fc["period"] or "").lower()
        dir_phrase = _speak_dir(fc["dir"])
        dir_phrase = f" {dir_phrase}" if dir_phrase else ""
        when = "" if period in ("", "now") else f", {period}"
        parts.append(f"Howe Sound forecast: peak wind {fc['wind']} knots{dir_phrase}{when}.")
        if fc["strong_wind_warning"]:
            parts.append("A strong wind warning is in effect.")
        elif fc["wind_warning"]:
            parts.append("A wind warning is in effect.")

    if data["overall"]:
        verdict_text = {
            "go": "Go. Conditions look good for the boat.",
            "caution": "Caution. Conditions are marginal — check carefully.",
            "nogo": "No go. Conditions are not recommended.",
        }[data["overall"]]
        parts.append("Overall status: " + verdict_text)

    return " ".join(parts)


# ── APL visual (Echo Show) ─────────────────────────────────────────────

_STATUS_COLOR = {"go": "#16a34a", "caution": "#d97706", "nogo": "#dc2626"}
_STATUS_LABEL = {"go": "GO", "caution": "CAUTION", "nogo": "NO-GO"}


def _supports_apl(body: dict) -> bool:
    """True if the requesting device can render APL (Echo Show, Fire TV…)."""
    try:
        ifaces = body["context"]["System"]["device"]["supportedInterfaces"]
        if "Alexa.Presentation.APL" in ifaces:
            return True
    except Exception:
        pass
    try:
        for v in body.get("context", {}).get("Viewports", []) or []:
            if v.get("type") == "APL":
                return True
    except Exception:
        pass
    return False


def _tile(label, value, sub=None):
    items = [
        {"type": "Text", "text": label, "fontSize": "20dp", "color": "#94a3b8",
         "textAlign": "center"},
        {"type": "Text", "text": value, "fontSize": "44dp", "fontWeight": "700",
         "color": "#f1f5f9", "textAlign": "center"},
    ]
    if sub:
        items.append({"type": "Text", "text": sub, "fontSize": "18dp",
                      "color": "#cbd5e1", "textAlign": "center"})
    return {
        "type": "Container", "grow": 1, "alignItems": "center",
        "justifyContent": "center", "paddingTop": "8dp", "paddingBottom": "8dp",
        "items": items,
    }


def build_apl_document(data: dict) -> dict:
    overall = data["overall"] or "caution"
    color = _STATUS_COLOR.get(overall, "#64748b")
    label = _STATUS_LABEL.get(overall, "—")

    # Banner subtitle from the forecast driver
    fc = data["forecast"]
    subtitle = "Vancouver / Howe Sound"
    if fc and fc["wind"] is not None:
        period = (fc["period"] or "").lower()
        dirn = _speak_dir(fc["dir"])
        when = "" if period in ("", "now") else f" {period}"
        subtitle = f"Peak {fc['wind']} kt{(' ' + dirn) if dirn else ''}{when}".strip()

    tiles = []
    eb = data["english_bay"]
    if eb:
        tiles.append(_tile("English Bay", f"{eb['wind']} kt",
                           f"{eb['wave_cm']} cm" if eb["wave_cm"] is not None else None))
    hs = data["pam_rocks"]
    if hs:
        tiles.append(_tile("Pam Rocks", f"{hs['wind']} kt", "Howe Sound"))
    if not tiles:
        tiles.append(_tile("Data", "—", "unavailable"))

    warning_items = []
    if fc and fc.get("strong_wind_warning"):
        warning_items.append({"type": "Text", "text": "⚠ STRONG WIND WARNING",
                              "fontSize": "24dp", "fontWeight": "700",
                              "color": "#fca5a5", "textAlign": "center"})
    elif fc and fc.get("wind_warning"):
        warning_items.append({"type": "Text", "text": "⚠ Wind Warning",
                              "fontSize": "22dp", "fontWeight": "700",
                              "color": "#fcd34d", "textAlign": "center"})

    return {
        "type": "APL",
        "version": "1.9",
        "mainTemplate": {
            "items": [{
                "type": "Container",
                "width": "100vw", "height": "100vh",
                "backgroundColor": "#0f172a",
                "paddingLeft": "32dp", "paddingRight": "32dp",
                "paddingTop": "24dp", "paddingBottom": "24dp",
                "items": [
                    {"type": "Text", "text": "Sailing Conditions",
                     "fontSize": "26dp", "color": "#94a3b8"},
                    {"type": "Frame", "backgroundColor": color, "borderRadius": "20dp",
                     "width": "100%", "marginTop": "12dp", "marginBottom": "16dp",
                     "paddingTop": "18dp", "paddingBottom": "18dp",
                     "item": {
                         "type": "Container", "alignItems": "center",
                         "items": [
                             {"type": "Text", "text": label, "fontSize": "72dp",
                              "fontWeight": "900", "color": "#ffffff", "textAlign": "center"},
                             {"type": "Text", "text": subtitle, "fontSize": "22dp",
                              "color": "#f8fafc", "textAlign": "center"},
                         ],
                     }},
                    {"type": "Container", "direction": "row", "width": "100%",
                     "items": tiles},
                ] + warning_items,
            }],
        },
    }


def handle_request(body: dict) -> dict:
    """Return an Alexa response envelope. LaunchRequest and any
    IntentRequest are treated identically — brief the conditions.
    Attaches an APL visual when the device has a screen."""
    data = _gather()
    speech = build_speech(data)
    response = {
        "outputSpeech": {"type": "PlainText", "text": speech},
        "card": {"type": "Simple", "title": "Sailing BC", "content": speech},
        "shouldEndSession": True,
    }
    if _supports_apl(body):
        response["directives"] = [{
            "type": "Alexa.Presentation.APL.RenderDocument",
            "token": "sailingConditions",
            "document": build_apl_document(data),
        }]
    return {"version": "1.0", "response": response}
