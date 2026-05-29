"""Marine forecast fetch + OpenAI parse for the helper-app.

Mirrors the main Streamlit app's fetch_forecast.openAIFetchForecastForURL:
scrape weather.gc.ca marine forecast HTML, hand the page to gpt-5-mini,
get back a CSV table (time, wind speed, max wind speed, wind direction).

Used by the Alexa briefing + the Marine Forecast tab. The forecast-based
Go/No-Go (gonogo_from_forecast) looks at conditions LATER in the day
rather than extrapolating past wind — answers 'will it blow up this
afternoon?' which the past-wind trend model can't.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import time as _time
from datetime import datetime

import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup

from . import settings

log = logging.getLogger("helper.forecast")
VAN_TZ = pytz.timezone("America/Vancouver")

REGIONS = {
    "howe_sound":    {"name": "Howe Sound",
                      "url": "https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400"},
    "georgia_south": {"name": "Strait of Georgia – south",
                      "url": "https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14305"},
}

# Go/No-Go thresholds (knots) — match fetch_gonogo.py / alexa.py
WIND_GO = 10
WIND_CAUTION = 15

_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# (region, time_bucket) → (fetched_at, payload). 30-min TTL.
_cache: dict[tuple, tuple[float, dict]] = {}
_lock = threading.Lock()
_TTL = 1800


def _time_bucket() -> str:
    h = datetime.now(VAN_TZ).hour
    if h >= 23:
        return "overnight"
    if h >= 21:
        return "tonight"
    if h >= 19:
        return "evening"
    return "day"


def _fetch_html(url: str) -> str:
    r = requests.get(url, headers={"User-Agent": _USER_AGENT, "Accept": "text/html"}, timeout=25)
    r.raise_for_status()
    return r.text


def _parse_summary(html: str) -> dict:
    soup = BeautifulSoup(html, "html.parser")
    out = {"issued": None, "period": "", "forecast_text": "",
           "wind_warning": False, "strong_wind_warning": False}

    banner = soup.find("div", id="warningBanner")
    if banner:
        t = banner.text.lower()
        out["strong_wind_warning"] = "strong wind warning" in t
        out["wind_warning"] = "wind warning" in t and not out["strong_wind_warning"]

    fc = soup.find("div", id="forecast-content")
    if not fc:
        return out
    issue = fc.find("span", class_="text-info")
    if issue:
        m = re.match(r"Issued\s+(\d{1,2}:\d{2}\s+(?:AM|PM)\s+\w+\s+\d{2}\s+\w+\s+\d{4})", issue.text.strip())
        if m:
            try:
                parts = m.group(1).replace("Issued", "").strip().split()
                if len(parts) == 6:
                    time_part, ampm, _tz, day, month, year = parts
                    dt = datetime.strptime(f"{time_part} {ampm} {day} {month} {year}", "%I:%M %p %d %B %Y")
                    out["issued"] = VAN_TZ.localize(dt).isoformat()
            except ValueError:
                pass
    period = fc.find("span", class_="periodOfCoverage")
    out["period"] = period.text.strip() if period else ""
    summary = fc.find("span", class_="textSummary")
    out["forecast_text"] = summary.text.strip() if summary else ""
    return out


def _openai_parse(html: str) -> str:
    key = settings.get_openai_key()
    if not key:
        raise RuntimeError("OpenAI key not set (Settings tab or OPENAI_API_KEY env)")

    now = datetime.now(VAN_TZ)
    now_str = now.strftime("%A %H:%M %Z")
    is_evening = now.hour >= 19

    prompt = (
        "Make it short and just the table. "
        "Parse this forecast from marine weather canada (the section called \"Marine Forecast\") and "
        "extract a table with the following columns: time, wind speed, max wind speed, wind direction. "
        "wind speed is the first number in the wind speed string. max wind speed is the second. "
        "for example - if it says 5 to 15 knots, wind speed is 5 and max wind speed is 15. "
        "If it says light winds, use a value of 3. "
        "Make sure the Max (Gust/gusting) wind speed if not mentioned is the value of the wind speed, never less. "
        "Make it a CSV. The first row is current conditions with time 'now'. "
        f"\n\nCurrent local time is {now_str}. "
        "For the FIRST row, if the forecast says 'winds X becoming Y this evening/tonight/overnight', "
        + ("it IS evening now, use the AFTER-transition value. "
           if is_evening else "it is NOT evening yet, use the BEFORE-transition value. ")
        + "\n\nHere's the forecast HTML:\n" + html
    )

    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json={"model": "gpt-5-mini",
              "messages": [
                  {"role": "system", "content": "You are an expert meteorologist."},
                  {"role": "user", "content": prompt},
              ]},
        timeout=60,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def _csv_to_rows(csv_text: str) -> list[dict]:
    csv_text = csv_text.replace("```csv", "").replace("```", "").strip()
    df = pd.read_csv(io.StringIO(csv_text), sep=",", on_bad_lines="skip")
    df = df.dropna(how="all").reset_index(drop=True)
    df.columns = df.columns.str.strip().str.lower()

    def num(x):
        if pd.isna(x):
            return 0
        s = str(x)
        if "light" in s.lower():
            return 3
        nums = re.findall(r"\d+", s)
        return max(int(n) for n in nums) if nums else 0

    rows = []
    for _, r in df.iterrows():
        rows.append({
            "time": str(r.get("time", "")).strip(),
            "wind": num(r.get("wind speed", r.get("wind_speed", 0))),
            "gust": num(r.get("max wind speed", r.get("max_wind_speed", 0))),
            "dir": str(r.get("wind direction", r.get("wind_direction", ""))).strip().upper(),
        })
    return rows


def get_cached(region: str = "howe_sound") -> dict | None:
    """Return a cached forecast if fresh, else None. Never does network I/O —
    used by latency-sensitive paths (Alexa, which has an ~8s timeout)."""
    with _lock:
        hit = _cache.get((region, _time_bucket()))
    if hit and _time.time() - hit[0] < _TTL:
        return hit[1]
    return None


def get_forecast(region: str = "howe_sound", allow_fetch: bool = True) -> dict:
    if region not in REGIONS:
        return {"error": f"unknown region {region}"}
    meta = REGIONS[region]
    cache_key = (region, _time_bucket())
    now_ts = _time.time()
    with _lock:
        hit = _cache.get(cache_key)
        if hit and now_ts - hit[0] < _TTL:
            return hit[1]
    if not allow_fetch:
        return {"error": "not cached", "region": region, "name": meta["name"], "stale": True}

    try:
        html = _fetch_html(meta["url"])
        summary = _parse_summary(html)
        rows = []
        ai_error = None
        try:
            rows = _csv_to_rows(_openai_parse(html))
        except Exception as e:
            ai_error = str(e)
            log.warning("OpenAI forecast parse failed for %s: %s", region, e)
        payload = {
            "region": region,
            "name": meta["name"],
            "url": meta["url"],
            **summary,
            "rows": rows,
            "ai_error": ai_error,
        }
    except Exception as e:
        return {"error": str(e), "region": region, "name": meta["name"]}

    with _lock:
        _cache[cache_key] = (now_ts, payload)
    return payload


def _verdict(wind: float) -> str:
    if wind > WIND_CAUTION:
        return "nogo"
    if wind > WIND_GO:
        return "caution"
    return "go"


def gonogo_from_forecast(region: str = "howe_sound", allow_fetch: bool = True) -> dict:
    """Verdict from the WORST forecast period for the rest of today.
    Looks past the 'now' row at upcoming periods so it flags an
    afternoon/evening blow-up before the wind actually arrives.
    allow_fetch=False → cached-only (for Alexa's 8s budget)."""
    fc = get_forecast(region, allow_fetch=allow_fetch)
    if fc.get("error") or not fc.get("rows"):
        return {"error": fc.get("error", "no forecast rows"), "region": region}

    rows = fc["rows"]
    # Consider all rows (incl. 'now') but report which drives the verdict.
    worst = max(rows, key=lambda r: max(r["wind"], r["gust"]))
    driving = max(worst["wind"], worst["gust"])
    status = _verdict(driving)
    # Strong-wind warning overrides to nogo
    if fc.get("strong_wind_warning"):
        status = "nogo"
    elif fc.get("wind_warning") and status == "go":
        status = "caution"
    return {
        "region": region,
        "name": fc["name"],
        "status": status,
        "driving_wind_kts": driving,
        "driving_period": worst["time"],
        "driving_dir": worst["dir"],
        "wind_warning": fc.get("wind_warning"),
        "strong_wind_warning": fc.get("strong_wind_warning"),
        "issued": fc.get("issued"),
        "period": fc.get("period"),
    }
