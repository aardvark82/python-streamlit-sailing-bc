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
from datetime import datetime, timedelta

import pandas as pd
import pytz
import requests
from bs4 import BeautifulSoup

from . import ai_provider, settings

log = logging.getLogger("helper.forecast")
VAN_TZ = pytz.timezone("America/Vancouver")

REGIONS = {
    "howe_sound":    {"name": "Howe Sound",
                      "url": "https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400"},
    "georgia_south": {"name": "Strait of Georgia – south",
                      "url": "https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14305"},
}

# Model for forecast parsing. gpt-5-nano was tried (v175) but returns
# unusable/empty CSV for this messy-HTML extraction task — reverted to
# gpt-5-mini. Change here to switch (openai_log records this name).
OPENAI_MODEL = "gpt-5-mini"

# Go/No-Go thresholds (knots) — match fetch_gonogo.py / alexa.py
WIND_GO = 10
WIND_CAUTION = 15

_USER_AGENT = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
               "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")

# region → (hour_slot, payload). weather.gc.ca re-issues marine forecasts
# hourly, so we cache per hour and roll over at HH:01 (1-min grace after
# the hour to let the new issue post). region: str → (slot:str, payload).
_cache: dict[str, tuple[str, dict]] = {}
_lock = threading.Lock()


def _hour_slot() -> str:
    """Identifier for the current hourly forecast window. Subtracting 1
    minute means the slot flips at HH:01, not HH:00 — so a forecast cached
    at 14:30 stays valid until 15:01, when the ~15:00 issue is expected."""
    return (datetime.now(VAN_TZ) - timedelta(minutes=1)).strftime("%Y-%m-%d %H")


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


def fetch_html(url: str) -> str:
    """Public alias of _fetch_html — used by the 'Test model' endpoint
    so it can bypass the per-region cache."""
    return _fetch_html(url)


def _extract_forecast_text(html: str) -> str:
    """Pull just the clean marine-forecast prose out of the full gc.ca
    page. Feeding the whole HTML (head, JS, nav, footer — tens of
    thousands of tokens) to a small local model makes it describe the
    webpage instead of parsing the forecast. We send only this.

    Priority: the <span class='textSummary'> we already parse for the
    summary; fall back to the #forecast-content div's text; last resort
    a stripped version of the whole page (truncated)."""
    try:
        soup = BeautifulSoup(html, "html.parser")
        summary = soup.find("span", class_="textSummary")
        if summary and summary.get_text(strip=True):
            return summary.get_text(" ", strip=True)
        fc = soup.find("div", id="forecast-content")
        if fc and fc.get_text(strip=True):
            return fc.get_text(" ", strip=True)
        text = soup.get_text(" ", strip=True)
        return text[:4000]
    except Exception:
        return html[:4000]


def _build_prompt(forecast_text: str) -> str:
    now = datetime.now(VAN_TZ)
    now_str = now.strftime("%A %H:%M %Z")
    is_evening = now.hour >= 19
    return (
        "You are parsing a marine wind forecast. Output ONLY a CSV table, "
        "no prose, no explanation, no code fences. "
        "Columns: time,wind speed,max wind speed,wind direction. "
        "wind speed is the first number in a range; max wind speed is the second. "
        "Example: '5 to 15 knots' -> wind speed 5, max wind speed 15. "
        "If it says 'light', use 3. "
        "If a max/gust isn't given, set max wind speed equal to wind speed, never less. "
        "wind direction is the compass word (e.g. northerly, southerly, northwesterly). "
        "The FIRST row is the current/nearest period with time 'now'. "
        "Add one row per subsequent period mentioned (this morning, this evening, "
        "Tuesday morning, etc.). "
        f"\n\nCurrent local time is {now_str}. "
        "For the 'now' row, if the text says 'winds X becoming Y this evening/tonight/overnight', "
        + ("it IS evening now, use the AFTER-transition value. "
           if is_evening else "it is NOT evening yet, use the BEFORE-transition value. ")
        + "\n\nForecast text:\n" + forecast_text
        + "\n\nCSV:"
    )


def _run_parse(html: str, *, reason: str, source_label: str):
    """Shared core: extract clean text → build prompt → call provider.
    Returns the ai_provider.AIResult."""
    from . import ai_provider
    forecast_text = _extract_forecast_text(html)
    prompt = _build_prompt(forecast_text)
    result = ai_provider.chat(
        messages=[
            {"role": "system", "content":
             "You are an expert meteorologist that outputs only CSV tables."},
            {"role": "user", "content": prompt},
        ],
        reason=reason,
        source_data=source_label,
    )
    # Attach the cleaned input so the test modal can show what was actually sent
    result_extra = forecast_text
    return result, result_extra


def ai_parse_html(html: str, reason: str, source_label: str) -> tuple[list[dict], dict]:
    """Run AI parsing and also return raw output + metadata for the UI."""
    result, clean_input = _run_parse(html, reason=reason, source_label=source_label)
    rows = _csv_to_rows(result.content)
    meta = {
        "provider": result.provider,
        "model": result.model,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "elapsed_sec": round(result.elapsed_sec, 2),
        "cost_usd": round(result.cost_usd, 6),
        "raw_output": result.content,
        "clean_input": clean_input,
    }
    return rows, meta


def _ai_parse(html: str, reason: str = "forecast parsing",
              source_label: str = "Marine forecast text") -> str:
    """Hand the cleaned forecast text to the configured provider and get
    back the CSV table."""
    result, _ = _run_parse(html, reason=reason, source_label=source_label)
    return result.content


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
        time = str(r.get("time", "")).strip()
        wind = num(r.get("wind speed", r.get("wind_speed", 0)))
        gust = num(r.get("max wind speed", r.get("max_wind_speed", 0)))
        dirn = str(r.get("wind direction", r.get("wind_direction", ""))).strip().upper()
        # Drop fully-empty rows (a weak model can emit blank/garbage lines)
        if not time and wind == 0 and gust == 0 and dirn in ("", "NAN"):
            continue
        rows.append({"time": time, "wind": wind, "gust": gust, "dir": dirn})
    return rows


_refreshing: set[str] = set()
_refresh_lock = threading.Lock()


def trigger_async_refresh(region: str = "howe_sound") -> bool:
    """Kick off a background forecast fetch (OpenAI) if one isn't already
    running for this region. Returns True if a new refresh was started.
    Used by the Alexa endpoint on a cache miss so the NEXT invocation
    has a warm forecast — without blocking the current 8s response."""
    if region not in REGIONS:
        return False
    with _refresh_lock:
        if region in _refreshing:
            return False
        _refreshing.add(region)

    def _run():
        try:
            get_forecast(region, allow_fetch=True)
        except Exception as e:
            log.warning("async forecast refresh failed for %s: %s", region, e)
        finally:
            with _refresh_lock:
                _refreshing.discard(region)

    threading.Thread(target=_run, daemon=True, name=f"fc-refresh-{region}").start()
    return True


def get_cached(region: str = "howe_sound") -> dict | None:
    """Return the forecast cached for the CURRENT hour slot, else None.
    Never does network I/O — used by latency-sensitive paths (Alexa, 8s)."""
    slot = _hour_slot()
    with _lock:
        hit = _cache.get(region)
    if hit and hit[0] == slot:
        return hit[1]
    return None


def get_forecast(region: str = "howe_sound", allow_fetch: bool = True) -> dict:
    if region not in REGIONS:
        return {"error": f"unknown region {region}"}
    meta = REGIONS[region]
    slot = _hour_slot()
    with _lock:
        hit = _cache.get(region)
        if hit and hit[0] == slot:
            return hit[1]
    if not allow_fetch:
        return {"error": "not cached", "region": region, "name": meta["name"], "stale": True}

    try:
        html = _fetch_html(meta["url"])
        summary = _parse_summary(html)
        rows = []
        ai_error = None
        try:
            rows = _csv_to_rows(_ai_parse(
                html,
                reason=f"forecast parsing ({meta['name']})",
                source_label=f"Marine forecast HTML — {meta['name']}",
            ))
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
        _cache[region] = (slot, payload)
    return payload


def _verdict(wind: float) -> str:
    if wind > WIND_CAUTION:
        return "nogo"
    if wind > WIND_GO:
        return "caution"
    return "go"


_WEEKDAYS = ("monday", "tuesday", "wednesday", "thursday",
             "friday", "saturday", "sunday")


def _is_today_label(time_label: str) -> bool:
    """True if a forecast period refers to today (now / today / this
    afternoon / this evening / tonight / overnight). Future days carry a
    weekday name (e.g. 'Saturday', 'Sunday night') or 'tomorrow'."""
    t = (time_label or "").lower()
    if "tomorrow" in t:
        return False
    if any(d in t for d in _WEEKDAYS):
        return False
    return True


def gonogo_from_forecast(region: str = "howe_sound", allow_fetch: bool = True,
                         window_periods: int = 2) -> dict:
    """Verdict from the WORST period inside the near-term window only.

    The window is the next `window_periods` forecast periods that belong
    to today (now + the immediate next period ≈ a 3-6h horizon). This
    ignores both future named days (Saturday gale on a calm Friday) AND
    later-today periods like 'overnight' that are well beyond a typical
    outing. The full multi-day table is still returned by /api/forecast
    for reference.

    allow_fetch=False → cached-only (for Alexa's 8s budget)."""
    fc = get_forecast(region, allow_fetch=allow_fetch)
    if fc.get("error") or not fc.get("rows"):
        return {"error": fc.get("error", "no forecast rows"), "region": region}

    today_rows = [r for r in fc["rows"] if _is_today_label(r["time"])]
    # Near-term window: first N today periods (now + next). Falls back to
    # whatever today rows exist, then to the 'now' row.
    window = (today_rows or fc["rows"])[:max(1, window_periods)]
    rows = window
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
        "window_periods_considered": [r["time"] for r in window],
    }
