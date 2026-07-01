"""Annual tide envelope + moon phases for Pt Atkinson.

Pulls a year of hi/lo tide PREDICTIONS (±6 months) from the DFO IWLS
REST API, reduces to a daily high/low envelope, and computes moon-phase
events (new / first-quarter / full / last-quarter) across the window.

New & full moons → spring tides (largest range: highest highs + lowest
lows). Quarter moons → neap tides (smallest range). Overlaying the two
makes the ~2-week spring/neap cycle and the seasonal extreme tides
obvious.
"""
from __future__ import annotations

import logging
import math
import threading
from datetime import datetime, timedelta, timezone

import pytz
import requests

log = logging.getLogger("helper.tides")

VAN_TZ = pytz.timezone("America/Vancouver")
IWLS_API_BASE = "https://api-iwls.dfo-mpo.gc.ca/api/v1"
PT_ATKINSON_STATION_ID = "5cebf1de3d0f4a073c4bb94c"

_cache: dict[str, tuple[str, dict]] = {}
_lock = threading.Lock()


# ── IWLS tide predictions ─────────────────────────────────────────────

def _fetch_hilo(start_dt: datetime, end_dt: datetime):
    """Fetch wlp-hilo extremes in ≤30-day chunks (IWLS caps long ranges).
    Returns list of (iso_utc, height)."""
    out = []
    cur = start_dt
    while cur < end_dt:
        chunk_end = min(cur + timedelta(days=30), end_dt)
        params = {
            "time-series-code": "wlp-hilo",
            "from": cur.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "to": chunk_end.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        try:
            r = requests.get(f"{IWLS_API_BASE}/stations/{PT_ATKINSON_STATION_ID}/data",
                             params=params, timeout=25, headers={"Accept": "application/json"})
            r.raise_for_status()
            for item in (r.json() or []):
                if not isinstance(item, dict):
                    continue
                ts = item.get("eventDate") or item.get("event_date")
                val = item.get("value")
                if ts is None or val is None:
                    continue
                out.append((ts, float(val)))
        except Exception as e:
            log.warning("IWLS chunk %s failed: %s", cur.date(), e)
        cur = chunk_end
    return out


def _daily_envelope(points):
    """Group hi/lo points by Vancouver date → daily high (max) + low (min)."""
    days = {}
    for ts, h in points:
        try:
            dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00")).astimezone(VAN_TZ)
        except Exception:
            continue
        d = dt.date().isoformat()
        rec = days.setdefault(d, {"high": h, "low": h})
        rec["high"] = max(rec["high"], h)
        rec["low"] = min(rec["low"], h)
    return [{"date": d, "high": round(v["high"], 2), "low": round(v["low"], 2)}
            for d, v in sorted(days.items())]


# ── moon phases (dependency-free) ─────────────────────────────────────

_SYNODIC = 29.530588853              # mean synodic month (days)
_REF_NEW_MOON_JD = 2451550.1         # known new moon: 2000-01-06 18:14 UTC
_PHASES = [(0.0, "new"), (0.25, "first_quarter"), (0.5, "full"), (0.75, "last_quarter")]


def _to_jd(dt_utc: datetime) -> float:
    return dt_utc.timestamp() / 86400.0 + 2440587.5


def _from_jd(jd: float) -> datetime:
    return datetime.fromtimestamp((jd - 2440587.5) * 86400.0, tz=timezone.utc)


def _moon_events(start_dt: datetime, end_dt: datetime):
    events = []
    k = math.floor((_to_jd(start_dt) - _REF_NEW_MOON_JD) / _SYNODIC) - 1
    while True:
        base = _REF_NEW_MOON_JD + k * _SYNODIC
        if _from_jd(base) > end_dt + timedelta(days=2):
            break
        for frac, name in _PHASES:
            dt = _from_jd(_REF_NEW_MOON_JD + (k + frac) * _SYNODIC)
            if start_dt <= dt <= end_dt:
                events.append({
                    "date": dt.astimezone(VAN_TZ).date().isoformat(),
                    "phase": name,
                    "spring": name in ("new", "full"),
                })
        k += 1
    events.sort(key=lambda e: e["date"])
    return events


# ── public ────────────────────────────────────────────────────────────

def annual_tides(months_back: int = 6, months_ahead: int = 6) -> dict:
    """Daily tide envelope + moon phases for [now-6mo, now+6mo].
    Cached per calendar day (predictions don't change intraday)."""
    now = datetime.now(timezone.utc)
    day_key = now.strftime("%Y-%m-%d")
    ck = f"annual:{months_back}:{months_ahead}"
    with _lock:
        hit = _cache.get(ck)
        if hit and hit[0] == day_key:
            return hit[1]

    start = now - timedelta(days=months_back * 30)
    end = now + timedelta(days=months_ahead * 30)
    points = _fetch_hilo(start, end)
    days = _daily_envelope(points)
    moons = _moon_events(start, end)
    payload = {
        "generated_at": now.astimezone(VAN_TZ).isoformat(timespec="seconds"),
        "station": "Point Atkinson",
        "now": datetime.now(VAN_TZ).date().isoformat(),
        "days": days,
        "moons": moons,
        "error": None if days else "no tide data returned by IWLS",
    }
    with _lock:
        _cache[ck] = (day_key, payload)
    return payload
