"""Cloudflare KV client — same key namespace as the main Streamlit app.

Keys (compatible with st.py::record_buoy_data_history and
_fetch_buoy_wind_history_df):
    {buoy_id}_wind_{iso}       → float (knots)
    {buoy_id}_direction_{iso}  → compass label (e.g. 'W')
    {buoy_id}_wave_{iso}       → float metres (omitted if N/A)
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import Optional

import requests
import pytz

VAN_TZ = pytz.timezone("America/Vancouver")


def _config():
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    namespace_id = os.environ.get("CLOUDFLARE_NAMESPACE_ID")
    api_token = os.environ.get("CLOUDFLARE_API_TOKEN")
    if not all([account_id, namespace_id, api_token]):
        raise RuntimeError("Cloudflare env vars missing (CLOUDFLARE_ACCOUNT_ID / NAMESPACE_ID / API_TOKEN)")
    base = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    return base, {"Authorization": f"Bearer {api_token}"}


def slot_timestamp(dt: Optional[datetime] = None) -> str:
    """Bucket to 30-min slot in Vancouver tz — matches main app's slotting."""
    dt = dt or datetime.now(VAN_TZ)
    if dt.tzinfo is None:
        dt = VAN_TZ.localize(dt)
    else:
        dt = dt.astimezone(VAN_TZ)
    dt = dt.replace(minute=(dt.minute // 30) * 30, second=0, microsecond=0)
    return dt.isoformat(timespec="minutes")


def write_reading(buoy_id: str, wind_speed: float, direction: Optional[str],
                  wave_height_m: Optional[float], ts: Optional[str] = None) -> str:
    """Write the 3-key triplet for one observation. Returns the timestamp used."""
    base, headers = _config()
    ts = ts or slot_timestamp()
    keys = {
        f"{buoy_id}_wind_{ts}": str(wind_speed),
        f"{buoy_id}_direction_{ts}": str(direction or "N/A"),
    }
    if wave_height_m is not None:
        keys[f"{buoy_id}_wave_{ts}"] = str(wave_height_m)
    for k, v in keys.items():
        r = requests.put(f"{base}/values/{quote(k, safe='')}", headers=headers, data=v, timeout=15)
        r.raise_for_status()
    return ts


def _list_keys(prefix: str, limit: int = 1000):
    base, headers = _config()
    out = []
    cursor = None
    while True:
        params = {"prefix": prefix, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        r = requests.get(f"{base}/keys", params=params, headers=headers, timeout=20)
        r.raise_for_status()
        body = r.json()
        out.extend(item["name"] for item in body.get("result", []))
        cursor = body.get("result_info", {}).get("cursor")
        if not cursor:
            break
    return out


def _get(key: str) -> Optional[str]:
    base, headers = _config()
    r = requests.get(f"{base}/values/{quote(key, safe='')}", headers=headers, timeout=15)
    if r.status_code == 200:
        return r.text
    return None


def read_history(buoy_id: str, days_back: int = 14) -> list[dict]:
    """Pull triplets for `buoy_id` newer than `days_back` days, sorted oldest→newest.
    Returns list of {timestamp(datetime), wind_speed, direction, wave_height}."""
    cutoff = datetime.now(VAN_TZ) - timedelta(days=days_back)
    wind_keys = _list_keys(f"{buoy_id}_wind_")

    out = []
    for k in wind_keys:
        ts_str = k.replace(f"{buoy_id}_wind_", "")
        try:
            ts = datetime.fromisoformat(ts_str)
        except ValueError:
            continue
        if ts.tzinfo is None:
            ts = VAN_TZ.localize(ts)
        if ts < cutoff:
            continue
        wind = _get(k)
        direction = _get(f"{buoy_id}_direction_{ts_str}")
        wave = _get(f"{buoy_id}_wave_{ts_str}")
        try:
            wind_f = float(wind) if wind is not None else None
        except ValueError:
            wind_f = None
        try:
            wave_f = float(wave) if wave is not None else None
        except (ValueError, TypeError):
            wave_f = None
        out.append({
            "timestamp": ts,
            "wind_speed": wind_f,
            "direction": direction,
            "wave_height": wave_f,
        })

    out.sort(key=lambda r: r["timestamp"])
    return out
