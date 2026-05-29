"""Cloudflare KV client — same key namespace as the main Streamlit app.

Keys (compatible with st.py::record_buoy_data_history and
_fetch_buoy_wind_history_df):
    {buoy_id}_wind_{iso}       → float (knots)
    {buoy_id}_direction_{iso}  → compass label (e.g. 'W')
    {buoy_id}_wave_{iso}       → float metres (omitted if N/A)
"""
from __future__ import annotations

import logging
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from urllib.parse import quote
from typing import Optional

import requests
import pytz

from . import db, usage
from .envutil import getenv_ci

log = logging.getLogger("helper.kv")

VAN_TZ = pytz.timezone("America/Vancouver")

# Shared pool — capped at 50 concurrent connections so a single
# read_history() doesn't fan out unbounded against CF.
_MAX_PARALLEL = 50
_pool = ThreadPoolExecutor(max_workers=_MAX_PARALLEL, thread_name_prefix="cfkv")

# Reuse a single requests.Session so the TCP/TLS handshake to CF is
# pooled across the 50 worker threads (huge latency win on 300+ reads).
_session = requests.Session()
# urllib3 defaults to pool_connections=10/pool_maxsize=10 — would bottleneck
# the 50 worker threads down to 10. Mount adapters sized to match.
_adapter = requests.adapters.HTTPAdapter(pool_connections=_MAX_PARALLEL,
                                           pool_maxsize=_MAX_PARALLEL,
                                           max_retries=1)
_session.mount("https://", _adapter)
_session.mount("http://", _adapter)


def _config():
    account_id = getenv_ci("CLOUDFLARE_ACCOUNT_ID")
    namespace_id = getenv_ci("CLOUDFLARE_NAMESPACE_ID")
    api_token = getenv_ci("CLOUDFLARE_API_TOKEN")
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
    """Dual-write one observation:
    - SQLite always (local source of truth, drives dashboard reads)
    - CF KV best-effort (so the main Streamlit app sees the value too)

    If KV fails, the SQLite row is marked kv_synced=0 and Reconcile can
    push it later. Returns the timestamp slot used.
    """
    ts = ts or slot_timestamp()

    # 1) Local write — must succeed; provisionally mark as not-yet-synced
    db.upsert(buoy_id, ts, wind_speed, direction, wave_height_m, kv_synced=False)

    # 2) KV write — best-effort
    base, headers = _config()
    keys = {
        f"{buoy_id}_wind_{ts}": str(wind_speed),
        f"{buoy_id}_direction_{ts}": str(direction or "N/A"),
    }
    if wave_height_m is not None:
        keys[f"{buoy_id}_wave_{ts}"] = str(wave_height_m)

    def _put(kv):
        k, v = kv
        r = _session.put(f"{base}/values/{quote(k, safe='')}", headers=headers, data=v, timeout=15)
        r.raise_for_status()
        usage.record("write")

    try:
        list(_pool.map(_put, keys.items()))
        db.mark_kv_synced(buoy_id, ts)
    except Exception as e:
        log.warning("KV write for %s @ %s failed (kept in SQLite for later sync): %s",
                    buoy_id, ts, e)
    return ts


def _list_keys(prefix: str, limit: int = 1000):
    base, headers = _config()
    t0 = _time.time()
    out = []
    cursor = None
    pages = 0
    while True:
        pages += 1
        params = {"prefix": prefix, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        r = _session.get(f"{base}/keys", params=params, headers=headers, timeout=20)
        r.raise_for_status()
        usage.record("list")
        body = r.json()
        out.extend(item["name"] for item in body.get("result", []))
        cursor = body.get("result_info", {}).get("cursor")
        if not cursor:
            break
    log.info("_list_keys prefix=%r → %d keys in %d pages, %.2fs", prefix, len(out), pages, _time.time() - t0)
    return out


def _get(key: str) -> Optional[str]:
    base, headers = _config()
    r = _session.get(f"{base}/values/{quote(key, safe='')}", headers=headers, timeout=15)
    usage.record("read")
    if r.status_code == 200:
        return r.text
    return None


def _bulk_get(keys: list[str]) -> dict[str, Optional[str]]:
    """Fan out reads through the shared pool (cap 50). Returns {key: value or None}."""
    if not keys:
        return {}
    results = list(_pool.map(_get, keys))
    return dict(zip(keys, results))


# ── 60-second in-process cache for read_history ──────────────────────
# Log → Graph → Log click bursts triple-fetched the same data.
import time as _time
_HIST_TTL_SEC = 60
_hist_cache: dict[tuple, tuple[float, list]] = {}
_hist_lock = __import__("threading").Lock()


def invalidate_history(buoy_id: Optional[str] = None):
    """Drop cached histories (called by the fetch cycle after writes)."""
    with _hist_lock:
        if buoy_id is None:
            _hist_cache.clear()
        else:
            for k in list(_hist_cache.keys()):
                if k[0] == buoy_id:
                    del _hist_cache[k]


def _list_recent_wind_keys(buoy_id: str, want_n: int, max_days_back: int = 7) -> list[str]:
    """List wind-keys for `buoy_id` walking backwards day-by-day until we
    have at least `want_n` keys (or hit `max_days_back`). Uses a narrow
    date-prefix on each list call so we never paginate through the buoy's
    full history just to find the tail. ISO keys are sort-safe.
    """
    out: list[str] = []
    now = datetime.now(VAN_TZ)
    for offset in range(max_days_back):
        day = (now - timedelta(days=offset)).strftime("%Y-%m-%d")
        prefix = f"{buoy_id}_wind_{day}"
        out.extend(_list_keys(prefix))
        if len(out) >= want_n:
            break
    return sorted(out)


def read_history(buoy_id: str, days_back: int = 14,
                 fields: tuple[str, ...] = ("wind", "direction", "wave"),
                 last_n: Optional[int] = None) -> list[dict]:
    """Dashboard read — now served from SQLite (no CF KV traffic).

    `fields` is kept for API compatibility but ignored (SQLite read is one
    SQL query regardless). `last_n` slices the tail.
    """
    rows = db.read_history(buoy_id, days_back=days_back, last_n=last_n)
    log.debug("read_history(db) %s days=%s last_n=%s → %d rows",
              buoy_id, days_back, last_n, len(rows))
    return rows


# ── KV-only helpers (Reconcile) ────────────────────────────────────────

def kv_list_timestamps(buoy_id: str) -> set[str]:
    """All ts strings present in KV for `buoy_id` (parsed from wind_ keys)."""
    keys = _list_keys(f"{buoy_id}_wind_")
    out = set()
    for k in keys:
        ts_str = k.replace(f"{buoy_id}_wind_", "")
        out.add(ts_str)
    return out


def kv_fetch_one(buoy_id: str, ts: str) -> dict:
    """One triplet from KV (used by reconcile sync-from-kv backfill).
    Does its 3 GETs sequentially — reconcile already parallelizes the
    outer per-timestamp loop 50-wide, so pooling here too would just
    nest executors."""
    wind = _get(f"{buoy_id}_wind_{ts}")
    direction = _get(f"{buoy_id}_direction_{ts}")
    wave = _get(f"{buoy_id}_wave_{ts}")
    try:
        wind_f = float(wind) if wind is not None else None
    except ValueError:
        wind_f = None
    try:
        wave_f = float(wave) if wave is not None else None
    except (ValueError, TypeError):
        wave_f = None
    return {"wind_speed": wind_f, "direction": direction, "wave_height_m": wave_f}


# Legacy KV-based read_history (preserved for diagnostic/manual use).
def _kv_read_history(buoy_id: str, days_back: int = 14,
                 fields: tuple[str, ...] = ("wind", "direction", "wave"),
                 last_n: Optional[int] = None) -> list[dict]:
    cache_key = (buoy_id, days_back, fields, last_n)
    now_ts = _time.time()
    with _hist_lock:
        hit = _hist_cache.get(cache_key)
        if hit and (now_ts - hit[0]) < _HIST_TTL_SEC:
            log.info("read_history %s cache HIT (%d rows)", buoy_id, len(hit[1]))
            return hit[1]
    t0 = _time.time()

    cutoff = datetime.now(VAN_TZ) - timedelta(days=days_back)
    # Fast path: when caller only wants the last N readings, walk back
    # day-by-day with a date-narrowed prefix instead of listing the
    # buoy's full history (which could be thousands of keys → multi-second
    # pagination). Otherwise fall back to full list.
    if last_n is not None:
        wind_keys = _list_recent_wind_keys(buoy_id, want_n=last_n,
                                             max_days_back=max(1, days_back))
    else:
        wind_keys = sorted(_list_keys(f"{buoy_id}_wind_"))

    # First pass: filter to in-window timestamps
    triples = []   # (ts, ts_str)
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
        triples.append((ts, ts_str))

    if last_n is not None:
        triples = triples[-last_n:]

    # Build only the keys we actually need based on `fields`
    fetch_keys: list[str] = []
    for _ts, ts_str in triples:
        if "wind" in fields:
            fetch_keys.append(f"{buoy_id}_wind_{ts_str}")
        if "direction" in fields:
            fetch_keys.append(f"{buoy_id}_direction_{ts_str}")
        if "wave" in fields:
            fetch_keys.append(f"{buoy_id}_wave_{ts_str}")

    values = _bulk_get(fetch_keys)

    out = []
    for ts, ts_str in triples:
        wind = values.get(f"{buoy_id}_wind_{ts_str}") if "wind" in fields else None
        direction = values.get(f"{buoy_id}_direction_{ts_str}") if "direction" in fields else None
        wave = values.get(f"{buoy_id}_wave_{ts_str}") if "wave" in fields else None
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
    with _hist_lock:
        _hist_cache[cache_key] = (now_ts, out)
    log.info("read_history %s days=%s last_n=%s fields=%s → %d rows in %.2fs (%d kv reads)",
             buoy_id, days_back, last_n, fields, len(out), _time.time() - t0, len(fetch_keys))
    return out
