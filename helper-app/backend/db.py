"""SQLite local store for buoy readings.

Acts as the primary read path for the dashboard (Log/Graph/Trends) so
those views never touch CF KV. Writes still dual-target KV for the
main Streamlit app's benefit.

Schema is intentionally aligned with the KV key triplet:
    {buoy_id}_wind_{ts}      → readings.wind_speed
    {buoy_id}_direction_{ts} → readings.direction
    {buoy_id}_wave_{ts}      → readings.wave_height_m

`ts` matches the KV key suffix exactly (Vancouver ISO, 30-min slot).
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pytz

from .envutil import getenv_ci

log = logging.getLogger("helper.db")

VAN_TZ = pytz.timezone("America/Vancouver")
DB_PATH = Path(getenv_ci("HELPER_DATA_DIR", "/data")) / "readings.sqlite"

_init_lock = threading.Lock()
_initialized = False


def _conn() -> sqlite3.Connection:
    """Fresh connection per call. WAL is set on the file so concurrent
    readers don't block each other."""
    c = sqlite3.connect(str(DB_PATH), timeout=10, isolation_level=None)
    c.row_factory = sqlite3.Row
    return c


def init():
    global _initialized
    with _init_lock:
        if _initialized:
            return
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = _conn()
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        c.execute("""
            CREATE TABLE IF NOT EXISTS readings (
                buoy_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                wind_speed REAL,
                direction TEXT,
                wave_height_m REAL,
                written_at TEXT NOT NULL,
                kv_synced INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY (buoy_id, ts)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS idx_buoy_ts ON readings(buoy_id, ts)")
        c.close()
        log.info("SQLite ready at %s", DB_PATH)
        _initialized = True


def upsert(buoy_id: str, ts: str, wind_speed: Optional[float],
           direction: Optional[str], wave_height_m: Optional[float],
           kv_synced: bool = True):
    init()
    c = _conn()
    try:
        c.execute("""
            INSERT INTO readings(buoy_id, ts, wind_speed, direction, wave_height_m,
                                  written_at, kv_synced)
            VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(buoy_id, ts) DO UPDATE SET
              wind_speed = excluded.wind_speed,
              direction = excluded.direction,
              wave_height_m = excluded.wave_height_m,
              written_at = excluded.written_at,
              kv_synced = max(readings.kv_synced, excluded.kv_synced)
        """, (buoy_id, ts, wind_speed, direction, wave_height_m,
              datetime.now(VAN_TZ).isoformat(timespec="seconds"),
              1 if kv_synced else 0))
    finally:
        c.close()


def mark_kv_synced(buoy_id: str, ts: str):
    init()
    c = _conn()
    try:
        c.execute("UPDATE readings SET kv_synced=1 WHERE buoy_id=? AND ts=?", (buoy_id, ts))
    finally:
        c.close()


def read_history(buoy_id: str, days_back: int = 14,
                 last_n: Optional[int] = None) -> list[dict]:
    """SQLite read — replaces the KV-backed version for dashboard views."""
    init()
    cutoff = (datetime.now(VAN_TZ) - timedelta(days=days_back)).isoformat(timespec="minutes")
    c = _conn()
    try:
        q = """SELECT ts, wind_speed, direction, wave_height_m
               FROM readings WHERE buoy_id=? AND ts >= ? ORDER BY ts"""
        rows = c.execute(q, (buoy_id, cutoff)).fetchall()
    finally:
        c.close()
    if last_n is not None:
        rows = rows[-last_n:]
    out = []
    for r in rows:
        try:
            ts = datetime.fromisoformat(r["ts"])
            if ts.tzinfo is None:
                ts = VAN_TZ.localize(ts)
        except ValueError:
            continue
        out.append({
            "timestamp": ts,
            "wind_speed": r["wind_speed"],
            "direction": r["direction"],
            "wave_height": r["wave_height_m"],
        })
    return out


def list_timestamps(buoy_id: str) -> set[str]:
    """All ts strings stored locally for a buoy. Used by Reconcile."""
    init()
    c = _conn()
    try:
        return {r["ts"] for r in c.execute("SELECT ts FROM readings WHERE buoy_id=?", (buoy_id,))}
    finally:
        c.close()


def get_row(buoy_id: str, ts: str) -> Optional[dict]:
    init()
    c = _conn()
    try:
        r = c.execute("""SELECT wind_speed, direction, wave_height_m, kv_synced
                          FROM readings WHERE buoy_id=? AND ts=?""", (buoy_id, ts)).fetchone()
    finally:
        c.close()
    return dict(r) if r else None


def stats() -> dict:
    init()
    c = _conn()
    try:
        out = {}
        for r in c.execute("""SELECT buoy_id, COUNT(*) AS n,
                                       SUM(kv_synced) AS synced,
                                       MIN(ts) AS first_ts, MAX(ts) AS last_ts
                                FROM readings GROUP BY buoy_id"""):
            out[r["buoy_id"]] = {
                "rows": r["n"],
                "kv_synced": r["synced"],
                "first_ts": r["first_ts"],
                "last_ts": r["last_ts"],
            }
        return out
    finally:
        c.close()
