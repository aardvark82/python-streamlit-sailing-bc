"""KV + SQLite cleanup of stale readings.

Deletes both sides for symmetry — otherwise the next Reconcile would
flag everything-just-deleted as 'missing in KV' and a Sync would push
it all back.

Uses CF's bulk-delete endpoint (DELETE /bulk, JSON body = array of
keys, up to 10000 per call) so a 3-month sweep costs one HTTP roundtrip
per buoy per chunk.
"""
from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timedelta
from typing import Optional

import requests

from . import db, usage
from .kv_client import _config, _list_keys, _session

log = logging.getLogger("helper.cleanup")

_BULK_CHUNK = 10000  # CF bulk endpoint limit


def _kv_bulk_delete(keys: list[str]) -> int:
    if not keys:
        return 0
    base, headers = _config()
    headers = {**headers, "Content-Type": "application/json"}
    deleted = 0
    for i in range(0, len(keys), _BULK_CHUNK):
        chunk = keys[i:i + _BULK_CHUNK]
        r = _session.delete(f"{base}/bulk", headers=headers, json=chunk, timeout=60)
        r.raise_for_status()
        usage.record("delete", n=len(chunk))
        deleted += len(chunk)
    return deleted


def cleanup_older_than(buoys: list[dict], months: int = 3) -> dict:
    """Delete every reading older than `months` from BOTH KV and SQLite.

    Returns per-buoy counts: {bid: {kv_deleted, db_deleted, cutoff}}.
    """
    cutoff_dt = datetime.now() - timedelta(days=months * 30)
    cutoff_iso = cutoff_dt.replace(microsecond=0).isoformat(timespec="minutes")[:16]
    # KV keys are like '46146_wind_2026-02-20T14:30-08:00'.
    # Compare on the ts substring lexicographically — ISO is sort-safe.
    out = {}
    for meta in buoys:
        bid = meta["id"]
        entry = {"name": meta["name"], "cutoff": cutoff_iso, "kv_deleted": 0, "db_deleted": 0}
        try:
            # 1) Gather every old key across all three field types
            old_keys: list[str] = []
            for field in ("wind", "direction", "wave"):
                prefix = f"{bid}_{field}_"
                for k in _list_keys(prefix):
                    ts_part = k[len(prefix):][:16]  # first 16 chars = 'YYYY-MM-DDTHH:MM'
                    if ts_part < cutoff_iso:
                        old_keys.append(k)
            entry["kv_deleted"] = _kv_bulk_delete(old_keys)

            # 2) Drop matching SQLite rows
            db.init()
            c = sqlite3.connect(str(db.DB_PATH), timeout=10, isolation_level=None)
            try:
                cur = c.execute("DELETE FROM readings WHERE buoy_id=? AND ts < ?",
                                (bid, cutoff_iso))
                entry["db_deleted"] = cur.rowcount
            finally:
                c.close()
            log.info("cleanup %s — kv:%d db:%d (cutoff %s)",
                     bid, entry["kv_deleted"], entry["db_deleted"], cutoff_iso)
        except Exception as e:
            entry["error"] = str(e)
            log.exception("cleanup %s failed", bid)
        out[bid] = entry
    return out
