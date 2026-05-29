"""Reconcile SQLite ↔ CF KV.

`status(buoys)` reports per-buoy:
  - rows in SQLite, keys in KV
  - missing_in_kv  (in SQLite, not in KV — likely a failed KV write)
  - missing_in_db  (in KV, not in SQLite — written by main Streamlit app)

`sync_kv_to_db(buoy)` backfills SQLite from KV (free, KV reads).
`sync_db_to_kv(buoy)` pushes any kv_synced=0 rows to KV.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from . import db
from . import kv_client as kv

log = logging.getLogger("helper.reconcile")

_SYNC_WORKERS = 50


def status(buoys: list[dict]) -> dict:
    out = {}
    for meta in buoys:
        bid = meta["id"]
        try:
            db_ts = db.list_timestamps(bid)
            kv_ts = kv.kv_list_timestamps(bid)
            missing_in_kv = sorted(db_ts - kv_ts, reverse=True)
            missing_in_db = sorted(kv_ts - db_ts, reverse=True)
            matched = db_ts & kv_ts
            out[bid] = {
                "name": meta["name"],
                "db_count": len(db_ts),
                "kv_count": len(kv_ts),
                "matched": len(matched),
                "missing_in_kv": missing_in_kv[:50],
                "missing_in_db": missing_in_db[:50],
                "missing_in_kv_total": len(missing_in_kv),
                "missing_in_db_total": len(missing_in_db),
                "ok": True,
            }
        except Exception as e:
            log.exception("reconcile %s failed", bid)
            out[bid] = {"name": meta["name"], "ok": False, "error": str(e)}
    return out


def sync_kv_to_db(buoy_id: str, max_records: int = 5000) -> dict:
    """Pull anything in KV but not in SQLite into SQLite. Each timestamp's
    triplet is fetched over the network, so we fan out 50-wide — the
    dominant cost is the KV round-trips, not the local SQLite upsert."""
    db_ts = db.list_timestamps(buoy_id)
    kv_ts = kv.kv_list_timestamps(buoy_id)
    missing = sorted(kv_ts - db_ts)[-max_records:]

    def _pull(ts):
        triple = kv.kv_fetch_one(buoy_id, ts)
        db.upsert(buoy_id, ts,
                  wind_speed=triple["wind_speed"],
                  direction=triple["direction"],
                  wave_height_m=triple["wave_height_m"],
                  kv_synced=True)
        return 1

    n = 0
    if missing:
        with ThreadPoolExecutor(max_workers=_SYNC_WORKERS) as ex:
            n = sum(ex.map(_pull, missing))
    log.info("sync_kv_to_db %s → wrote %d rows", buoy_id, n)
    return {"buoy_id": buoy_id, "synced": n}


def sync_db_to_kv(buoy_id: str) -> dict:
    """Push SQLite rows missing in KV (or marked kv_synced=0) to KV.
    Parallelized 50-wide — each push is 3 KV PUTs."""
    db_ts = db.list_timestamps(buoy_id)
    kv_ts = kv.kv_list_timestamps(buoy_id)
    missing = sorted(db_ts - kv_ts)

    def _push(ts):
        row = db.get_row(buoy_id, ts)
        if not row:
            return 0
        try:
            kv.write_reading(buoy_id,
                             wind_speed=row["wind_speed"] or 0,
                             direction=row["direction"],
                             wave_height_m=row["wave_height_m"],
                             ts=ts)
            return 1
        except Exception as e:
            log.warning("sync_db_to_kv %s @ %s failed: %s", buoy_id, ts, e)
            return 0

    n = 0
    if missing:
        with ThreadPoolExecutor(max_workers=_SYNC_WORKERS) as ex:
            n = sum(ex.map(_push, missing))
    log.info("sync_db_to_kv %s → pushed %d rows", buoy_id, n)
    return {"buoy_id": buoy_id, "pushed": n}
