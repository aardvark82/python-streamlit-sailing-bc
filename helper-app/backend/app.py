"""Flask + APScheduler — single-process headless helper.

Endpoints:
  GET  /                       → static index.html (React app via CDN)
  GET  /api/locations          → buoy registry
  GET  /api/log?location=&limit=12   → recent readings
  GET  /api/series?location=&days=3  → time-series for charting
  GET  /api/trends?location=&days=14 → morning/afternoon/evening summary
  GET  /api/settings           → settings (with key masked)
  POST /api/settings           → update settings
  POST /api/fetch_now          → trigger one fetch cycle immediately
  GET  /api/health             → liveness + last-cycle status
"""
from __future__ import annotations

import logging
import os
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from apscheduler.schedulers.background import BackgroundScheduler

from . import cleanup, db, reconcile, settings, usage, wave_model
from .buoy_fetcher import BUOYS, BUOY_BY_ID, fetch_buoy
from .envutil import getenv_ci
from .kv_client import read_history, write_reading, VAN_TZ, invalidate_history
from .trends import summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
log = logging.getLogger("helper")

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"
VERSION_PATH = Path(__file__).resolve().parent.parent / "VERSION"
try:
    APP_VERSION = VERSION_PATH.read_text().strip() or "dev"
except OSError:
    APP_VERSION = "dev"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

# In-memory cycle status (last run per buoy)
_cycle_status: dict[str, dict] = {}
_cycle_lock = threading.Lock()


def _fetch_one(meta):
    """Single-buoy fetch+write. Runs in a worker thread for the parallel cycle."""
    bid = meta["id"]
    entry = {"buoy_id": bid, "name": meta["name"], "started_at": datetime.now(VAN_TZ).isoformat()}
    try:
        r = fetch_buoy(bid)
        ts = write_reading(bid, r.wind_speed, r.direction, r.wave_height_m)
        entry.update({
            "ok": True, "ts": ts,
            "wind_speed": r.wind_speed, "direction": r.direction,
            "wave_m": r.wave_height_m,
        })
        log.info("[%s] %s — wrote %s kts %s waves=%s", bid, meta["name"],
                 r.wind_speed, r.direction, r.wave_height_m)
    except Exception as e:
        entry.update({"ok": False, "error": str(e)})
        log.exception("[%s] failed", bid)
    with _cycle_lock:
        _cycle_status[bid] = entry


def run_fetch_cycle():
    """Fetch all buoys in parallel, write to CF KV. Logs per-buoy success/error."""
    log.info("Starting fetch cycle")
    # 4 buoys, each ~1-2s for the gc.ca scrape — parallel cuts cycle time ~4×
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=len(BUOYS), thread_name_prefix="buoy") as ex:
        list(ex.map(_fetch_one, BUOYS))
    # Invalidate all caches — new data landed
    try:
        with _trends_lock:
            _trends_cache.clear()
    except NameError:
        pass
    invalidate_history()
    log.info("Fetch cycle complete")


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/api/version")
def api_version():
    return jsonify(version=APP_VERSION)


@app.route("/api/locations")
def api_locations():
    return jsonify(BUOYS)


@app.route("/api/log")
def api_log():
    bid = request.args.get("location")
    limit = int(request.args.get("limit", 12))
    if bid not in BUOY_BY_ID:
        return jsonify(error="unknown location"), 400
    # last_n + days_back=1 → list 1 day, fetch only the last `limit` triplets
    rows = read_history(bid, days_back=1, last_n=limit)
    return jsonify([{
        "timestamp": r["timestamp"].isoformat(),
        "wind_speed": r["wind_speed"],
        "direction": r["direction"],
        "wave_height_m": r["wave_height"],
    } for r in reversed(rows)])


@app.route("/api/series")
def api_series():
    bid = request.args.get("location")
    days = int(request.args.get("days", 3))
    if bid not in BUOY_BY_ID:
        return jsonify(error="unknown location"), 400
    # Graph view only plots wind + wave — skip direction reads
    rows = read_history(bid, days_back=days, fields=("wind", "wave"))
    return jsonify([{
        "timestamp": r["timestamp"].isoformat(),
        "wind_speed": r["wind_speed"],
        "direction": r["direction"],
        "wave_height_m": r["wave_height"],
    } for r in rows])


# Trends cache — heaviest endpoint (~8k KV reads per "all,14d" call).
# 10-min TTL is fine: trends look at a 14-day window so a click-to-click
# delta is invisible. Cuts CF KV reads ~60× on repeated page loads.
import time as _time
_TRENDS_TTL_SEC = 600
_trends_cache: dict[tuple, tuple[float, dict]] = {}
_trends_lock = threading.Lock()


def _cached_trends(bid: str, days: int):
    key = (bid, days)
    now = _time.time()
    with _trends_lock:
        hit = _trends_cache.get(key)
        if hit and (now - hit[0]) < _TRENDS_TTL_SEC:
            return hit[1], True  # cached
    # Compute outside the lock so concurrent buoys aren't serialized
    # Trends only uses wind + wave averages — skip direction reads
    if bid == "all":
        data = {b["id"]: summarize(read_history(b["id"], days_back=days, fields=("wind", "wave"))) for b in BUOYS}
    else:
        data = summarize(read_history(bid, days_back=days, fields=("wind", "wave")))
    with _trends_lock:
        _trends_cache[key] = (now, data)
    return data, False


@app.route("/api/trends")
def api_trends():
    bid = request.args.get("location")
    days = int(request.args.get("days", 14))
    if bid != "all" and bid not in BUOY_BY_ID:
        return jsonify(error="unknown location"), 400
    data, cached = _cached_trends(bid, days)
    resp = jsonify(data)
    resp.headers["X-Cache"] = "HIT" if cached else "MISS"
    return resp


@app.route("/api/trends/clear", methods=["POST"])
def api_trends_clear():
    with _trends_lock:
        n = len(_trends_cache)
        _trends_cache.clear()
    return jsonify(ok=True, cleared=n)


@app.route("/api/settings", methods=["GET"])
def api_settings_get():
    s = settings.load()
    key = s.get("openai_api_key") or ""
    return jsonify({
        "openai_api_key_set": bool(key),
        "openai_api_key_masked": (key[:6] + "…" + key[-4:]) if len(key) >= 12 else None,
    })


@app.route("/api/settings", methods=["POST"])
def api_settings_post():
    body = request.get_json(silent=True) or {}
    updates = {}
    if "openai_api_key" in body:
        v = (body["openai_api_key"] or "").strip()
        if v:
            updates["openai_api_key"] = v
        else:
            # Empty value clears the key
            current = settings.load()
            current.pop("openai_api_key", None)
            settings.SETTINGS_PATH.write_text(__import__("json").dumps(current, indent=2))
            return jsonify(ok=True, cleared=True)
    if updates:
        settings.save(updates)
    return jsonify(ok=True)


@app.route("/api/settings/test", methods=["POST"])
def api_settings_test():
    """Verify each configured credential without exposing its value.
    - Cloudflare: list one KV key (read-only smoke test).
    - OpenAI: GET /v1/models with the configured key.
    Each returns {ok, detail} so the UI can show per-provider status."""
    import requests as _rq

    results = {}

    # Cloudflare
    account_id = getenv_ci("CLOUDFLARE_ACCOUNT_ID")
    namespace_id = getenv_ci("CLOUDFLARE_NAMESPACE_ID")
    api_token = getenv_ci("CLOUDFLARE_API_TOKEN")
    if not all([account_id, namespace_id, api_token]):
        missing = [n for n, v in [("ACCOUNT_ID", account_id),
                                    ("NAMESPACE_ID", namespace_id),
                                    ("API_TOKEN", api_token)] if not v]
        results["cloudflare"] = {"ok": False, "detail": f"missing env: {', '.join(missing)}"}
    else:
        try:
            url = (f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
                   f"/storage/kv/namespaces/{namespace_id}/keys")
            # CF requires limit ≥ 10 on the /keys endpoint
            r = _rq.get(url, headers={"Authorization": f"Bearer {api_token}"},
                        params={"limit": 10}, timeout=10)
            if r.status_code == 200 and r.json().get("success"):
                count = len(r.json().get("result", []))
                results["cloudflare"] = {"ok": True, "detail": f"KV reachable ({count} sample keys)"}
            else:
                results["cloudflare"] = {"ok": False, "detail": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            results["cloudflare"] = {"ok": False, "detail": str(e)[:200]}

    # OpenAI — prefer env var, fall back to /data/settings.json
    openai_key = settings.get_openai_key()
    if not openai_key:
        results["openai"] = {"ok": False, "detail": "no key set (env OPENAI_API_KEY or Settings tab)"}
    else:
        try:
            r = _rq.get("https://api.openai.com/v1/models",
                        headers={"Authorization": f"Bearer {openai_key}"}, timeout=10)
            if r.status_code == 200:
                n = len(r.json().get("data", []))
                results["openai"] = {"ok": True, "detail": f"{n} models accessible"}
            elif r.status_code == 401:
                results["openai"] = {"ok": False, "detail": "401 unauthorized (bad key)"}
            else:
                results["openai"] = {"ok": False, "detail": f"HTTP {r.status_code}: {r.text[:200]}"}
        except Exception as e:
            results["openai"] = {"ok": False, "detail": str(e)[:200]}

    return jsonify(results)


@app.route("/api/fetch_now", methods=["POST"])
def api_fetch_now():
    threading.Thread(target=run_fetch_cycle, daemon=True).start()
    return jsonify(ok=True, message="fetch cycle started")


@app.route("/api/usage")
def api_usage():
    return jsonify(usage.snapshot())


@app.route("/api/reconcile")
def api_reconcile_status():
    return jsonify(reconcile.status(BUOYS))


@app.route("/api/reconcile/sync_kv_to_db", methods=["POST"])
def api_reconcile_kv_to_db():
    bid = (request.get_json(silent=True) or {}).get("location")
    if bid not in BUOY_BY_ID:
        return jsonify(error="unknown location"), 400
    return jsonify(reconcile.sync_kv_to_db(bid))


@app.route("/api/reconcile/sync_db_to_kv", methods=["POST"])
def api_reconcile_db_to_kv():
    bid = (request.get_json(silent=True) or {}).get("location")
    if bid not in BUOY_BY_ID:
        return jsonify(error="unknown location"), 400
    return jsonify(reconcile.sync_db_to_kv(bid))


@app.route("/api/reconcile/sync_all", methods=["POST"])
def api_reconcile_sync_all():
    """Full bidirectional sync across every buoy. Pull-from-KV first
    (cheap on the dashboard side), then push any local-only rows to KV."""
    pulled = {}
    pushed = {}
    for meta in BUOYS:
        bid = meta["id"]
        try:
            pulled[bid] = reconcile.sync_kv_to_db(bid)["synced"]
        except Exception as e:
            pulled[bid] = f"error: {e}"
        try:
            pushed[bid] = reconcile.sync_db_to_kv(bid)["pushed"]
        except Exception as e:
            pushed[bid] = f"error: {e}"
    return jsonify(pulled=pulled, pushed=pushed)


@app.route("/api/db/stats")
def api_db_stats():
    return jsonify(db.stats())


@app.route("/api/wave_model")
def api_wave_model():
    """Fit wave_m ≈ a·wind² + b at the best lag for each waves-bearing buoy."""
    out = {}
    for meta in BUOYS:
        if not meta.get("waves"):
            continue
        out[meta["id"]] = {"name": meta["name"], **wave_model.fit(meta["id"])}
    return jsonify(out)


@app.route("/api/wave_model/predict")
def api_wave_model_predict():
    bid = request.args.get("location")
    wind = float(request.args.get("wind_kts", 0))
    direction = request.args.get("direction")
    if bid not in BUOY_BY_ID:
        return jsonify(error="unknown location"), 400
    return jsonify(wave_model.predict(bid, wind, direction) or {"error": "model not available"})


@app.route("/api/cleanup/old", methods=["POST"])
def api_cleanup_old():
    body = request.get_json(silent=True) or {}
    months = int(body.get("months", 3))
    result = cleanup.cleanup_older_than(BUOYS, months=months)
    return jsonify(result)


@app.route("/api/health")
def api_health():
    with _cycle_lock:
        return jsonify(ok=True, last_cycle=dict(_cycle_status))


# ── Scheduler bootstrap ────────────────────────────────────────────────

_scheduler: BackgroundScheduler | None = None


def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return
    interval_min = int(getenv_ci("FETCH_INTERVAL_MIN", "60"))
    _scheduler = BackgroundScheduler(timezone=str(VAN_TZ))
    _scheduler.add_job(run_fetch_cycle, "interval", minutes=interval_min,
                       next_run_time=datetime.now(VAN_TZ), id="hourly_fetch",
                       max_instances=1, coalesce=True)
    _scheduler.start()
    log.info("Scheduler started — every %d min", interval_min)


# Start on import so both `flask run` and `gunicorn` get the scheduler
db.init()
start_scheduler()


if __name__ == "__main__":
    port = int(getenv_ci("PORT", "5111"))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
