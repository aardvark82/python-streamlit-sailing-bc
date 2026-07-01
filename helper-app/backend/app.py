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

from . import (ai_log, ai_provider, alexa, cleanup, db, forecast,
                  healthcheck, openai_log, reconcile, settings, tides, usage, wave_model)
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


# Allow third-party clients (iOS chart plotter, LLM agents, etc.) to call
# the /api/v1/ public surface from any origin. Other endpoints are
# internal to the UI and don't need CORS.
@app.after_request
def _response_headers(resp):
    if request.path.startswith("/api/v1/"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "*"
    # The SPA shell (index.html) must always be revalidated so a fresh
    # Coolify deploy is picked up on a normal reload — no hard-refresh
    # needed. The single-file React app has no hashed asset bundle, so
    # the HTML is the only thing browsers were caching stalely.
    if resp.mimetype == "text/html":
        resp.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
        resp.headers["Expires"] = "0"
    return resp

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
    # NOTE: forecast is NOT warmed here — that would fire an OpenAI call
    # every hour and burn tokens. Forecast is fetched on demand instead
    # (Marine Forecast tab, or async-refreshed by the Alexa endpoint on
    # a cache miss).
    log.info("Fetch cycle complete")


# ── Routes ─────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


# ── /api/v1 — public read-only surface (CORS-enabled, stable contract) ─

import pytz as _pytz_v1


def _current_for(buoy_id: str):
    """Latest reading from SQLite, normalized for the v1 public API."""
    rows = db.read_history(buoy_id, days_back=1) or db.read_history(buoy_id, days_back=3)
    if not rows:
        return None
    r = rows[-1]
    ts = r["timestamp"]   # tz-aware Vancouver datetime
    ts_utc = ts.astimezone(_pytz_v1.UTC)
    age_min = int((datetime.now(VAN_TZ) - ts).total_seconds() / 60)
    meta = BUOY_BY_ID.get(buoy_id) or {}
    wave_m = r.get("wave_height")
    return {
        "id": buoy_id,
        "name": meta.get("name", buoy_id),
        "wind_kts": int(round(r["wind_speed"])) if r.get("wind_speed") is not None else None,
        "wind_direction": r.get("direction"),
        "wave_height_m": round(float(wave_m), 2) if wave_m is not None else None,
        "wave_height_cm": int(round(wave_m * 100)) if wave_m is not None else None,
        "waves_supported": bool(meta.get("waves", False)),
        "updated_at_utc": ts_utc.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "updated_at_local": ts.isoformat(timespec="seconds"),
        "age_minutes": age_min,
        "stale": age_min > 90,
    }


@app.route("/api/v1/locations")
def api_v1_locations():
    """Registry — id, human name, waves capability."""
    return jsonify([
        {"id": b["id"], "name": b["name"], "waves": bool(b.get("waves", False))}
        for b in BUOYS
    ])


@app.route("/api/v1/current")
def api_v1_current_all():
    """Current readings for every tracked location."""
    out = {}
    for b in BUOYS:
        c = _current_for(b["id"])
        if c:
            out[b["id"]] = c
    return jsonify({
        "fetched_at_utc": datetime.now(_pytz_v1.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "locations": out,
    })


@app.route("/api/v1/current/<bid>")
def api_v1_current_one(bid):
    """Current reading for a single location."""
    if bid not in BUOY_BY_ID:
        return jsonify(error=f"unknown location '{bid}'"), 404
    c = _current_for(bid)
    if not c:
        return jsonify(error="no data available", id=bid), 503
    return jsonify(c)


@app.route("/api/annual_tides")
def api_annual_tides():
    return jsonify(tides.annual_tides())


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
        "ai_provider": ai_provider.get_provider(),
        "openai_model": ai_provider.get_openai_model(),
        "ollama_model": ai_provider.get_ollama_model(),
        "ollama_url": ai_provider.get_ollama_url(),
        "ollama_api_key_set": bool(ai_provider.get_ollama_api_key()),
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
            current = settings.load()
            current.pop("openai_api_key", None)
            settings.SETTINGS_PATH.write_text(__import__("json").dumps(current, indent=2))
            return jsonify(ok=True, cleared=True)
    if "ai_provider" in body:
        v = (body["ai_provider"] or "").strip().lower()
        if v in ("openai", "ollama"):
            updates["ai_provider"] = v
    if "ollama_model" in body:
        v = (body["ollama_model"] or "").strip()
        if v:
            updates["ollama_model"] = v
    if "openai_model" in body:
        v = (body["openai_model"] or "").strip()
        if v:
            updates["openai_model"] = v
    if "ollama_url" in body:
        v = (body["ollama_url"] or "").strip()
        if v:
            updates["ollama_url"] = v
        else:
            # Empty value clears the override → falls back to env/default
            current = settings.load()
            current.pop("ollama_url", None)
            settings.SETTINGS_PATH.write_text(__import__("json").dumps(current, indent=2))
            return jsonify(ok=True, cleared="ollama_url")
    if "ollama_api_key" in body:
        v = (body["ollama_api_key"] or "").strip()
        if v:
            updates["ollama_api_key"] = v
        else:
            current = settings.load()
            current.pop("ollama_api_key", None)
            settings.SETTINGS_PATH.write_text(__import__("json").dumps(current, indent=2))
            return jsonify(ok=True, cleared="ollama_api_key")
    if updates:
        settings.save(updates)
    return jsonify(ok=True, updates=updates)


@app.route("/api/ai/test_forecast", methods=["POST"])
def api_ai_test_forecast():
    """Force-fresh forecast parse for diagnostics. Does NOT touch the
    region cache — always fetches the HTML and re-runs the AI parse with
    whatever provider/model is currently configured in Settings.
    Returns parsed rows + raw model output + tokens/cost/elapsed."""
    body = request.get_json(silent=True) or {}
    region = (body.get("region") or "howe_sound").strip()
    if region not in forecast.REGIONS:
        return jsonify(error=f"unknown region '{region}'"), 400
    meta = forecast.REGIONS[region]
    try:
        html = forecast.fetch_html(meta["url"])
        summary = forecast._parse_summary(html)
        rows, parse_meta = forecast.ai_parse_html(
            html,
            reason=f"forecast TEST ({meta['name']})",
            source_label=f"Marine forecast text — {meta['name']} (test)",
        )
        return jsonify({
            "ok": True,
            "region": region,
            "name": meta["name"],
            "url": meta["url"],
            "issued": summary.get("issued"),
            "period": summary.get("period"),
            "forecast_text": summary.get("forecast_text"),
            "rows": rows,
            **parse_meta,
        })
    except Exception as e:
        log.exception("forecast test failed")
        return jsonify(ok=False, error=str(e)), 500


@app.route("/api/ollama/status")
def api_ollama_status():
    return jsonify(ai_provider.ollama_status())


@app.route("/api/ollama/pull", methods=["POST"])
def api_ollama_pull():
    body = request.get_json(silent=True) or {}
    model = (body.get("model") or ai_provider.get_ollama_model()).strip()
    return jsonify(ai_provider.ollama_pull(model))


@app.route("/api/healthcheck", methods=["POST"])
def api_healthcheck():
    return jsonify(healthcheck.run_all())


@app.route("/api/ai_log")
def api_ai_log():
    return jsonify(ai_log.snapshot())


@app.route("/api/ai_log/reset", methods=["POST"])
def api_ai_log_reset():
    return jsonify(ok=True, cleared=ai_log.reset())


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


# Back-compat: old /api/openai_log routes now serve the unified ai_log.
@app.route("/api/openai_log")
def api_openai_log_compat():
    return jsonify(ai_log.snapshot())


@app.route("/api/openai_log/reset", methods=["POST"])
def api_openai_log_reset_compat():
    return jsonify(ok=True, cleared=ai_log.reset())


@app.route("/api/forecast")
def api_forecast():
    region = request.args.get("region", "howe_sound")
    return jsonify(forecast.get_forecast(region))


@app.route("/api/forecast/gonogo")
def api_forecast_gonogo():
    region = request.args.get("region", "howe_sound")
    return jsonify(forecast.gonogo_from_forecast(region))


@app.route("/api/forecast/regions")
def api_forecast_regions():
    return jsonify([{"id": k, "name": v["name"]} for k, v in forecast.REGIONS.items()])


@app.route("/api/alexa", methods=["GET", "POST"])
def api_alexa():
    """Alexa custom-skill HTTPS endpoint.
    GET  → {speech} for previewing in the UI / browser.
    POST → full Alexa response envelope (what the skill returns)."""
    if request.method == "GET":
        return jsonify(speech=alexa.build_speech())
    body = request.get_json(silent=True) or {}
    return jsonify(alexa.handle_request(body))


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
