"""One-shot setup diagnostics for the Settings → Health Check panel.

Runs a battery of checks (Cloudflare creds + rate limits, OpenAI key,
Ollama server/model/generation, active provider, local SQLite, buoy data
freshness, marine-forecast cache, Alexa briefing) and returns a list of
{name, status, detail, category} where status ∈ pass|warn|fail|skip|info.
"""
from __future__ import annotations

import time
from datetime import datetime

import pytz
import requests

from . import ai_provider, alexa, db, forecast, settings, usage
from .buoy_fetcher import BUOYS
from .envutil import getenv_ci

VAN_TZ = pytz.timezone("America/Vancouver")


def _r(name, status, detail, category="general"):
    return {"name": name, "status": status, "detail": detail, "category": category}


def _check_cloudflare():
    account = getenv_ci("CLOUDFLARE_ACCOUNT_ID")
    ns = getenv_ci("CLOUDFLARE_NAMESPACE_ID")
    tok = getenv_ci("CLOUDFLARE_API_TOKEN")
    missing = [n for n, v in [("ACCOUNT_ID", account), ("NAMESPACE_ID", ns),
                              ("API_TOKEN", tok)] if not v]
    if missing:
        return _r("Cloudflare credentials", "fail", f"missing env: {', '.join(missing)}", "cloudflare")
    try:
        url = (f"https://api.cloudflare.com/client/v4/accounts/{account}"
               f"/storage/kv/namespaces/{ns}/keys")
        r = requests.get(url, headers={"Authorization": f"Bearer {tok}"},
                         params={"limit": 10}, timeout=10)
        if r.status_code == 200 and r.json().get("success"):
            return _r("Cloudflare KV", "pass", "namespace reachable, token valid", "cloudflare")
        return _r("Cloudflare KV", "fail", f"HTTP {r.status_code}: {r.text[:150]}", "cloudflare")
    except Exception as e:
        return _r("Cloudflare KV", "fail", str(e)[:150], "cloudflare")


def _check_cf_rate_limits():
    try:
        snap = usage.snapshot()
        worst = "pass"
        bits = []
        for metric in ("reads", "writes", "lists"):
            blk = snap.get(metric) or {}
            used = blk.get("today_used", 0)
            limit = blk.get("today_limit", 0)
            pct = (used / limit * 100) if limit else 0
            bits.append(f"{metric} {used}/{limit} ({pct:.0f}%)")
            if blk.get("today_will_exceed"):
                worst = "fail"
            elif pct >= 80 and worst != "fail":
                worst = "warn"
        return _r("Cloudflare rate limits", worst, "today: " + ", ".join(bits), "cloudflare")
    except Exception as e:
        return _r("Cloudflare rate limits", "warn", f"usage data unavailable: {e}", "cloudflare")


def _check_openai():
    key = settings.get_openai_key()
    active = ai_provider.get_provider()
    if not key:
        sev = "fail" if active == "openai" else "warn"
        return _r("OpenAI key", sev, "no key set (Settings or OPENAI_API_KEY env)", "openai")
    try:
        r = requests.get("https://api.openai.com/v1/models",
                         headers={"Authorization": f"Bearer {key}"}, timeout=10)
        if r.status_code == 200:
            return _r("OpenAI key", "pass", f"{len(r.json().get('data', []))} models accessible", "openai")
        if r.status_code == 401:
            return _r("OpenAI key", "fail", "401 unauthorized (bad key)", "openai")
        return _r("OpenAI key", "fail", f"HTTP {r.status_code}: {r.text[:120]}", "openai")
    except Exception as e:
        return _r("OpenAI key", "fail", str(e)[:150], "openai")


def _check_ollama(run_generation):
    out = []
    st = ai_provider.ollama_status()
    active = ai_provider.get_provider()
    if not st.get("ok"):
        sev = "fail" if active == "ollama" else "warn"
        out.append(_r("Ollama server", sev,
                      f"unreachable at {st.get('url')}: {st.get('error')}", "ollama"))
        return out
    out.append(_r("Ollama server", "pass",
                  f"reachable at {st['url']} — {len(st['models'])} model(s)", "ollama"))
    model = ai_provider.get_ollama_model()
    if model in st["models"]:
        out.append(_r("Ollama model", "pass", f"'{model}' installed", "ollama"))
        if run_generation:
            try:
                t0 = time.time()
                res = ai_provider.chat(
                    [{"role": "user", "content": "Reply with the single word OK."}],
                    reason="healthcheck", source_data="healthcheck ping",
                    provider="ollama", model=model)
                dt = time.time() - t0
                ok = "ok" in (res.content or "").lower()
                out.append(_r("Ollama generation", "pass" if ok else "warn",
                              f"responded in {dt:.1f}s: {(res.content or '')[:60]!r}", "ollama"))
            except Exception as e:
                out.append(_r("Ollama generation", "fail", str(e)[:200], "ollama"))
        else:
            out.append(_r("Ollama generation", "skip",
                          "skipped — active provider is OpenAI", "ollama"))
    else:
        out.append(_r("Ollama model", "fail",
                      f"'{model}' NOT installed; available: {', '.join(st['models']) or 'none'}", "ollama"))
    return out


def _check_active_provider():
    p = ai_provider.get_provider()
    if p == "openai":
        has = bool(settings.get_openai_key())
        return _r("Active AI provider", "pass" if has else "fail",
                  "OpenAI" + ("" if has else " — but no key set!"), "provider")
    st = ai_provider.ollama_status()
    model = ai_provider.get_ollama_model()
    ok = st.get("ok") and model in st.get("models", [])
    return _r("Active AI provider", "pass" if ok else "fail",
              f"Ollama (model '{model}')" + ("" if ok else " — server or model not ready"), "provider")


def _check_db():
    try:
        stats = db.stats()
        total = sum(v.get("rows", 0) for v in stats.values())
        if total == 0:
            return _r("Local SQLite", "warn",
                      "no readings yet — run INIT / Reconcile to backfill from KV", "storage")
        return _r("Local SQLite", "pass",
                  f"{total} readings across {len(stats)} stations", "storage")
    except Exception as e:
        return _r("Local SQLite", "fail", str(e)[:150], "storage")


def _check_buoy_freshness():
    now = datetime.now(VAN_TZ)
    stale, fresh = [], 0
    try:
        stats = db.stats()
        for b in BUOYS:
            info = stats.get(b["id"])
            last_ts = info.get("last_ts") if info else None
            if not last_ts:
                stale.append(b["name"])
                continue
            try:
                last = datetime.fromisoformat(last_ts)
                if last.tzinfo is None:
                    last = VAN_TZ.localize(last)
                age_h = (now - last).total_seconds() / 3600
                if age_h > 3:
                    stale.append(f"{b['name']} ({age_h:.0f}h)")
                else:
                    fresh += 1
            except Exception:
                stale.append(b["name"])
        if stale:
            return _r("Buoy data freshness", "warn",
                      f"{fresh} fresh; stale/missing: {', '.join(stale)}", "data")
        return _r("Buoy data freshness", "pass",
                  f"all {fresh} stations updated within 3h", "data")
    except Exception as e:
        return _r("Buoy data freshness", "warn", str(e)[:150], "data")


def _check_forecast():
    try:
        fc = forecast.get_cached("howe_sound")
        if fc and fc.get("rows"):
            return _r("Marine forecast", "pass",
                      f"{len(fc['rows'])} rows cached (Howe Sound)", "forecast")
        return _r("Marine forecast", "info",
                  "not cached yet — fetched on demand (one AI call)", "forecast")
    except Exception as e:
        return _r("Marine forecast", "warn", str(e)[:150], "forecast")


def _check_alexa():
    try:
        speech = alexa.build_speech()
        if speech and "condition" in speech.lower():
            return _r("Alexa briefing", "pass",
                      f"{len(speech)} chars — '{speech[:70]}…'", "alexa")
        return _r("Alexa briefing", "warn", f"unexpected output: {speech[:80]}", "alexa")
    except Exception as e:
        return _r("Alexa briefing", "fail", str(e)[:150], "alexa")


def run_all():
    """Run every check. The Ollama generation test only runs when Ollama is
    the active provider (CPU inference is slow; no point waiting on it
    otherwise)."""
    run_gen = ai_provider.get_provider() == "ollama"
    checks = [
        _check_cloudflare(),
        _check_cf_rate_limits(),
        _check_openai(),
        *_check_ollama(run_gen),
        _check_active_provider(),
        _check_db(),
        _check_buoy_freshness(),
        _check_forecast(),
        _check_alexa(),
    ]
    summary = {"pass": 0, "warn": 0, "fail": 0, "skip": 0, "info": 0}
    for c in checks:
        summary[c["status"]] = summary.get(c["status"], 0) + 1
    return {
        "ran_at": datetime.now(VAN_TZ).isoformat(timespec="seconds"),
        "summary": summary,
        "checks": checks,
    }
