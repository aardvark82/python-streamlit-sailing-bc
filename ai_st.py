"""AI provider + settings + log for the Streamlit app.

Mirrors the helper-app's ai_provider/ai_log: a single chat() entry point
that routes to OpenAI or a (hosted) Ollama server based on a persisted
setting, logging every call (tokens, cost, time, source data).

Persistence: a JSON file in the app dir. On Streamlit Cloud this is
ephemeral (resets on redeploy) which is fine for a provider toggle;
the OpenAI key still comes from st.secrets.
"""
from __future__ import annotations

import json
import os
import threading
import time as _time
from datetime import datetime
from pathlib import Path

import pytz
import requests
import streamlit as st

VAN_TZ = pytz.timezone("America/Vancouver")


def _pick_writable_dir() -> Path:
    """Where to persist ai_settings.json / ai_log.json. Prefer an explicit
    AI_DATA_DIR, else a dedicated temp subdir. We deliberately do NOT write
    into the app/git directory — on Streamlit Cloud it can be read-only and
    writes there could trip the file watcher into reruns."""
    import tempfile
    default_tmp = Path(tempfile.gettempdir()) / "pysail_ai"
    for cand in [os.environ.get("AI_DATA_DIR"), str(default_tmp), tempfile.gettempdir()]:
        if not cand:
            continue
        try:
            p = Path(cand)
            p.mkdir(parents=True, exist_ok=True)
            test = p / ".ai_write_test"
            test.write_text("ok")
            test.unlink()
            return p
        except Exception:
            continue
    return default_tmp


_DIR = _pick_writable_dir()
_SETTINGS_PATH = _DIR / "ai_settings.json"
_LOG_PATH = _DIR / "ai_log.json"
_lock = threading.Lock()
_MAX_LOG = 1000

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OLLAMA_MODEL = "qwen2.5:3b"

# USD per 1M tokens (OpenAI only; Ollama is $0).
PRICING = {
    "gpt-5-mini": {"in": 0.25, "out": 2.00},
    "gpt-5-nano": {"in": 0.05, "out": 0.40},
    "gpt-4o": {"in": 2.50, "out": 10.00},
    "gpt-4o-mini": {"in": 0.15, "out": 0.60},
}


# ── settings ──────────────────────────────────────────────────────────

def load_settings() -> dict:
    try:
        return json.loads(_SETTINGS_PATH.read_text() or "{}")
    except Exception:
        return {}


def save_settings(updates: dict) -> dict:
    with _lock:
        cur = load_settings()
        cur.update({k: v for k, v in updates.items() if v is not None})
        try:
            _SETTINGS_PATH.write_text(json.dumps(cur, indent=2))
        except Exception as e:
            print(f"ai_st: could not persist settings: {e}")
        return cur


def get_provider() -> str:
    return (load_settings().get("ai_provider") or "openai").strip().lower()


def get_openai_model() -> str:
    return load_settings().get("openai_model") or DEFAULT_OPENAI_MODEL


def get_ollama_model() -> str:
    return load_settings().get("ollama_model") or DEFAULT_OLLAMA_MODEL


def _normalize_ollama_url(raw: str) -> str:
    raw = (raw or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    return raw


def get_ollama_url() -> str:
    return _normalize_ollama_url(load_settings().get("ollama_url") or os.environ.get("OLLAMA_URL") or "")


def get_ollama_api_key() -> str:
    return (load_settings().get("ollama_api_key") or os.environ.get("OLLAMA_API_KEY") or "").strip()


def get_openai_key():
    # Same secret name the rest of the app uses.
    try:
        return st.secrets["OpenAI_key"]
    except Exception:
        return None


# ── log ───────────────────────────────────────────────────────────────

def _load_log() -> list:
    try:
        return json.loads(_LOG_PATH.read_text() or "[]")
    except Exception:
        return []


def _save_log(entries: list):
    try:
        _LOG_PATH.write_text(json.dumps(entries[-_MAX_LOG:], indent=2))
    except Exception as e:
        print(f"ai_st: could not persist log: {e}")


def cost_for(provider, model, pt, ct) -> float:
    if provider != "openai":
        return 0.0
    p = PRICING.get(model, PRICING["gpt-5-mini"])
    return pt / 1e6 * p["in"] + ct / 1e6 * p["out"]


def _record(*, reason, provider, model, source_data, pt, ct, elapsed):
    cost = cost_for(provider, model, pt, ct)
    entry = {
        "ts": datetime.now(VAN_TZ).isoformat(timespec="seconds"),
        "reason": reason, "provider": provider, "model": model,
        "source_data": source_data or reason,
        "prompt_tokens": int(pt), "completion_tokens": int(ct),
        "total_tokens": int(pt + ct),
        "elapsed_sec": round(float(elapsed), 2) if elapsed is not None else None,
        "cost_usd": round(cost, 6),
    }
    with _lock:
        log = _load_log()
        log.append(entry)
        _save_log(log)
    return entry


def log_snapshot(limit: int = 200) -> dict:
    log = _load_log()
    total_tokens = sum(e.get("total_tokens", 0) for e in log)
    total_cost = sum(e.get("cost_usd", 0.0) for e in log)
    est_monthly = None
    if log:
        try:
            first = datetime.fromisoformat(log[0]["ts"])
            span_days = max((datetime.now(VAN_TZ) - first).total_seconds() / 86400.0, 1 / 24.0)
            est_monthly = total_cost / span_days * 30.0
        except Exception:
            pass
    by_provider = {}
    for e in log:
        p = e.get("provider", "openai")
        b = by_provider.setdefault(p, {"calls": 0, "tokens": 0, "cost_usd": 0.0})
        b["calls"] += 1
        b["tokens"] += e.get("total_tokens", 0)
        b["cost_usd"] += e.get("cost_usd", 0.0)
    for b in by_provider.values():
        b["cost_usd"] = round(b["cost_usd"], 4)
    return {
        "total_calls": len(log),
        "total_tokens": total_tokens,
        "total_cost_usd": round(total_cost, 4),
        "est_monthly_cost_usd": round(est_monthly, 4) if est_monthly is not None else None,
        "by_provider": by_provider,
        "entries": list(reversed(log))[:limit],
    }


def reset_log() -> int:
    with _lock:
        n = len(_load_log())
        _save_log([])
    return n


# ── providers ─────────────────────────────────────────────────────────

def _ollama_headers(json_body=False) -> dict:
    h = {}
    if json_body:
        h["Content-Type"] = "application/json"
    key = get_ollama_api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


def ollama_status() -> dict:
    url = get_ollama_url()
    if not url:
        return {"ok": False, "url": "", "error": "no Ollama URL set", "models": []}
    try:
        r = requests.get(f"{url}/api/tags", headers=_ollama_headers(), timeout=4)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
        return {"ok": True, "url": url, "models": models}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e), "models": []}


def _openai_chat(messages, *, reason, source_data, model):
    key = get_openai_key()
    if not key:
        raise RuntimeError("OpenAI key not set in st.secrets['OpenAI_key']")
    t0 = _time.time()
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages}, timeout=600)
    r.raise_for_status()
    body = r.json()
    usage = body.get("usage", {}) or {}
    entry = _record(reason=reason, provider="openai", model=model, source_data=source_data,
                    pt=usage.get("prompt_tokens", 0), ct=usage.get("completion_tokens", 0),
                    elapsed=_time.time() - t0)
    return body["choices"][0]["message"]["content"], entry


def _ollama_chat(messages, *, reason, source_data, model):
    url = get_ollama_url()
    if not url:
        raise RuntimeError("Ollama URL not set (Settings → AI provider)")
    t0 = _time.time()
    try:
        r = requests.post(f"{url}/api/chat", headers=_ollama_headers(json_body=True),
                          json={"model": model, "messages": messages, "stream": False}, timeout=600)
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Ollama unreachable at {url}: {e}") from e
    if r.status_code == 404:
        detail = ""
        try:
            detail = (r.json() or {}).get("error", "")
        except Exception:
            detail = (r.text or "")[:150]
        raise RuntimeError(f"Model '{model}' not found on Ollama. Pull it first. {detail}")
    r.raise_for_status()
    body = r.json()
    content = (body.get("message") or {}).get("content", "") or body.get("response", "")
    entry = _record(reason=reason, provider="ollama", model=model, source_data=source_data,
                    pt=body.get("prompt_eval_count", 0), ct=body.get("eval_count", 0),
                    elapsed=_time.time() - t0)
    return content, entry


def chat(messages, *, reason, source_data=None, provider=None, model=None):
    """Single chat-completion call routed by Settings. Returns the content
    string (logging handled internally)."""
    provider = (provider or get_provider()).lower()
    if provider == "ollama":
        content, _ = _ollama_chat(messages, reason=reason, source_data=source_data,
                                  model=model or get_ollama_model())
    else:
        content, _ = _openai_chat(messages, reason=reason, source_data=source_data,
                                  model=model or get_openai_model())
    return content


# ── Streamlit UI ──────────────────────────────────────────────────────

def render_settings_page(container=None):
    draw = container or st
    draw.title("⚙️ AI / LLM Settings")
    draw.caption("Choose the model and server used to parse marine forecasts. "
                 "Switch between OpenAI (cloud) and a hosted Ollama server.")

    s = load_settings()
    provider = draw.radio(
        "Provider", ["openai", "ollama"],
        index=0 if get_provider() == "openai" else 1,
        format_func=lambda p: "☁️ OpenAI (cloud)" if p == "openai" else "🖥 Ollama (hosted)",
        horizontal=True,
    )

    draw.markdown("**OpenAI**")
    openai_model = draw.text_input("OpenAI model", value=get_openai_model(),
                                   help="e.g. gpt-5-mini, gpt-4o-mini")
    draw.caption("OpenAI key comes from `st.secrets['OpenAI_key']`: "
                 + ("✅ set" if get_openai_key() else "❌ not set"))

    draw.markdown("**Ollama (hosted)**")
    ollama_url = draw.text_input("Ollama endpoint", value=s.get("ollama_url", ""),
                                 placeholder="https://ollama.example.com:11434 or http://host:11434",
                                 help="A trailing /v1 is auto-stripped — native API is used.")
    ollama_key = draw.text_input("API key (optional, Bearer)", value="", type="password",
                                 placeholder="•••• set (leave blank to keep)" if get_ollama_api_key() else "none")

    if draw.button("🔄 Load models / test endpoint"):
        save_settings({"ollama_url": ollama_url,
                       **({"ollama_api_key": ollama_key} if ollama_key.strip() else {})})
        st.session_state["_ollama_status"] = ollama_status()

    stt = st.session_state.get("_ollama_status")
    models = []
    if stt:
        if stt["ok"]:
            draw.success(f"✓ reachable at {stt['url']} — {len(stt['models'])} model(s)")
            models = stt["models"]
        else:
            draw.error(f"✗ {stt['error']}")

    if models:
        cur = get_ollama_model()
        idx = models.index(cur) if cur in models else 0
        ollama_model = draw.selectbox("Ollama model", models, index=idx)
    else:
        ollama_model = draw.text_input("Ollama model", value=get_ollama_model(),
                                       help="Load models above, or type one. Use a TEXT model (qwen2.5:3b), not vl/-vision.")

    if draw.button("💾 Save settings", type="primary"):
        save_settings({
            "ai_provider": provider,
            "openai_model": openai_model,
            "ollama_url": ollama_url,
            "ollama_model": ollama_model,
            **({"ollama_api_key": ollama_key} if ollama_key.strip() else {}),
        })
        draw.success(f"Saved — provider: {provider}. Next forecast parse uses it "
                     "(cache keyed on provider, so it refreshes automatically).")

    draw.divider()
    if draw.button("🧪 Test with Howe Sound forecast"):
        with draw.spinner("Parsing Howe Sound forecast with the selected provider…"):
            try:
                from fetch_forecast import openAIFetchForecastForURL
                out = openAIFetchForecastForURL(
                    'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400')
                draw.code(out, language="csv")
                last = log_snapshot(1)["entries"]
                if last:
                    e = last[0]
                    draw.caption(f"{e['provider']} · {e['model']} · "
                                 f"{e['prompt_tokens']}/{e['completion_tokens']} tok · "
                                 f"{e.get('elapsed_sec','?')}s · ${e['cost_usd']:.5f}")
            except Exception as e:
                draw.error(f"Test failed: {e}")


def render_ai_log(container=None):
    draw = container or st
    snap = log_snapshot()
    draw.subheader("🤖 AI query log")
    c1, c2, c3 = draw.columns(3)
    c1.metric("AI calls", snap["total_calls"])
    c2.metric("Total cost", f"${snap['total_cost_usd']:.4f}")
    c3.metric("Est. / month", f"${snap['est_monthly_cost_usd']:.4f}"
              if snap["est_monthly_cost_usd"] is not None else "—")

    if snap.get("by_provider"):
        bits = [f"{p}: {b['calls']} calls / {b['tokens']:,} tok / ${b['cost_usd']:.4f}"
                for p, b in snap["by_provider"].items()]
        draw.caption(" · ".join(bits))

    if snap["entries"]:
        import pandas as pd
        df = pd.DataFrame(snap["entries"])
        cols = [c for c in ["ts", "provider", "model", "source_data",
                            "prompt_tokens", "completion_tokens", "elapsed_sec", "cost_usd"]
                if c in df.columns]
        # NOTE: omit width= on st.dataframe (this Streamlit raises on str width).
        draw.dataframe(df[cols], hide_index=True)
    else:
        draw.caption("No AI calls logged yet — they appear here after a forecast is parsed.")

    if draw.button("Reset AI log"):
        n = reset_log()
        draw.success(f"Cleared {n} entries.")
