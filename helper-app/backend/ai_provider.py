"""Single chat-completion façade for OpenAI + Ollama.

Callers don't pick a provider — they just call ai_provider.chat(...) and
the routing happens via Settings (ai_provider: 'openai' | 'ollama').
Every call is logged through ai_log.record so the Settings → AI Log
panel reflects both providers identically.
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass
from typing import Optional

import requests

from . import ai_log, settings
from .envutil import getenv_ci

log = logging.getLogger("helper.ai")

DEFAULT_OPENAI_MODEL = "gpt-5-mini"
DEFAULT_OLLAMA_MODEL = "llama3.2:3b"

# Default Ollama endpoint (the standalone Coolify resource, joined via the
# predefined network). Overridable in Settings (persisted) or OLLAMA_URL env.
DEFAULT_OLLAMA_URL = "http://ollama-api-gkv762rc1urjlhnvu8y4m5vb:11434"


def _normalize_ollama_url(raw: str) -> str:
    """Trim to the native-API base. We call /api/tags, /api/chat, /api/pull
    which live at the ROOT — so strip a trailing '/v1' (the OpenAI-compat
    path) and any trailing slash if the user pasted that form."""
    raw = (raw or "").strip().rstrip("/")
    if raw.endswith("/v1"):
        raw = raw[:-3].rstrip("/")
    return raw


def get_ollama_url() -> str:
    """Resolve the Ollama base URL: Settings (persisted) → OLLAMA_URL env →
    default. Read live so editing it in the UI takes effect on the next
    call without a container restart."""
    raw = settings.load().get("ollama_url") or getenv_ci("OLLAMA_URL") or DEFAULT_OLLAMA_URL
    return _normalize_ollama_url(raw)


def get_ollama_api_key() -> str:
    """Optional Bearer token for an authed/proxied Ollama. Settings →
    OLLAMA_API_KEY env. Empty string when not configured."""
    return (settings.load().get("ollama_api_key") or getenv_ci("OLLAMA_API_KEY") or "").strip()


def _ollama_headers(json_body: bool = False) -> dict:
    h = {}
    if json_body:
        h["Content-Type"] = "application/json"
    key = get_ollama_api_key()
    if key:
        h["Authorization"] = f"Bearer {key}"
    return h


@dataclass
class AIResult:
    content: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_sec: float
    cost_usd: float


def get_provider() -> str:
    """Returns 'openai' (default) or 'ollama'."""
    return (settings.load().get("ai_provider") or "openai").strip().lower()


def get_openai_model() -> str:
    return settings.load().get("openai_model") or DEFAULT_OPENAI_MODEL


def get_ollama_model() -> str:
    return settings.load().get("ollama_model") or DEFAULT_OLLAMA_MODEL


# ── OpenAI ────────────────────────────────────────────────────────────

def _openai_chat(messages, *, reason, source_data, model):
    key = settings.get_openai_key()
    if not key:
        raise RuntimeError("OpenAI key not set (Settings tab or OPENAI_API_KEY env)")
    t0 = _time.time()
    r = requests.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
        json={"model": model, "messages": messages},
        timeout=600,
    )
    r.raise_for_status()
    body = r.json()
    elapsed = _time.time() - t0
    usage = body.get("usage", {}) or {}
    content = body["choices"][0]["message"]["content"]
    entry = ai_log.record(
        reason=reason, source_data=source_data,
        provider="openai", model=model,
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        elapsed_sec=elapsed,
    )
    return AIResult(
        content=content, provider="openai", model=model,
        prompt_tokens=entry["prompt_tokens"],
        completion_tokens=entry["completion_tokens"],
        elapsed_sec=elapsed, cost_usd=entry["cost_usd"],
    )


# ── Ollama ────────────────────────────────────────────────────────────

def ollama_status() -> dict:
    """Cheap reachability + installed-models probe for the Settings UI."""
    url = get_ollama_url()
    try:
        r = requests.get(f"{url}/api/tags", headers=_ollama_headers(), timeout=3)
        r.raise_for_status()
        models = [m.get("name") for m in r.json().get("models", []) if m.get("name")]
        return {"ok": True, "url": url, "models": models}
    except Exception as e:
        return {"ok": False, "url": url, "error": str(e), "models": []}


def ollama_pull(model: str) -> dict:
    """Trigger model download. Streaming progress is collapsed to a
    final ok/error to keep the HTTP response simple."""
    url = get_ollama_url()
    try:
        with requests.post(f"{url}/api/pull", headers=_ollama_headers(json_body=True),
                           json={"name": model, "stream": False},
                           timeout=1800) as r:
            r.raise_for_status()
        return {"ok": True, "model": model}
    except Exception as e:
        return {"ok": False, "model": model, "error": str(e)}


def _ollama_chat(messages, *, reason, source_data, model):
    # Ollama supports the OpenAI-compatible /v1/chat/completions endpoint
    # AND a native /api/chat. The native one returns token counts more
    # reliably across model versions, so we use it.
    url = get_ollama_url()
    t0 = _time.time()
    try:
        r = requests.post(
            f"{url}/api/chat", headers=_ollama_headers(json_body=True),
            json={"model": model, "messages": messages, "stream": False},
            timeout=600,    # CPU inference is slow; first call may pull the model
        )
    except requests.exceptions.ConnectionError as e:
        raise RuntimeError(f"Ollama unreachable at {url} ({e}). "
                            "Check the Ollama endpoint in Settings.") from e

    # Ollama returns 404 on /api/chat when the requested MODEL isn't pulled.
    # Surface its actual message instead of a bare '404 Not Found'.
    if r.status_code == 404:
        detail = ""
        try:
            detail = (r.json() or {}).get("error", "")
        except Exception:
            detail = (r.text or "")[:200]
        raise RuntimeError(
            f"Model '{model}' not found on the Ollama server ({url}). "
            f"Pull it first (Settings → Pull, or `ollama pull {model}`), or set the "
            f"Ollama model field to one that's installed. Server said: {detail or '404'}"
        )
    r.raise_for_status()
    body = r.json()
    elapsed = _time.time() - t0
    content = (body.get("message") or {}).get("content", "") or body.get("response", "")
    prompt_tokens = body.get("prompt_eval_count", 0)
    completion_tokens = body.get("eval_count", 0)
    entry = ai_log.record(
        reason=reason, source_data=source_data,
        provider="ollama", model=model,
        prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
        elapsed_sec=elapsed,
    )
    return AIResult(
        content=content, provider="ollama", model=model,
        prompt_tokens=entry["prompt_tokens"],
        completion_tokens=entry["completion_tokens"],
        elapsed_sec=elapsed, cost_usd=0.0,
    )


# ── Public entry point ────────────────────────────────────────────────

def chat(messages, *, reason: str, source_data: Optional[str] = None,
         provider: Optional[str] = None, model: Optional[str] = None) -> AIResult:
    """Single chat-completion call. provider/model can be forced for
    one-off operations; otherwise both come from Settings."""
    provider = (provider or get_provider()).lower()
    if provider == "ollama":
        return _ollama_chat(messages, reason=reason, source_data=source_data,
                             model=model or get_ollama_model())
    return _openai_chat(messages, reason=reason, source_data=source_data,
                         model=model or get_openai_model())
