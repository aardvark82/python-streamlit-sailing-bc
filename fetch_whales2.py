"""Whale-watching boat tracker — VesselAPI Ship Tracking variant.

Uses VesselAPI's Ship Tracking REST API
(https://vesselapi.com/ship-tracking-api) to look up fleet vessels by name
and then batch-fetch their latest reported positions in one call. Position
fetches are gated behind a manual 'Fetch positions' button to limit billable
API usage; the page also tracks the running call count.
"""

import json
from datetime import datetime, timedelta
from urllib.parse import quote

import requests
import streamlit as st
import pandas as pd
import pytz
import plotly.graph_objects as go


VESSELAPI_BASE = "https://api.vesselapi.com/v1"
SOURCE_URL = "https://vesselapi.com/ship-tracking-api"


def _pos_field(pos, *names):
    """Read a field from a position record, tolerating both snake_case
    (vessel_name) and camelCase (vesselName) shapes returned by the API."""
    if not pos:
        return None
    for n in names:
        if n in pos and pos[n] is not None:
            return pos[n]
    return None

# Curated fleet. `mmsi` is optional — when present we skip the unreliable
# search-by-name and call /vessel/{mmsi}/position directly. MMSIs collected
# from cross-checking with MyShipTracking iOS.
# `region` lets us filter the fleet for cheaper region-only refreshes.
WHALE_FLEET = [
    {'name': 'Aurora I',         'operator': 'Wild Whales Vancouver',  'region': 'Vancouver',
     'icon_color': '#1f77b4'},
    {'name': 'Aurora II',        'operator': 'Wild Whales Vancouver',  'region': 'Vancouver',
     'icon_color': '#1f77b4'},
    {'name': 'Eagle Eyes',       'operator': 'Wild Whales Vancouver',  'region': 'Vancouver',
     'icon_color': '#1f77b4'},
    {'name': 'Jing Yu',          'operator': 'Wild Whales Vancouver',  'region': 'Vancouver',
     'icon_color': '#1f77b4'},
    {'name': 'Explorathor II',   'operator': 'Vancouver Whale Watch',  'region': 'Vancouver',
     'icon_color': '#2ca02c'},
    {'name': 'Express',          'operator': 'Vancouver Whale Watch',  'region': 'Vancouver',
     'icon_color': '#2ca02c'},
    {'name': 'Strider',          'operator': 'Vancouver Whale Watch',  'region': 'Vancouver',
     'icon_color': '#2ca02c'},
    {'name': 'Lightship',        'operator': 'Vancouver Whale Watch',  'region': 'Vancouver',
     'icon_color': '#2ca02c'},
    {'name': 'Salish Sea Dream',   'operator': 'Prince of Whales', 'region': 'Vancouver',
     'icon_color': '#ff7f0e'},
    {'name': 'Salish Sea Freedom', 'operator': 'Prince of Whales', 'region': 'Vancouver',
     'icon_color': '#ff7f0e', 'mmsi': 316042213},
    {'name': 'Salish Sea Eclipse', 'operator': 'Prince of Whales', 'region': 'Victoria',
     'icon_color': '#ff7f0e'},
    {'name': 'Ocean Magic II',     'operator': 'Prince of Whales', 'region': 'Telegraph Cove',
     'icon_color': '#ff7f0e'},
    {'name': 'Ocean Magic',        'operator': 'Prince of Whales', 'region': 'Telegraph Cove',
     'icon_color': '#ff7f0e'},
]


def _api_keys_list():
    """Return VesselAPI keys in priority order: primary first, then backup.
    Empty list when no keys configured."""
    keys = []
    primary = None
    for key_name in ('vesselapi_key', 'vessel_api_key', 'vesselapi-com_key'):
        try:
            primary = st.secrets[key_name]
            break
        except (KeyError, FileNotFoundError):
            continue
    if primary:
        keys.append(('primary', primary))
    try:
        backup = st.secrets['vesselapi2_key']
        if backup:
            keys.append(('backup', backup))
    except (KeyError, FileNotFoundError):
        pass
    return keys


def _http_get(url, params=None, timeout=15):
    """GET helper that automatically falls back from primary→backup VesselAPI
    key on HTTP 429 (rate limit). Returns (response_or_None, key_label_used).
    Raises RuntimeError if no keys are configured. The caller is responsible
    for inspecting response.status_code on the returned response."""
    keys = _api_keys_list()
    if not keys:
        raise RuntimeError("No VesselAPI keys configured (set vesselapi_key in secrets).")
    last_response = None
    for label, key in keys:
        try:
            r = requests.get(
                url, params=params,
                headers={'Authorization': f'Bearer {key}'},
                timeout=timeout,
            )
        except Exception as e:
            print(f"VesselAPI {label} key request error: {e}")
            continue
        last_response = r
        if r.status_code == 429:
            # Mark this key rate-limited so the UI can surface it.
            st.session_state.setdefault('vesselapi_rate_limited_keys', set()).add(label)
            print(f"VesselAPI {label} key rate-limited; trying next key.")
            continue
        return r, label
    return last_response, 'rate_limited_all'


# ──────────────────────────────────────────────
# Cloudflare KV-backed call tracking (so the count persists across users
# and reruns, not just this session). Reuses the same KV namespace already
# powering the buoy wind-history feature.
# ──────────────────────────────────────────────

VESSELAPI_KV_PREFIX = "vesselapi_call_"


def _kv_credentials():
    """Return Cloudflare KV creds dict or None when not configured."""
    try:
        return {
            'account_id': st.secrets["cloudflare_account_id"],
            'namespace_id_raw': st.secrets["cloudflare_namespace_id"],
            'api_token': st.secrets["cloudflare_api_token"],
        }
    except (KeyError, FileNotFoundError):
        return None


@st.cache_data(ttl=3600, show_spinner=False)
def _kv_resolve_namespace(account_id, api_token, name_or_id):
    """Cache namespace-id resolution for an hour so we don't hammer the API."""
    if "storage/kv/namespaces/" in name_or_id:
        return name_or_id
    headers = {"Authorization": f"Bearer {api_token}"}
    url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces"
    try:
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        for ns in r.json().get("result", []):
            if ns.get("title") == name_or_id:
                return ns.get("id")
    except Exception as e:
        print(f"KV namespace resolve failed: {e}")
    return name_or_id


def _record_vesselapi_call(label=''):
    """Write a tiny key to Cloudflare KV recording one billable VesselAPI call.
    Key format: vesselapi_call_<utc_iso_with_dashes>. Failures are silent."""
    creds = _kv_credentials()
    if not creds:
        return
    namespace_id = _kv_resolve_namespace(
        creds['account_id'], creds['api_token'], creds['namespace_id_raw']
    )
    base_url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{creds['account_id']}/storage/kv/namespaces/{namespace_id}"
    )
    now_utc = datetime.utcnow()
    # Use dashes in time so the key is filename-safe and unambiguous to parse.
    key = f"{VESSELAPI_KV_PREFIX}{now_utc.strftime('%Y-%m-%dT%H-%M-%S-%f')}"
    try:
        requests.put(
            f"{base_url}/values/{quote(key, safe='')}",
            headers={"Authorization": f"Bearer {creds['api_token']}"},
            data=(label or '1'),
            timeout=10,
        )
    except Exception as e:
        print(f"VesselAPI KV record failed: {e}")


@st.cache_data(ttl=300, show_spinner=False)
def _count_vesselapi_calls_last_30_days():
    """List Cloudflare KV keys with prefix vesselapi_call_ and count those
    whose timestamp falls within the last 30 days. Cached 5 minutes."""
    creds = _kv_credentials()
    if not creds:
        return None
    namespace_id = _kv_resolve_namespace(
        creds['account_id'], creds['api_token'], creds['namespace_id_raw']
    )
    base_url = (
        f"https://api.cloudflare.com/client/v4/accounts/"
        f"{creds['account_id']}/storage/kv/namespaces/{namespace_id}"
    )
    headers = {"Authorization": f"Bearer {creds['api_token']}"}
    cutoff = datetime.utcnow() - timedelta(days=30)
    count = 0
    cursor = None
    pages = 0
    try:
        while pages < 30:  # hard ceiling — 30k keys should be more than enough
            pages += 1
            params = {'prefix': VESSELAPI_KV_PREFIX, 'limit': 1000}
            if cursor:
                params['cursor'] = cursor
            r = requests.get(f"{base_url}/keys", params=params, headers=headers, timeout=15)
            r.raise_for_status()
            payload = r.json() or {}
            for item in payload.get('result', []):
                key = item.get('name', '')
                ts_str = key.replace(VESSELAPI_KV_PREFIX, '', 1)
                try:
                    date_part, time_part = ts_str.split('T', 1)
                    h, m, s, _us = time_part.split('-', 3)
                    dt = datetime.fromisoformat(f"{date_part}T{h}:{m}:{s}")
                    if dt >= cutoff:
                        count += 1
                except Exception:
                    continue
            cursor = (payload.get('result_info') or {}).get('cursor')
            if not cursor:
                break
    except Exception as e:
        print(f"VesselAPI KV count failed: {e}")
        return None
    return count


def _bump_count(n=1, label=''):
    """Increment the persistent KV-backed counter (and the session counter as a
    quick local mirror). One KV write per call so /vessels/positions usage is
    tracked across every user and rerun, not just this session."""
    st.session_state.setdefault('vesselapi_request_count', 0)
    st.session_state['vesselapi_request_count'] += n
    for _ in range(n):
        _record_vesselapi_call(label)


@st.cache_data(ttl=86400, show_spinner=False)
def _search_vessel_by_name(name):
    """Search VesselAPI by vessel name. Tries primary key, falls back to
    backup on 429. Cached per name for 24h.
    Returns (mmsi, raw_record, diagnostic_payload)."""
    url = f"{VESSELAPI_BASE}/search/vessels"
    params = {'filter.name': name, 'pagination.limit': 5}
    try:
        r, key_used = _http_get(url, params=params)
        if r is None:
            return None, None, {'error': 'no response', 'request_url': url, 'params': params}
        if r.status_code == 429:
            return None, None, {
                'error': 'all VesselAPI keys rate-limited (429)',
                'request_url': r.url, 'key_used': key_used,
            }
        if r.status_code != 200:
            return None, None, {
                'error': f"HTTP {r.status_code}",
                'request_url': r.url,
                'body': r.text[:500],
                'key_used': key_used,
            }
        data = r.json() or {}
        vessels = data.get('vessels') or []
        diag = {'request_url': r.url, 'raw': data, 'key_used': key_used}
        if not vessels:
            return None, None, diag
        target = name.strip().lower()
        best = None
        for v in vessels:
            if (v.get('name') or '').strip().lower() == target:
                best = v
                break
        best = best or vessels[0]
        return best.get('mmsi'), best, diag
    except Exception as e:
        return None, None, {'error': str(e), 'request_url': url, 'params': params}


def _get_vessel_position(mmsi):
    """Fetch latest position for ONE MMSI via /vessel/{id}/position.
    Tries primary VesselAPI key, falls back to backup on 429.
    Returns (position_record_or_None, diagnostic_payload).
    Each call counts as 1 billable API request."""
    url = f"{VESSELAPI_BASE}/vessel/{mmsi}/position"
    params = {'filter.idType': 'mmsi'}
    try:
        r, key_used = _http_get(url, params=params)
        if r is None:
            return None, {'error': 'no response', 'request_url': url, 'params': params}
        if r.status_code == 429:
            return None, {
                'error': 'all VesselAPI keys rate-limited (429)',
                'request_url': r.url, 'key_used': key_used,
            }
        if r.status_code != 200:
            return None, {
                'error': f"HTTP {r.status_code}",
                'request_url': r.url,
                'body': r.text[:500],
                'key_used': key_used,
            }
        data = r.json() or {}
        pos = (
            data.get('vesselPosition')
            or data.get('vessel')
            or data.get('position')
            or data
        )
        return pos, {'request_url': r.url, 'raw': data, 'key_used': key_used}
    except Exception as e:
        return None, {'error': str(e), 'request_url': url, 'params': params}


def _format_age(iso_ts, now_van):
    """Return a friendly relative-time string like '2min ago' or '3 hours ago'."""
    if not iso_ts:
        return ''
    try:
        # Accept '2026-01-15T12:30:00Z', '+00:00', or trailing fractional seconds.
        s = str(iso_ts).replace('Z', '+00:00')
        dt = datetime.fromisoformat(s)
        delta = now_van - dt.astimezone(now_van.tzinfo)
        secs = int(delta.total_seconds())
        if secs < 0:
            return 'just now'
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            mins = secs // 60
            return f"{mins}min ago"
        if secs < 86400:
            hours = secs // 3600
            return f"{hours} hour ago" if hours == 1 else f"{hours} hours ago"
        days = secs // 86400
        return f"{days} day ago" if days == 1 else f"{days} days ago"
    except Exception:
        return ''


def _resolve_mmsi(boat_name, diagnostics):
    """Resolve MMSI for one boat with this priority:
       1. Session-level user override (set via the override UI).
       2. Hard-coded MMSI on the WHALE_FLEET entry.
       3. Cached search-by-name (least reliable for short generic names).
    Returns (mmsi, meta_or_none)."""
    # 1. Session override
    overrides = st.session_state.get('vesselapi_mmsi_overrides') or {}
    if boat_name in overrides and overrides[boat_name]:
        try:
            return int(overrides[boat_name]), {'name': boat_name, 'source': 'manual_override'}
        except (TypeError, ValueError):
            pass

    # 2. Hard-coded MMSI on the fleet entry
    fleet_entry = next((b for b in WHALE_FLEET if b['name'] == boat_name), None)
    if fleet_entry and fleet_entry.get('mmsi'):
        return int(fleet_entry['mmsi']), {'name': boat_name, 'source': 'fleet_hardcoded'}

    # 3. Cached search-by-name fallback (handles primary→backup key on 429)
    if 'vesselapi_searched_names' not in st.session_state:
        st.session_state['vesselapi_searched_names'] = set()
    seen = st.session_state['vesselapi_searched_names']
    if boat_name not in seen:
        _bump_count(1)
        seen.add(boat_name)
    mmsi, meta, raw = _search_vessel_by_name(boat_name)
    diagnostics.setdefault('searches', {})[boat_name] = raw
    return (int(mmsi) if mmsi is not None else None, meta)


def _build_result(boat, mmsi, meta, pos):
    last_seen_iso = _pos_field(
        pos, 'timestamp', 'lastSeen', 'last_seen', 'processed_timestamp')
    return {
        **boat,
        'mmsi': mmsi,
        'position': pos,
        'meta': meta,
        'last_seen_iso': last_seen_iso,
    }


def _do_fetch_one(boat):
    """Resolve MMSI then fetch the latest position for ONE fleet boat using
    /vessel/{id}/position. Returns (result_dict, diagnostics_dict)."""
    diagnostics = {'searches': {}, 'positions': {}}
    mmsi, meta = _resolve_mmsi(boat['name'], diagnostics)
    pos = None
    if mmsi is not None:
        _bump_count(1)
        pos, pos_diag = _get_vessel_position(mmsi)
        diagnostics['positions'][boat['name']] = pos_diag
    return _build_result(boat, mmsi, meta, pos), diagnostics


def _do_fetch_all(region=None):
    """Resolve fleet MMSIs (cached) then fetch each position via the
    single-vessel /vessel/{id}/position endpoint — one call per boat.
    Pass `region=...` to limit the fetch to a subset (e.g. 'Vancouver').
    Returns (results_list, diagnostics_dict)."""
    diagnostics = {'searches': {}, 'positions': {}}
    results = []
    for boat in WHALE_FLEET:
        if region and boat.get('region') != region:
            continue
        mmsi, meta = _resolve_mmsi(boat['name'], diagnostics)
        pos = None
        if mmsi is not None:
            _bump_count(1)
            pos, pos_diag = _get_vessel_position(mmsi)
            diagnostics['positions'][boat['name']] = pos_diag
        results.append(_build_result(boat, mmsi, meta, pos))
    return results, diagnostics


def _merge_results(prior, new_results):
    """Patch prior fleet results with new entries (matched by boat name)
    so a region-only refresh updates only those rows, leaving others as-is."""
    if not prior:
        return list(new_results)
    by_name = {r['name']: r for r in new_results}
    out = []
    for r in prior:
        if r.get('name') in by_name:
            out.append(by_name[r['name']])
        else:
            out.append(r)
    # Append any new entries not previously present
    prior_names = {r.get('name') for r in prior}
    for r in new_results:
        if r['name'] not in prior_names:
            out.append(r)
    return out


def display_whales2_page(container=None):
    """Render the VesselAPI-backed whale tracker."""
    draw = container or st

    draw.subheader("🐋 Whale boats 2 — VesselAPI Ship Tracking")
    draw.markdown(
        f"**Source:** [{SOURCE_URL}]({SOURCE_URL}) — Ship Tracking REST lookups by name + MMSI."
    )

    keys_available = _api_keys_list()
    if not keys_available:
        draw.error(
            "VesselAPI key missing. Add `vesselapi_key` (and optionally a "
            "backup `vesselapi2_key`) to `.streamlit/secrets.toml`."
        )
        return

    rate_limited = st.session_state.get('vesselapi_rate_limited_keys') or set()
    if rate_limited:
        labels = ', '.join(sorted(rate_limited))
        draw.warning(
            f"⚠️ {labels} VesselAPI key(s) hit rate-limit this session — "
            f"requests are auto-falling-back to the next available key."
        )

    # Initialize session state
    st.session_state.setdefault('vesselapi_request_count', 0)
    st.session_state.setdefault('vesselapi_last_results', None)
    st.session_state.setdefault('vesselapi_last_fetched_at', None)

    # Prominent shared staleness banner (color-coded by age) — replaces the
    # 'Last fetched' metric tile to save vertical space and stay consistent
    # with the other Live Data pages.
    from utils import display_last_updated_badge
    display_last_updated_badge(
        draw,
        st.session_state.get('vesselapi_last_fetched_at'),
        label="Last updated",
    )

    # ── Header row: counter + Fetch buttons ──
    c1, c3, c4 = draw.columns([1.3, 1.1, 1.4])

    # Pull persistent count from Cloudflare KV; fall back to session counter.
    last_30_count = _count_vesselapi_calls_last_30_days()
    if last_30_count is not None:
        c1.metric("API requests (last 30 days)", last_30_count)
    else:
        c1.metric(
            "API requests this session",
            st.session_state['vesselapi_request_count'],
            help="Cloudflare KV not configured — falling back to session count.",
        )

    fetch_clicked = c3.button(
        "🛰 Fetch all positions",
        type='primary',
        help="Resolves MMSIs (cached 24h) then calls /vessel/{id}/position for "
             "each fleet boat — 1 API request per boat. Each call increments "
             "the counter on the left.",
    )

    fetch_one_clicked = c4.button(
        "🔎 Lookup Salish Sea Freedom",
        help="Search + single-vessel position lookup for Salish Sea Freedom only. "
             "Useful for testing without burning the full-fleet quota.",
    )

    # Second button row: region update + zoom controls
    rb1, rb2, rb3, rb4 = draw.columns([1.3, 0.5, 0.5, 1.7])
    fetch_van_clicked = rb1.button(
        "🌊 Update Vancouver",
        help="Refresh positions for Vancouver-area boats only. Cheaper than "
             "the full fleet fetch — leaves Victoria and Telegraph Cove rows "
             "as they were.",
    )
    zoom_in_clicked = rb2.button(
        "🔍+",
        help="Zoom map in.",
    )
    zoom_out_clicked = rb3.button(
        "🔎−",
        help="Zoom map out.",
    )

    # Map zoom state — adjusted by the buttons above (±2 per click)
    st.session_state.setdefault('whales2_map_zoom', 8.5)
    if zoom_in_clicked:
        st.session_state['whales2_map_zoom'] = min(
            st.session_state['whales2_map_zoom'] + 2, 16
        )
    if zoom_out_clicked:
        st.session_state['whales2_map_zoom'] = max(
            st.session_state['whales2_map_zoom'] - 2, 3
        )

    def _bust_count_cache():
        try:
            _count_vesselapi_calls_last_30_days.clear()
        except Exception:
            pass

    if fetch_clicked:
        with st.spinner("Querying VesselAPI for fleet positions…"):
            try:
                results, diagnostics = _do_fetch_all()
                st.session_state['vesselapi_last_results'] = results
                st.session_state['vesselapi_last_diagnostics'] = diagnostics
                st.session_state['vesselapi_last_fetched_at'] = datetime.now(
                    pytz.timezone('America/Vancouver'))
                _bust_count_cache()
            except Exception as e:
                draw.error(f"Fetch failed: {e}")

    if fetch_van_clicked:
        with st.spinner("Updating Vancouver-area boats…"):
            try:
                new_results, diag = _do_fetch_all(region='Vancouver')
                merged = _merge_results(
                    st.session_state.get('vesselapi_last_results'), new_results
                )
                st.session_state['vesselapi_last_results'] = merged
                st.session_state['vesselapi_last_diagnostics'] = diag
                st.session_state['vesselapi_last_fetched_at'] = datetime.now(
                    pytz.timezone('America/Vancouver'))
                _bust_count_cache()
            except Exception as e:
                draw.error(f"Update Vancouver failed: {e}")

    if fetch_one_clicked:
        # Find the Salish Sea Freedom entry from the fleet list
        target_boat = next(
            (b for b in WHALE_FLEET if b['name'].lower() == 'salish sea freedom'),
            None,
        )
        if target_boat is None:
            draw.error("Salish Sea Freedom not in fleet list.")
        else:
            with st.spinner(f"Looking up {target_boat['name']}…"):
                try:
                    one_result, one_diag = _do_fetch_one(target_boat)
                    # Merge: replace Salish Sea Freedom's row in stored results, or
                    # initialize results with just that boat if no prior fetch.
                    prior = st.session_state.get('vesselapi_last_results') or []
                    if prior:
                        prior = [
                            one_result if r.get('name') == target_boat['name'] else r
                            for r in prior
                        ]
                    else:
                        prior = [one_result]
                    st.session_state['vesselapi_last_results'] = prior
                    st.session_state['vesselapi_last_diagnostics'] = one_diag
                    st.session_state['vesselapi_last_fetched_at'] = datetime.now(
                        pytz.timezone('America/Vancouver'))
                    try:
                        _count_vesselapi_calls_last_30_days.clear()
                    except Exception:
                        pass
                except Exception as e:
                    draw.error(f"Lookup failed: {e}")

    results = st.session_state.get('vesselapi_last_results')
    diagnostics = st.session_state.get('vesselapi_last_diagnostics')

    if not results:
        draw.info(
            "Click **🛰 Fetch positions** to query VesselAPI for the fleet. "
            "Each click costs roughly 1 search + 1 position lookup per boat (10 boats → ~20 calls). "
            "Name searches are cached for 24 hours; positions are not cached."
        )
        return

    # ── Map ──
    matched = [
        r for r in results
        if r.get('position') and _pos_field(r['position'], 'latitude') is not None
    ]

    if matched:
        try:
            now_van = datetime.now(pytz.timezone('America/Vancouver'))
            fig = go.Figure()
            by_op = {}
            for r in matched:
                by_op.setdefault(r['operator'], []).append(r)

            for operator, items in by_op.items():
                color = items[0]['icon_color']
                fig.add_trace(go.Scattermapbox(
                    lat=[_pos_field(r['position'], 'latitude') for r in items],
                    lon=[_pos_field(r['position'], 'longitude') for r in items],
                    mode='markers+text',
                    marker=dict(size=14, color=color),
                    text=[r['name'] for r in items],
                    textposition='top center',
                    textfont=dict(size=11, color=color),
                    name=operator,
                    customdata=[
                        [
                            r['name'],
                            operator,
                            (
                                f"{_pos_field(r['position'], 'sog'):.1f}"
                                if _pos_field(r['position'], 'sog') is not None else '?'
                            ),
                            (
                                f"{_pos_field(r['position'], 'cog'):.0f}°"
                                if _pos_field(r['position'], 'cog') is not None else '?'
                            ),
                            _format_age(r.get('last_seen_iso'), now_van) or '–',
                        ]
                        for r in items
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]}<br>"
                        "Speed: %{customdata[2]} kts · Course: %{customdata[3]}<br>"
                        "Last seen: %{customdata[4]}<extra></extra>"
                    ),
                ))

            # Re-center on Vancouver-area subset when one is in focus, else
            # auto-fit to the points we have.
            van_pts = [r for r in matched
                       if (r.get('region') == 'Vancouver'
                           and _pos_field(r['position'], 'latitude') is not None)]
            anchor = van_pts[0] if van_pts else matched[0]
            center_lat = _pos_field(anchor['position'], 'latitude') or 49.20
            center_lon = _pos_field(anchor['position'], 'longitude') or -123.30

            fig.update_layout(
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=st.session_state.get('whales2_map_zoom', 8.5),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
                legend=dict(orientation='h', y=-0.05),
            )
            # Plotly's native scroll/pinch-to-zoom is enabled by default;
            # the explicit + / − buttons above are for users on touchscreens
            # (e.g. Nvidia Shield kiosk) where Plotly's UI is hidden.
            draw.plotly_chart(fig, width='stretch')
        except Exception as e:
            draw.warning(f"Map render failed: {e}")
    else:
        draw.info("No positions returned — boats may not be transmitting AIS right now.")

    # ── Fleet table ──
    try:
        draw.markdown("**Fleet status**")
        rows = []
        now_van = datetime.now(pytz.timezone('America/Vancouver'))
        for r in results:
            pos = r.get('position') or {}
            sog = _pos_field(pos, 'sog')
            cog = _pos_field(pos, 'cog')
            lat = _pos_field(pos, 'latitude')
            lon = _pos_field(pos, 'longitude')
            has_pos = lat is not None and lon is not None
            rows.append({
                'Boat': r['name'],
                'Operator': r['operator'],
                'MMSI': r.get('mmsi') or '—',
                'Status': '🟢 Live' if has_pos else '⚫ Silent',
                'Latitude':  f"{lat:.5f}" if lat is not None else '–',
                'Longitude': f"{lon:.5f}" if lon is not None else '–',
                'Speed (kts)': f"{sog:.1f}" if sog is not None else '–',
                'Course': f"{cog:.0f}°" if cog is not None else '–',
                'Last seen': _format_age(r.get('last_seen_iso'), now_van) or '–',
            })
        draw.dataframe(pd.DataFrame(rows))
    except Exception as e:
        draw.warning(f"Fleet table render failed: {e}")

    # ── Manual MMSI override editor ──
    # The search-by-name endpoint frequently picks the wrong vessel for short or
    # generic boat names. If MyShipTracking iOS shows the boat correctly, tap
    # the boat there → see the MMSI → enter it here. Saved overrides persist
    # for the session and bypass the search entirely on subsequent fetches.
    st.session_state.setdefault('vesselapi_mmsi_overrides', {})
    overrides = st.session_state['vesselapi_mmsi_overrides']

    with draw.expander("⚙️ Manual MMSI overrides (paste from MyShipTracking)", expanded=False):
        draw.caption(
            "Open MyShipTracking iOS → tap the boat → copy its MMSI → paste here. "
            "Overrides skip the buggy name-search entirely."
        )
        c_a, c_b, c_c = draw.columns([2, 1.4, 0.8])
        boat_choice = c_a.selectbox(
            "Boat",
            [b['name'] for b in WHALE_FLEET],
            index=[i for i, b in enumerate(WHALE_FLEET) if b['name'] == 'Salish Sea Freedom'][0],
            key='mmsi_override_boat',
        )
        mmsi_value = c_b.text_input(
            "MMSI",
            value=str(overrides.get(boat_choice, '') or ''),
            key='mmsi_override_value',
            placeholder='e.g. 316023456',
        )
        if c_c.button("Save", key='mmsi_override_save'):
            mmsi_clean = mmsi_value.strip()
            if mmsi_clean:
                try:
                    overrides[boat_choice] = int(mmsi_clean)
                    draw.success(f"Saved {boat_choice} → MMSI {overrides[boat_choice]}")
                except ValueError:
                    draw.error("MMSI must be a number.")
            else:
                # Empty input = clear the override
                overrides.pop(boat_choice, None)
                draw.success(f"Cleared override for {boat_choice}")

        if overrides:
            draw.markdown("**Active overrides:**")
            draw.dataframe(pd.DataFrame(
                [{'Boat': k, 'MMSI': v} for k, v in overrides.items()]
            ))

    # ── Diagnostics: show raw API responses so we can spot wrong MMSIs etc.
    if diagnostics:
        with draw.expander("🔍 Raw API responses (diagnostics)"):
            draw.markdown(
                "**Per-boat search results.** If `mmsi` here does NOT belong to "
                "the actual whale-watching boat, the search picked the wrong "
                "vessel and the position lookup will be useless. Tell me the "
                "correct MMSI and I'll switch to a hard-coded mapping."
            )
            search_rows = []
            for boat_name, payload in (diagnostics.get('searches') or {}).items():
                vessels = (payload or {}).get('raw', {}).get('vessels') or []
                if not vessels:
                    search_rows.append({
                        'Searched': boat_name,
                        'Top hit': '—',
                        'MMSI': '—',
                        'Country': '—',
                        'Type': '—',
                    })
                else:
                    top = vessels[0]
                    search_rows.append({
                        'Searched': boat_name,
                        'Top hit': top.get('name') or top.get('vessel_name') or '?',
                        'MMSI': top.get('mmsi') or '—',
                        'Country': top.get('country') or '—',
                        'Type': top.get('vessel_type') or top.get('type') or '—',
                    })
            try:
                draw.dataframe(pd.DataFrame(search_rows))
            except Exception:
                pass

            pos_diag = diagnostics.get('positions') or {}
            draw.markdown("---")
            draw.markdown("**Batch /vessels/positions response**")
            if 'error' in pos_diag:
                draw.error(f"HTTP error: {pos_diag.get('error')}")
            draw.code(json.dumps(pos_diag, indent=2, default=str)[:4000], language='json')
