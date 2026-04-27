"""Whale-watching boat tracker — VesselAPI Ship Tracking variant.

Uses VesselAPI's Ship Tracking REST API
(https://vesselapi.com/ship-tracking-api) to look up fleet vessels by name
and then batch-fetch their latest reported positions in one call. Position
fetches are gated behind a manual 'Fetch positions' button to limit billable
API usage; the page also tracks the running call count.
"""

import json
from datetime import datetime

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

# Same curated fleet as the AISStream tracker — kept in this module so the two
# pages stay independent and editable.
WHALE_FLEET = [
    {'name': 'Aurora I',         'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    {'name': 'Aurora II',        'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    {'name': 'Eagle Eyes',       'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    {'name': 'Jing Yu',          'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    {'name': 'Explorathor II',   'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    {'name': 'Express',          'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    {'name': 'Strider',          'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    {'name': 'Lightship',        'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    {'name': 'Salish Sea Dream',   'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'Salish Sea Freedom', 'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'Salish Sea Eclipse', 'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'Ocean Magic II',     'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'Ocean Magic',        'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
]


def _api_key():
    """Fetch the VesselAPI key from Streamlit secrets. Try common naming conventions."""
    for key_name in ('vesselapi_key', 'vessel_api_key', 'vesselapi-com_key'):
        try:
            return st.secrets[key_name]
        except (KeyError, FileNotFoundError):
            continue
    return None


def _bump_count(n=1):
    """Increment the running API request counter in session state."""
    st.session_state.setdefault('vesselapi_request_count', 0)
    st.session_state['vesselapi_request_count'] += n


@st.cache_data(ttl=86400, show_spinner=False)
def _search_vessel_by_name(api_key, name):
    """Search VesselAPI for a vessel name.
    Returns (mmsi, raw_record, full_search_payload) — third value used for
    diagnostics so the user can inspect what the API actually returned."""
    url = f"{VESSELAPI_BASE}/search/vessels"
    params = {'filter.name': name, 'pagination.limit': 5}
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        vessels = data.get('vessels') or []
        if not vessels:
            return None, None, {'request_url': r.url, 'raw': data}
        target = name.strip().lower()
        best = None
        for v in vessels:
            if (v.get('name') or '').strip().lower() == target:
                best = v
                break
        best = best or vessels[0]
        return best.get('mmsi'), best, {'request_url': r.url, 'raw': data}
    except Exception as e:
        return None, None, {'error': str(e), 'request_url': url, 'params': params}


def _get_vessel_positions_batch(api_key, mmsis):
    """Fetch latest positions for multiple MMSIs in a single Ship Tracking API call.
    Uses /vessels/positions — one billable request regardless of fleet size.
    Returns (positions_by_mmsi_dict, raw_response_or_error_dict)."""
    if not mmsis:
        return {}, {'note': 'no MMSIs to query'}
    url = f"{VESSELAPI_BASE}/vessels/positions"
    # Default time.from is "past 2 hours" which is too narrow for whale-watching
    # boats that may be docked overnight. Look back 7 days so we always see the
    # latest reported position.
    look_back = (datetime.utcnow() - pd.Timedelta(days=7))
    params = {
        'filter.ids': ','.join(str(m) for m in mmsis),
        'filter.idType': 'mmsi',
        'pagination.limit': 50,
        'time.from': look_back.strftime('%Y-%m-%dT%H:%M:%SZ'),
    }
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        items = data.get('vessels') or data.get('positions') or data.get('data') or []
        out = {}
        for v in items:
            mmsi = _pos_field(v, 'mmsi', 'MMSI')
            if mmsi is not None:
                out[int(mmsi)] = v
        return out, {'request_url': r.url, 'raw': data}
    except Exception as e:
        return {}, {'error': str(e), 'request_url': url, 'params': params}


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


def _do_fetch_all(api_key):
    """Resolve fleet MMSIs (cached search-by-name calls) then fetch all
    positions in a SINGLE batch /vessels/positions call.
    Returns (results_list, diagnostics_dict)."""
    if 'vesselapi_searched_names' not in st.session_state:
        st.session_state['vesselapi_searched_names'] = set()
    seen_names = st.session_state['vesselapi_searched_names']

    diagnostics = {'searches': {}, 'positions': None}

    boat_to_mmsi = {}
    boat_to_meta = {}
    for boat in WHALE_FLEET:
        if boat['name'] not in seen_names:
            _bump_count(1)
            seen_names.add(boat['name'])

        mmsi, meta, raw = _search_vessel_by_name(api_key, boat['name'])
        diagnostics['searches'][boat['name']] = raw
        if mmsi is not None:
            boat_to_mmsi[boat['name']] = int(mmsi)
            boat_to_meta[boat['name']] = meta

    positions_by_mmsi = {}
    if boat_to_mmsi:
        _bump_count(1)
        positions_by_mmsi, pos_diag = _get_vessel_positions_batch(
            api_key, list(boat_to_mmsi.values())
        )
        diagnostics['positions'] = pos_diag

    results = []
    for boat in WHALE_FLEET:
        mmsi = boat_to_mmsi.get(boat['name'])
        pos = positions_by_mmsi.get(mmsi) if mmsi is not None else None
        last_seen_iso = _pos_field(pos, 'timestamp', 'lastSeen', 'last_seen', 'processed_timestamp')
        results.append({
            **boat,
            'mmsi': mmsi,
            'position': pos,
            'meta': boat_to_meta.get(boat['name']),
            'last_seen_iso': last_seen_iso,
        })
    return results, diagnostics


def display_whales2_page(container=None):
    """Render the VesselAPI-backed whale tracker."""
    draw = container or st

    draw.subheader("🐋 Whale boats 2 — VesselAPI Ship Tracking")
    draw.markdown(
        f"**Source:** [{SOURCE_URL}]({SOURCE_URL}) — Ship Tracking REST lookups by name + MMSI."
    )

    api_key = _api_key()
    if not api_key:
        draw.error(
            "VesselAPI key missing. Add `vesselapi_key` to `.streamlit/secrets.toml`."
        )
        return

    # Initialize session state
    st.session_state.setdefault('vesselapi_request_count', 0)
    st.session_state.setdefault('vesselapi_last_results', None)
    st.session_state.setdefault('vesselapi_last_fetched_at', None)

    # ── Header row: counter + Fetch button ──
    c1, c2, c3 = draw.columns([1, 1, 1])
    c1.metric("API requests this session", st.session_state['vesselapi_request_count'])
    if st.session_state['vesselapi_last_fetched_at']:
        c2.metric(
            "Last fetched",
            st.session_state['vesselapi_last_fetched_at'].strftime('%I:%M:%S %p'),
        )
    else:
        c2.metric("Last fetched", "Never")

    fetch_clicked = c3.button(
        "🛰 Fetch positions",
        type='primary',
        help="Triggers a name-search (cached 24h) + position lookup per fleet vessel. "
             "Each call increments the counter on the left.",
    )

    if fetch_clicked:
        with st.spinner("Querying VesselAPI for fleet positions…"):
            try:
                results, diagnostics = _do_fetch_all(api_key)
                st.session_state['vesselapi_last_results'] = results
                st.session_state['vesselapi_last_diagnostics'] = diagnostics
                st.session_state['vesselapi_last_fetched_at'] = datetime.now(
                    pytz.timezone('America/Vancouver'))
            except Exception as e:
                draw.error(f"Fetch failed: {e}")

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

            fig.update_layout(
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=49.20, lon=-123.30),
                    zoom=8.5,
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
                legend=dict(orientation='h', y=-0.05),
            )
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
            has_pos = _pos_field(pos, 'latitude') is not None
            rows.append({
                'Boat': r['name'],
                'Operator': r['operator'],
                'MMSI': r.get('mmsi') or '—',
                'Status': '🟢 Live' if has_pos else '⚫ Silent',
                'Speed (kts)': f"{sog:.1f}" if sog is not None else '–',
                'Course': f"{cog:.0f}°" if cog is not None else '–',
                'Last seen': _format_age(r.get('last_seen_iso'), now_van) or '–',
            })
        draw.dataframe(pd.DataFrame(rows))
    except Exception as e:
        draw.warning(f"Fleet table render failed: {e}")

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
