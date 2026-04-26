"""Whale-watching boat tracker — VesselAPI variant.

Uses vesselapi.com REST API (https://vesselapi.com/docs/vessels) to look up
each fleet vessel by name (returns MMSI + metadata), then queries position
by MMSI. Position fetches are gated behind a manual "Fetch positions" button
to limit billable API usage; the page also tracks the running call count.
"""

from datetime import datetime

import requests
import streamlit as st
import pandas as pd
import pytz
import plotly.graph_objects as go


VESSELAPI_BASE = "https://api.vesselapi.com/v1"
SOURCE_URL = "https://vesselapi.com/docs/vessels"

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
    """Search VesselAPI for a vessel name. Returns (mmsi, raw_record) or (None, None).
    Cached for 24h so repeated runs don't re-burn the search quota."""
    url = f"{VESSELAPI_BASE}/search/vessels"
    params = {'filter.name': name, 'pagination.limit': 5}
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        vessels = data.get('vessels') or []
        if not vessels:
            return None, None
        # Pick the closest exact match if available
        target = name.strip().lower()
        best = None
        for v in vessels:
            if (v.get('name') or '').strip().lower() == target:
                best = v
                break
        best = best or vessels[0]
        return best.get('mmsi'), best
    except Exception as e:
        print(f"VesselAPI search '{name}' failed: {e}")
        return None, None


def _get_vessel_position(api_key, mmsi):
    """Fetch the latest position for an MMSI. NOT cached — counted as billable."""
    url = f"{VESSELAPI_BASE}/vessel/{mmsi}/position"
    params = {'filter.idType': 'mmsi'}
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        return (r.json() or {}).get('vessel')
    except Exception as e:
        print(f"VesselAPI position {mmsi} failed: {e}")
        return None


def _format_age(iso_ts, now_van):
    if not iso_ts:
        return ''
    try:
        # Accept '2026-01-15T12:30:00Z' or '+00:00'
        s = str(iso_ts).replace('Z', '+00:00')
        dt = datetime.fromisoformat(s)
        delta = now_van - dt.astimezone(now_van.tzinfo)
        secs = int(delta.total_seconds())
        if secs < 60:  return f'{secs}s ago'
        if secs < 3600:  return f'{secs // 60}m ago'
        if secs < 86400: return f'{secs // 3600}h ago'
        return f'{secs // 86400}d ago'
    except Exception:
        return ''


def _do_fetch_all(api_key):
    """Run a full fleet search + position fetch. Returns list of dicts and
    increments the request counter in session state for each network call."""
    results = []

    # Resolve MMSIs (cached → only first run truly hits the API)
    cache_was_warm_for = set()
    if 'vesselapi_searched_names' not in st.session_state:
        st.session_state['vesselapi_searched_names'] = set()
    cache_was_warm_for = st.session_state['vesselapi_searched_names']

    for boat in WHALE_FLEET:
        # Only count the search call when this name hasn't been resolved this session
        if boat['name'] not in cache_was_warm_for:
            _bump_count(1)
            cache_was_warm_for.add(boat['name'])

        mmsi, meta = _search_vessel_by_name(api_key, boat['name'])
        if mmsi is None:
            results.append({**boat, 'mmsi': None, 'position': None})
            continue

        # Position is always counted — gated behind the button so user controls cost
        _bump_count(1)
        pos = _get_vessel_position(api_key, mmsi)
        results.append({**boat, 'mmsi': mmsi, 'position': pos, 'meta': meta})

    return results


def display_whales2_page(container=None):
    """Render the VesselAPI-backed whale tracker."""
    draw = container or st

    draw.subheader("🐋 Whale boats 2 — VesselAPI")
    draw.markdown(
        f"**Source:** [{SOURCE_URL}]({SOURCE_URL}) — REST lookups by vessel name + MMSI."
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
                results = _do_fetch_all(api_key)
                st.session_state['vesselapi_last_results'] = results
                st.session_state['vesselapi_last_fetched_at'] = datetime.now(
                    pytz.timezone('America/Vancouver'))
            except Exception as e:
                draw.error(f"Fetch failed: {e}")
                results = None

    results = st.session_state.get('vesselapi_last_results')

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
        if r.get('position') and r['position'].get('latitude') is not None
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
                    lat=[r['position']['latitude'] for r in items],
                    lon=[r['position']['longitude'] for r in items],
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
                            f"{r['position'].get('sog', 0):.1f}" if r['position'].get('sog') is not None else '?',
                            f"{r['position'].get('cog', 0):.0f}°" if r['position'].get('cog') is not None else '?',
                            _format_age(r['position'].get('timestamp'), now_van),
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
            rows.append({
                'Boat': r['name'],
                'Operator': r['operator'],
                'MMSI': r.get('mmsi') or '—',
                'Status': '🟢 Live' if pos.get('latitude') is not None else '⚫ Silent',
                'Speed (kts)': f"{pos.get('sog', 0):.1f}" if pos.get('sog') is not None else '–',
                'Course': f"{pos.get('cog', 0):.0f}°" if pos.get('cog') is not None else '–',
                'Last seen': _format_age(pos.get('timestamp'), now_van),
            })
        draw.dataframe(pd.DataFrame(rows))
    except Exception as e:
        draw.warning(f"Fleet table render failed: {e}")
