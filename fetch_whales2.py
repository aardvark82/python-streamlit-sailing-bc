"""Whale-watching boat tracker — VesselAPI Ship Tracking variant.

Uses VesselAPI's Ship Tracking REST API
(https://vesselapi.com/ship-tracking-api) to look up fleet vessels by name
and then batch-fetch their latest reported positions in one call. Position
fetches are gated behind a manual 'Fetch positions' button to limit billable
API usage; the page also tracks the running call count.
"""

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


def _get_vessel_positions_batch(api_key, mmsis):
    """Fetch latest positions for multiple MMSIs in a single Ship Tracking API call.
    Uses /vessels/positions — one billable request regardless of fleet size.
    Returns dict {mmsi: position_record} for vessels that have a position."""
    if not mmsis:
        return {}
    url = f"{VESSELAPI_BASE}/vessels/positions"
    params = {
        'filter.ids': ','.join(str(m) for m in mmsis),
        'filter.idType': 'mmsi',
        'pagination.limit': 50,
    }
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=20)
        r.raise_for_status()
        data = r.json() or {}
        # Response array key may vary across API versions/products
        items = data.get('vessels') or data.get('positions') or data.get('data') or []
        out = {}
        for v in items:
            mmsi = _pos_field(v, 'mmsi', 'MMSI')
            if mmsi is not None:
                out[int(mmsi)] = v
        return out
    except Exception as e:
        print(f"VesselAPI Ship Tracking batch positions failed: {e}")
        return {}


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
    positions in a SINGLE batch /vessels/positions call. Returns list of
    dicts with the latest reported position + last-seen timestamp."""
    # Phase 1 — resolve MMSIs (cached in @st.cache_data for 24h)
    if 'vesselapi_searched_names' not in st.session_state:
        st.session_state['vesselapi_searched_names'] = set()
    seen_names = st.session_state['vesselapi_searched_names']

    boat_to_mmsi = {}     # boat name -> mmsi
    boat_to_meta = {}     # boat name -> static record from search
    for boat in WHALE_FLEET:
        if boat['name'] not in seen_names:
            _bump_count(1)
            seen_names.add(boat['name'])

        mmsi, meta = _search_vessel_by_name(api_key, boat['name'])
        if mmsi is not None:
            boat_to_mmsi[boat['name']] = int(mmsi)
            boat_to_meta[boat['name']] = meta

    # Phase 2 — single batch call for all known MMSIs
    positions_by_mmsi = {}
    if boat_to_mmsi:
        _bump_count(1)
        positions_by_mmsi = _get_vessel_positions_batch(
            api_key, list(boat_to_mmsi.values())
        )

    # Phase 3 — assemble result list, attaching latest position + last_seen
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
    return results


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
