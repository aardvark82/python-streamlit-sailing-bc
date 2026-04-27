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

# Curated fleet. `mmsi` is optional — when present we skip the unreliable
# search-by-name and call /vessel/{mmsi}/position directly. MMSIs collected
# from cross-checking with MyShipTracking iOS.
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
    {'name': 'Salish Sea Freedom', 'operator': 'Prince of Whales', 'icon_color': '#ff7f0e',
     'mmsi': 316042213},
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


def _get_vessel_position(api_key, mmsi):
    """Fetch latest position for ONE MMSI via /vessel/{id}/position.
    Returns (position_record_or_None, diagnostic_payload).
    Each call counts as 1 billable API request."""
    url = f"{VESSELAPI_BASE}/vessel/{mmsi}/position"
    params = {'filter.idType': 'mmsi'}
    headers = {'Authorization': f'Bearer {api_key}'}
    try:
        r = requests.get(url, params=params, headers=headers, timeout=15)
        r.raise_for_status()
        data = r.json() or {}
        # The Ship Tracking endpoint wraps the position fields under
        # 'vesselPosition' (camelCase). Older docs sometimes show 'vessel'.
        # Fall back to the top-level dict if neither wrapper is present.
        pos = (
            data.get('vesselPosition')
            or data.get('vessel')
            or data.get('position')
            or data
        )
        return pos, {'request_url': r.url, 'raw': data}
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


def _resolve_mmsi(api_key, boat_name, diagnostics):
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

    # 3. Cached search-by-name fallback
    if 'vesselapi_searched_names' not in st.session_state:
        st.session_state['vesselapi_searched_names'] = set()
    seen = st.session_state['vesselapi_searched_names']
    if boat_name not in seen:
        _bump_count(1)
        seen.add(boat_name)
    mmsi, meta, raw = _search_vessel_by_name(api_key, boat_name)
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


def _do_fetch_one(api_key, boat):
    """Resolve MMSI then fetch the latest position for ONE fleet boat using
    /vessel/{id}/position. Returns (result_dict, diagnostics_dict)."""
    diagnostics = {'searches': {}, 'positions': {}}
    mmsi, meta = _resolve_mmsi(api_key, boat['name'], diagnostics)
    pos = None
    if mmsi is not None:
        _bump_count(1)
        pos, pos_diag = _get_vessel_position(api_key, mmsi)
        diagnostics['positions'][boat['name']] = pos_diag
    return _build_result(boat, mmsi, meta, pos), diagnostics


def _do_fetch_all(api_key):
    """Resolve fleet MMSIs (cached) then fetch each position via the
    single-vessel /vessel/{id}/position endpoint — one call per boat.
    Returns (results_list, diagnostics_dict)."""
    diagnostics = {'searches': {}, 'positions': {}}
    results = []
    for boat in WHALE_FLEET:
        mmsi, meta = _resolve_mmsi(api_key, boat['name'], diagnostics)
        pos = None
        if mmsi is not None:
            _bump_count(1)
            pos, pos_diag = _get_vessel_position(api_key, mmsi)
            diagnostics['positions'][boat['name']] = pos_diag
        results.append(_build_result(boat, mmsi, meta, pos))
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
    c1, c2, c3, c4 = draw.columns([1, 1, 1.1, 1.4])
    c1.metric("API requests this session", st.session_state['vesselapi_request_count'])
    if st.session_state['vesselapi_last_fetched_at']:
        c2.metric(
            "Last fetched",
            st.session_state['vesselapi_last_fetched_at'].strftime('%I:%M:%S %p'),
        )
    else:
        c2.metric("Last fetched", "Never")

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
                    one_result, one_diag = _do_fetch_one(api_key, target_boat)
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
