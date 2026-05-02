"""Whale-watching boat tracker.

Subscribes briefly to AISStream.io's WebSocket API on each cache miss, collects
PositionReport + ShipStaticData messages within a Vancouver-area bounding box,
matches vessel names against our curated whale-watching fleet list, and renders
the matches on a map.
"""

import asyncio
import json
import re
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import pytz
import plotly.graph_objects as go


# Curated whale-watching fleet — names matched case-insensitively against the
# `ShipName` field of incoming AIS ShipStaticData messages. Smaller zodiacs
# (Aurora I/II, Strider, Lightship) often don't transmit AIS so they may not
# appear; the larger covered/semi-covered vessels usually do.
WHALE_FLEET = [
    # Wild Whales Vancouver
    {'name': 'AURORA I',         'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    {'name': 'AURORA II',        'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    {'name': 'EAGLE EYES',       'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    {'name': 'JING YU',          'operator': 'Wild Whales Vancouver',  'icon_color': '#1f77b4'},
    # Vancouver Whale Watch
    {'name': 'EXPLORATHOR II',   'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    {'name': 'EXPRESS',          'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    {'name': 'STRIDER',          'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    {'name': 'LIGHTSHIP',        'operator': 'Vancouver Whale Watch',  'icon_color': '#2ca02c'},
    # Prince of Whales (Vancouver, Victoria, Telegraph Cove)
    {'name': 'SALISH SEA DREAM',   'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'SALISH SEA FREEDOM', 'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'SALISH SEA ECLIPSE', 'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'OCEAN MAGIC II',     'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
    {'name': 'OCEAN MAGIC',        'operator': 'Prince of Whales', 'icon_color': '#ff7f0e'},
]

# Strait of Georgia + Howe Sound + Burrard Inlet bounding box
# AISStream uses [[lat_sw, lon_sw], [lat_ne, lon_ne]]
WHALE_BBOX = [[48.6, -124.4], [49.9, -122.7]]

# How long to listen on each cache miss (seconds). ShipStaticData (which carries
# the vessel name) is broadcast roughly every 6 minutes, so we need a longer
# window to have a meaningful chance of catching it on first contact.
LISTEN_SECONDS = 20


def _normalize_name(name):
    """Strip non-alphanumerics and uppercase for fuzzy matching."""
    if not name:
        return ''
    return re.sub(r'[^A-Z0-9]', '', str(name).upper())


_FLEET_LOOKUP = {_normalize_name(b['name']): b for b in WHALE_FLEET}


async def _collect_ais(api_key, listen_seconds):
    """Open AISStream WebSocket, subscribe to bounding box, listen briefly,
    accumulate the latest position + name per MMSI, return list of dicts."""
    import websockets

    url = "wss://stream.aisstream.io/v0/stream"
    by_mmsi = {}  # mmsi -> {name, lat, lon, sog, cog, time}

    subscribe = {
        "APIKey": api_key,
        "BoundingBoxes": [WHALE_BBOX],
        "FilterMessageTypes": ["PositionReport", "ShipStaticData"],
    }

    try:
        async with websockets.connect(url, open_timeout=8, close_timeout=2) as ws:
            await ws.send(json.dumps(subscribe))

            deadline = asyncio.get_event_loop().time() + listen_seconds
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    msg = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                try:
                    data = json.loads(msg)
                except (json.JSONDecodeError, TypeError):
                    continue

                metadata = data.get('MetaData') or {}
                mmsi = metadata.get('MMSI') or metadata.get('mmsi')
                if mmsi is None:
                    continue
                mmsi = int(mmsi)

                rec = by_mmsi.setdefault(mmsi, {
                    'mmsi': mmsi, 'name': None,
                    'lat': None, 'lon': None,
                    'sog': None, 'cog': None,
                    'time': None,
                })

                msg_type = data.get('MessageType')
                payload = (data.get('Message') or {}).get(msg_type) or {}

                # PositionReport: lat/lon/speed/course
                if msg_type == 'PositionReport':
                    lat = payload.get('Latitude')
                    lon = payload.get('Longitude')
                    if lat is not None and lon is not None:
                        rec['lat'] = float(lat)
                        rec['lon'] = float(lon)
                    sog = payload.get('Sog')
                    cog = payload.get('Cog')
                    if sog is not None:
                        rec['sog'] = float(sog)
                    if cog is not None:
                        rec['cog'] = float(cog)
                    rec['time'] = metadata.get('time_utc') or rec['time']

                # ShipStaticData: vessel name (and other identifiers)
                if msg_type == 'ShipStaticData':
                    ship_name = payload.get('Name') or metadata.get('ShipName')
                    if ship_name:
                        rec['name'] = ship_name.strip()

                # Some AISStream variants put ShipName directly in MetaData
                if not rec['name']:
                    rec['name'] = (metadata.get('ShipName') or '').strip() or rec['name']

    except Exception as e:
        print(f"AIS collection error: {e}")

    return list(by_mmsi.values())


def _match_fleet_record(r, fleet_lookup):
    """Return the fleet entry for a single record (by current name) or None."""
    if not r.get('name'):
        return None
    norm = _normalize_name(r['name'])
    fleet = fleet_lookup.get(norm)
    if fleet is not None:
        return fleet
    # Substring fallback (handles 'MV X' or 'X II' style minor variations)
    for key, val in fleet_lookup.items():
        if key and (key in norm or norm in key):
            return val
    return None


def fetch_whale_positions():
    """Listen briefly to AISStream, merge with persistent MMSI→name knowledge
    in session_state, and return:
      (matched_fleet, all_records_seen_in_window, learned_names_total, fetched_at)

    NOT cached — caching turns out to mask the persistent-learning behaviour
    we need across reruns. Repeated reruns within a few minutes will quickly
    accumulate enough name knowledge to match boats reliably.
    """
    try:
        api_key = st.secrets["aisstream-io_key"]
    except KeyError:
        return None, [], 0, None

    # Persistent MMSI→name knowledge: once we hear a name for an MMSI we
    # remember it for the rest of the session. New PositionReports for that
    # MMSI then match the fleet even though no name arrives this window.
    learned = st.session_state.setdefault('ais_mmsi_to_name', {})

    records = asyncio.run(_collect_ais(api_key, LISTEN_SECONDS))

    # Update the persistent name map with anything new we just heard
    for r in records:
        if r.get('name') and r.get('mmsi'):
            learned[int(r['mmsi'])] = r['name']

    # Build the final matched-fleet list using both fresh records and the
    # learned name map (so a PositionReport with no name in this window can
    # still be tied to our fleet if we've heard the name in a prior window).
    matched = []
    for r in records:
        mmsi = r.get('mmsi')
        # Inject persistent name if missing on this record
        if mmsi and not r.get('name') and mmsi in learned:
            r['name'] = learned[mmsi]
        fleet = _match_fleet_record(r, _FLEET_LOOKUP)
        if fleet is None:
            continue
        matched.append({
            **r,
            'fleet_name': fleet['name'],
            'operator': fleet['operator'],
            'icon_color': fleet['icon_color'],
        })

    return matched, records, len(learned), datetime.now(pytz.timezone('America/Vancouver'))


def _format_age(time_utc_str, now_van):
    """Convert AIS UTC timestamp string to a 'Xm ago' style age."""
    if not time_utc_str:
        return ''
    try:
        # AISStream times look like '2024-09-15 18:42:11.123456789 +0000 UTC'
        s = re.sub(r'\.\d+', '', str(time_utc_str))
        s = s.replace(' UTC', '').replace(' +0000', '+0000').strip()
        dt = datetime.strptime(s, '%Y-%m-%d %H:%M:%S%z')
        delta = now_van - dt.astimezone(now_van.tzinfo)
        secs = int(delta.total_seconds())
        if secs < 60:
            return f'{secs}s ago'
        if secs < 3600:
            return f'{secs // 60}m ago'
        if secs < 86400:
            return f'{secs // 3600}h ago'
        return f'{secs // 86400}d ago'
    except Exception:
        return ''


def display_whales_page(container=None):
    """Render the Whales tracker page: map + table."""
    draw = container or st
    draw.subheader("🐋 Whale watching boats — live AIS")
    draw.caption(
        "Live positions from AISStream.io for the Vancouver-area whale-watching "
        "fleet. Smaller zodiacs may not transmit AIS and won't appear."
    )

    with st.spinner(f"Listening for AIS broadcasts ({LISTEN_SECONDS}s)…"):
        matched, all_records, learned_names_total, fetched_at = fetch_whale_positions()

    if matched is None:
        draw.error("AISStream.io key missing — set `aisstream-io_key` in Streamlit secrets.")
        return

    if fetched_at is not None:
        # Shared prominent staleness banner across pages
        from utils import display_last_updated_badge
        display_last_updated_badge(draw, fetched_at, label="Last updated")
        draw.caption(
            f"{len(all_records)} vessels heard this window · "
            f"{learned_names_total} known MMSIs (cumulative this session) · "
            f"{len(matched)} matched whale-watching fleet"
        )

    # ── Diagnostic expander: show every named vessel observed in the bbox so
    # the user can spot whether a fleet boat is transmitting under a different
    # exact name. (Triggered once we've heard at least one name.)
    named_records = [
        r for r in all_records
        if r.get('name') and r.get('lat') is not None
    ]
    with draw.expander(
        f"🔍 Vessels heard in bounding box ({len(named_records)} with name + position)"
    ):
        if named_records:
            now_van = datetime.now(pytz.timezone('America/Vancouver'))
            df_all = pd.DataFrame([
                {
                    'Name': r['name'],
                    'MMSI': r.get('mmsi'),
                    'Lat': round(r['lat'], 4) if r.get('lat') is not None else None,
                    'Lon': round(r['lon'], 4) if r.get('lon') is not None else None,
                    'Speed (kts)': round(r['sog'], 1) if r.get('sog') is not None else None,
                    'Last seen': _format_age(r.get('time'), now_van),
                }
                for r in named_records
            ])
            try:
                draw.dataframe(df_all)
            except Exception as e:
                draw.warning(f"All-vessels table failed: {e}")
            draw.caption(
                "If you spot a fleet boat under a slightly different name "
                "(e.g. 'MV SALISH SEA DREAM'), tell me and I'll add it to "
                "the matcher."
            )
        else:
            draw.caption(
                "No named vessels caught in this window. ShipStaticData "
                "broadcasts are infrequent (~every 6 min); reload the page "
                "in a minute or two — names will accumulate over multiple "
                "windows."
            )

    if not matched:
        draw.info(
            "No whale-watching vessels broadcasting right now. "
            "They may be in port, off duty, or out of AIS range. "
            "The page auto-refreshes every 5 minutes."
        )

    # ── Map zoom controls (±2 per click) ──
    st.session_state.setdefault('whales1_map_zoom', 8.5)
    z1, z2, _ = draw.columns([0.4, 0.4, 4])
    if z1.button("🔍+", key='whales1_zoom_in', help="Zoom map in"):
        st.session_state['whales1_map_zoom'] = min(
            st.session_state['whales1_map_zoom'] + 2, 16)
    if z2.button("🔎−", key='whales1_zoom_out', help="Zoom map out"):
        st.session_state['whales1_map_zoom'] = max(
            st.session_state['whales1_map_zoom'] - 2, 3)

    # ── Map (only when we have positions) ──
    if matched:
        try:
            now_van = datetime.now(pytz.timezone('America/Vancouver'))
            fig = go.Figure()
            by_op = {}
            for m in matched:
                if m.get('lat') is None or m.get('lon') is None:
                    continue
                by_op.setdefault(m['operator'], []).append(m)

            for operator, items in by_op.items():
                color = items[0]['icon_color']
                fig.add_trace(go.Scattermapbox(
                    lat=[m['lat'] for m in items],
                    lon=[m['lon'] for m in items],
                    mode='markers+text',
                    marker=dict(size=14, color=color),
                    text=[m['fleet_name'] for m in items],
                    textposition='top center',
                    textfont=dict(size=11, color=color),
                    name=operator,
                    customdata=[
                        [
                            m['fleet_name'],
                            operator,
                            f"{m['sog']:.1f}" if m.get('sog') is not None else '?',
                            f"{m['cog']:.0f}°" if m.get('cog') is not None else '?',
                            _format_age(m.get('time'), now_van),
                        ]
                        for m in items
                    ],
                    hovertemplate=(
                        "<b>%{customdata[0]}</b><br>"
                        "%{customdata[1]}<br>"
                        "Speed: %{customdata[2]} kts · Course: %{customdata[3]}<br>"
                        "Last seen: %{customdata[4]}<extra></extra>"
                    ),
                ))

            # Center on the first matched fleet boat with a fix so zoom keeps
            # focus on the boats. Falls back to a Vancouver-area default if no
            # boat has a position.
            anchor = next(
                (m for m in matched
                 if m.get('lat') is not None and m.get('lon') is not None),
                None,
            )
            center_lat = anchor['lat'] if anchor else 49.20
            center_lon = anchor['lon'] if anchor else -123.30

            fig.update_layout(
                mapbox=dict(
                    style='open-street-map',
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=st.session_state.get('whales1_map_zoom', 8.5),
                ),
                margin=dict(l=0, r=0, t=0, b=0),
                height=520,
                legend=dict(orientation='h', y=-0.05),
            )
            draw.plotly_chart(fig, width='stretch')
        except Exception as e:
            draw.warning(f"Map render failed: {e}")

    # ── Fleet table (always shown so user knows which boats we're tracking) ──
    try:
        draw.markdown("**Fleet status**")
        matched_by_name = {m['fleet_name']: m for m in (matched or [])}
        rows = []
        now_van = datetime.now(pytz.timezone('America/Vancouver'))
        for boat in WHALE_FLEET:
            m = matched_by_name.get(boat['name'])
            if m:
                rows.append({
                    'Boat': boat['name'],
                    'Operator': boat['operator'],
                    'Status': '🟢 Live',
                    'Speed (kts)': f"{m['sog']:.1f}" if m.get('sog') is not None else '–',
                    'Course': f"{m['cog']:.0f}°" if m.get('cog') is not None else '–',
                    'Last seen': _format_age(m.get('time'), now_van),
                })
            else:
                rows.append({
                    'Boat': boat['name'],
                    'Operator': boat['operator'],
                    'Status': '⚫ Silent',
                    'Speed (kts)': '–', 'Course': '–', 'Last seen': '–',
                })
        # Omit width entirely — st.dataframe in this Streamlit version had a
        # str→int error on width='stretch'.
        draw.dataframe(pd.DataFrame(rows))
    except Exception as e:
        draw.warning(f"Fleet table render failed: {e}")
