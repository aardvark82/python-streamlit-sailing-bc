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

# How long to listen on each cache miss (seconds). Trade-off: longer catches
# more vessels but blocks the UI longer.
LISTEN_SECONDS = 8


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


def _match_fleet(records):
    """Filter raw vessel records down to those whose name matches our fleet."""
    matched = []
    for r in records:
        if not r.get('name'):
            continue
        norm = _normalize_name(r['name'])
        # Try exact match first, then prefix match (some boats prefix MV/MS/etc)
        fleet = _FLEET_LOOKUP.get(norm)
        if fleet is None:
            for key, val in _FLEET_LOOKUP.items():
                if key and key in norm:
                    fleet = val
                    break
        if fleet is None:
            continue
        matched.append({
            **r,
            'fleet_name': fleet['name'],
            'operator': fleet['operator'],
            'icon_color': fleet['icon_color'],
        })
    return matched


@st.cache_data(ttl=120, show_spinner=False)
def fetch_whale_positions():
    """Cached entry point — returns (matched_list, all_in_bbox_count, fetched_at)."""
    try:
        api_key = st.secrets["aisstream-io_key"]
    except KeyError:
        return None, 0, None

    records = asyncio.run(_collect_ais(api_key, LISTEN_SECONDS))
    matched = _match_fleet(records)
    return matched, len(records), datetime.now(pytz.timezone('America/Vancouver'))


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
        matched, total_in_bbox, fetched_at = fetch_whale_positions()

    if matched is None:
        draw.error("AISStream.io key missing — set `aisstream-io_key` in Streamlit secrets.")
        return

    if fetched_at is not None:
        draw.caption(
            f"Last fetch: {fetched_at.strftime('%I:%M:%S %p')} · "
            f"{total_in_bbox} vessels heard in bounding box · "
            f"{len(matched)} matched whale-watching fleet"
        )

    if not matched:
        draw.info(
            "No whale-watching vessels broadcasting right now. "
            "They may be in port, off duty, or out of AIS range. "
            "The page auto-refreshes every 5 minutes."
        )

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
