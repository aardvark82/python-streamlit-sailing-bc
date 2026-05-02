"""Live location tracker for Alex's Zodiac Pro 420 (Teltonika FMM13A) via flespi.io.

Reads `flespi_api_key` from `.streamlit/secrets.toml`. Resolves the device's
internal flespi ID from its IMEI on first call (cached 10 min), then queries:
  - /devices/{id}/telemetry/all  → latest position, speed
  - /devices/{id}/messages       → last 6 hours of position history

Renders a Plotly Mapbox map (open-street-map style — no Mapbox token needed)
with the historical trail and the current marker.
"""

import json
import time as time_module
from datetime import datetime, timedelta

import requests
import streamlit as st
import pandas as pd
import pytz
import plotly.graph_objects as go

from utils import display_last_updated_badge


FLESPI_BASE = "https://flespi.io/gw"
DEVICE_IMEI = "862094065008336"
DEVICE_NAME = "Zodiac Pro 420"
DEVICE_MODEL = "Teltonika FMM13A"

# Bounding box per user spec:
#   Gibsons BC (W) ↔ Indian Arm tip (E)
#   Tsawwassen (S) ↔ Porteau Cove (N)
BBOX = {
    'south': 49.00,   # Tsawwassen
    'north': 49.55,   # Porteau Cove
    'west':  -123.51, # Gibsons
    'east':  -122.86, # Indian Arm tip
}
CENTER_LAT = (BBOX['south'] + BBOX['north']) / 2
CENTER_LON = (BBOX['west'] + BBOX['east']) / 2


def _flespi_headers():
    token = st.secrets["flespi_api_key"]
    return {"Authorization": f"FlespiToken {token}"}


@st.cache_data(ttl=600, show_spinner=False)
def _resolve_device_id_by_imei(imei):
    """Look up flespi internal device ID from IMEI. Cached 10 minutes —
    the ID is stable; the underlying token rarely changes."""
    url = f"{FLESPI_BASE}/devices/all"
    r = requests.get(url, headers=_flespi_headers(), timeout=15)
    r.raise_for_status()
    items = (r.json() or {}).get('result') or []
    for d in items:
        cfg = d.get('configuration') or {}
        if str(cfg.get('ident', '')) == str(imei):
            return d.get('id')
    return None


def _fetch_latest_telemetry(device_id):
    """Get latest telemetry values for the device.
    Returns flat dict {key: value} drawn from telemetry[key]['value']."""
    url = f"{FLESPI_BASE}/devices/{device_id}/telemetry/all"
    r = requests.get(url, headers=_flespi_headers(), timeout=15)
    r.raise_for_status()
    items = (r.json() or {}).get('result') or []
    if not items:
        return {}
    tele = items[0].get('telemetry') or {}
    flat = {}
    for k, v in tele.items():
        if isinstance(v, dict) and 'value' in v:
            flat[k] = v['value']
            # Track per-key timestamp so we can derive freshness
            if 'ts' in v:
                flat.setdefault('_ts', v['ts'])
                flat['_ts'] = max(flat['_ts'], v['ts'])
    return flat


def _fetch_messages_last_n_hours(device_id, hours=6):
    """Pull all device messages from the last N hours."""
    now_unix = int(time_module.time())
    url = f"{FLESPI_BASE}/devices/{device_id}/messages"
    params = {
        "data": json.dumps({
            "from": now_unix - hours * 3600,
            "to": now_unix,
            "count": 5000,
        })
    }
    r = requests.get(url, headers=_flespi_headers(), params=params, timeout=20)
    r.raise_for_status()
    return (r.json() or {}).get('result') or []


def _format_age(unix_ts, now_van):
    if not unix_ts:
        return ''
    try:
        dt = datetime.fromtimestamp(int(unix_ts), tz=pytz.UTC).astimezone(now_van.tzinfo)
        delta = now_van - dt
        secs = int(delta.total_seconds())
        if secs < 0:
            return 'just now'
        if secs < 60:
            return f"{secs}s ago"
        if secs < 3600:
            return f"{secs // 60}min ago"
        if secs < 86400:
            h = secs // 3600
            return f"{h} hour ago" if h == 1 else f"{h} hours ago"
        d = secs // 86400
        return f"{d} day ago" if d == 1 else f"{d} days ago"
    except Exception:
        return ''


def display_alex_page(container=None):
    draw = container or st
    draw.subheader("📍 Alex's Zodiac Pro 420 — Live Location")

    # Secret presence check
    try:
        st.secrets["flespi_api_key"]
    except (KeyError, FileNotFoundError):
        draw.error("`flespi_api_key` is missing from `.streamlit/secrets.toml`.")
        return

    # Resolve device ID from IMEI
    try:
        device_id = _resolve_device_id_by_imei(DEVICE_IMEI)
    except Exception as e:
        draw.error(f"Failed to look up flespi device by IMEI {DEVICE_IMEI}: {e}")
        return
    if device_id is None:
        draw.error(f"No flespi device found for IMEI {DEVICE_IMEI}")
        return

    # Latest telemetry
    try:
        with st.spinner("Fetching latest telemetry…"):
            tele = _fetch_latest_telemetry(device_id)
    except Exception as e:
        draw.error(f"Telemetry fetch failed: {e}")
        return

    lat   = tele.get('position.latitude')
    lon   = tele.get('position.longitude')
    speed = tele.get('position.speed')
    last_ts = tele.get('_ts') or tele.get('server.timestamp') or tele.get('timestamp')

    # Teltonika reports the internal battery voltage; flespi may keep the raw
    # millivolts or normalise to volts depending on device profile. Try a
    # handful of common key names and convert mV → V if the number looks big.
    def _read_voltage(*keys):
        for k in keys:
            v = tele.get(k)
            if v is None:
                continue
            try:
                f = float(v)
            except (TypeError, ValueError):
                continue
            # Heuristic: anything > 100 is almost certainly millivolts.
            return f / 1000.0 if f > 100 else f
        return None

    battery_v = _read_voltage(
        'battery.voltage',
        'battery.level',
        'battery.current.voltage',
    )
    external_v = _read_voltage(
        'external.powersource.voltage',
        'external.power.voltage',
        'power.supply.voltage',
        'power.voltage',
    )

    now_van = datetime.now(pytz.timezone('America/Vancouver'))
    last_seen_str = _format_age(last_ts, now_van) or "no recent fix"

    # Prominent staleness banner so the user immediately knows how fresh
    # the position is (color-coded green/orange/red by age).
    display_last_updated_badge(draw, last_ts, label="Last seen")

    # ── Top metrics ──
    # Row 1: Latitude / Speed / Longitude (the user-facing primary fix)
    c1, c2, c3 = draw.columns(3)
    c1.metric("Latitude",  f"{lat:.5f}"  if lat   is not None else "—")
    c2.metric("Speed (kph)", f"{speed:.1f}" if speed is not None else "—")
    c3.metric("Longitude", f"{lon:.5f}"  if lon   is not None else "—")

    # Row 2: Internal battery + external power (boat 12V) when reported
    bcol1, bcol2 = draw.columns(2)
    bcol1.metric(
        "🔋 Battery (V)",
        f"{battery_v:.2f}" if battery_v is not None else "—",
        help="Teltonika FMM13A internal backup battery voltage.",
    )
    bcol2.metric(
        "⚡ External Power (V)",
        f"{external_v:.2f}" if external_v is not None else "—",
        help="External power source (e.g. boat 12V) feeding the tracker.",
    )

    # ── Last 6h trail ──
    try:
        with st.spinner("Fetching last 6 hours of positions…"):
            msgs = _fetch_messages_last_n_hours(device_id, hours=6)
    except Exception as e:
        draw.warning(f"History fetch failed: {e}")
        msgs = []

    pts = []
    for m in msgs:
        plat = m.get('position.latitude')
        plon = m.get('position.longitude')
        ts   = m.get('timestamp') or m.get('server.timestamp')
        if plat is not None and plon is not None:
            pts.append({
                'lat': plat, 'lon': plon, 'ts': ts,
                'speed': m.get('position.speed'),
            })
    # Sort oldest-first so the line draws in chronological order
    pts.sort(key=lambda p: p.get('ts') or 0)

    # ── Map zoom controls ──
    st.session_state.setdefault('alex_map_zoom', 9.4)
    z1, z2, _ = draw.columns([0.4, 0.4, 4])
    if z1.button("🔍+", key='alex_zoom_in', help="Zoom map in"):
        st.session_state['alex_map_zoom'] = min(
            st.session_state['alex_map_zoom'] + 1, 14)
    if z2.button("🔎−", key='alex_zoom_out', help="Zoom map out"):
        st.session_state['alex_map_zoom'] = max(
            st.session_state['alex_map_zoom'] - 1, 4)

    # ── Map ──
    fig = go.Figure()

    if pts:
        fig.add_trace(go.Scattermapbox(
            lat=[p['lat'] for p in pts],
            lon=[p['lon'] for p in pts],
            mode='lines+markers',
            marker=dict(size=6, color='#1f77b4'),
            line=dict(color='#1f77b4', width=2),
            name='Last 6h trail',
            customdata=[
                [
                    _format_age(p['ts'], now_van),
                    f"{p['speed']:.1f}" if p.get('speed') is not None else '?',
                ]
                for p in pts
            ],
            hovertemplate=(
                "Time: %{customdata[0]}<br>"
                "Speed: %{customdata[1]} kph<extra></extra>"
            ),
        ))

    # Current marker (red, on top)
    if lat is not None and lon is not None:
        speed_str = f"{speed:.1f}" if speed is not None else "?"
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode='markers+text',
            marker=dict(size=18, color='#e74c3c'),
            text=[f"📍 {last_seen_str}"],
            textposition='top center',
            textfont=dict(size=12, color='#e74c3c'),
            name='Current',
            hovertemplate=(
                f"<b>{DEVICE_NAME}</b><br>"
                f"Last seen: {last_seen_str}<br>"
                f"Speed: {speed_str} kph<extra></extra>"
            ),
        ))

    # Center on the boat's last known position so zoom in/out stays focused
    # on the vessel; fall back to the bbox center if we have no fix.
    center_lat = lat if lat is not None else CENTER_LAT
    center_lon = lon if lon is not None else CENTER_LON

    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=st.session_state.get('alex_map_zoom', 9.4),
            bounds=dict(
                west=BBOX['west'], east=BBOX['east'],
                south=BBOX['south'], north=BBOX['north'],
            ),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=560,
        legend=dict(orientation='h', y=-0.05),
    )
    draw.plotly_chart(fig, width='stretch')

    # ── Last 6 hours history table (most recent first) ──
    if pts:
        rows = []
        van_tz = pytz.timezone('America/Vancouver')
        for p in reversed(pts):  # newest at top
            try:
                dt_local = (
                    datetime.fromtimestamp(int(p['ts']), tz=pytz.UTC)
                            .astimezone(van_tz)
                )
                time_str = dt_local.strftime('%I:%M:%S %p')
            except Exception:
                time_str = '—'
            rows.append({
                'Time (PDT)': time_str,
                'Age': _format_age(p.get('ts'), now_van),
                'Latitude':  f"{p['lat']:.5f}",
                'Longitude': f"{p['lon']:.5f}",
                'Speed (kph)': f"{p['speed']:.1f}" if p.get('speed') is not None else '–',
            })
        draw.markdown("**Last 6 hours · positions**")
        try:
            draw.dataframe(pd.DataFrame(rows))
        except Exception as e:
            draw.warning(f"History table render failed: {e}")
    else:
        draw.caption("No position fixes recorded in the last 6 hours.")

    # Token expiry reminder — moved below the map per user request so it
    # doesn't push the live data down on first paint.
    draw.caption(
        f"_{DEVICE_NAME} · {DEVICE_MODEL} · IMEI {DEVICE_IMEI}_"
    )
    draw.info(
        "🔑 Flespi token expires **May 1, 2027** — generate a new one at "
        "[flespi.io](https://flespi.io) when needed."
    )
