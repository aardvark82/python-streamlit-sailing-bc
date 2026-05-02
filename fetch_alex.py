"""Live location tracker for Alex's Zodiac Pro 420 (Teltonika FMM13A) via flespi.io.

Reads `flespi_api_key` from `.streamlit/secrets.toml`. Resolves the device's
internal flespi ID from its IMEI on first call (cached 10 min), then queries:
  - /devices/{id}/telemetry/all  → latest position, speed
  - /devices/{id}/messages       → last 6 hours of position history

Renders a Plotly Mapbox map (open-street-map style — no Mapbox token needed)
with the historical trail and the current marker.
"""

import io
import json
import re
import time as time_module
from datetime import datetime, timedelta

import requests
import streamlit as st
import pandas as pd
import pytz
import plotly.graph_objects as go
from bs4 import BeautifulSoup

from utils import display_last_updated_badge, cached_fetch_url, cached_fetch_url_live


# Local marine stations to overlay on the Alex map
MARINE_STATIONS = [
    {'name': 'Pam Rocks',     'kind': 'buoy', 'buoy_id': 'WAS',
     'lat': 49.490, 'lon': -123.300, 'color': '#9333ea'},
    {'name': 'Halibut Bank',  'kind': 'buoy', 'buoy_id': '46146',
     'lat': 49.340, 'lon': -123.720, 'color': '#0891b2'},
    {'name': 'Pt Atkinson',   'kind': 'buoy', 'buoy_id': 'WSB',
     'lat': 49.3304, 'lon': -123.2646, 'color': '#16a34a'},
    {'name': 'Jericho',       'kind': 'jericho',
     'lat': 49.275, 'lon': -123.198, 'color': '#ca8a04'},
    {'name': 'Howe Sound',    'kind': 'howe_forecast',
     'lat': 49.580, 'lon': -123.300, 'color': '#dc2626'},
]


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


@st.cache_data(ttl=180, show_spinner=False)
def _fetch_buoy_summary(buoy_id):
    """Returns dict {direction, wind_kts, wave_m, raw} for a weather.gc.ca buoy.
    None values when the field isn't reported. Cached 3 min like the live buoy
    fetch elsewhere."""
    url = (
        'https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html'
        f'?mapID=02&siteID=14305&stationID={buoy_id}'
    )
    try:
        res = cached_fetch_url_live(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        table = soup.find('table', class_='table')
        if not table or not table.tbody:
            return None
        rows = table.tbody.find_all('tr')
        wind_text = rows[0].find_all('td')[0].text.strip()
        parts = wind_text.split()
        direction = parts[0] if parts else None
        nums = re.findall(r'\d+', wind_text)
        wind_kts = max(int(n) for n in nums) if nums else None
        wave_m = None
        if buoy_id in ('46146', '46304') and len(rows) > 1:
            wave_text = rows[1].find_all('td')[0].text.strip()
            wave_nums = re.findall(r'[-+]?\d*\.\d+|\d+', wave_text)
            if wave_nums:
                try:
                    wave_m = float(wave_nums[0])
                except ValueError:
                    pass
        return {'direction': direction, 'wind_kts': wind_kts,
                'wave_m': wave_m, 'raw': wind_text}
    except Exception as e:
        print(f"Buoy {buoy_id} summary failed: {e}")
        return None


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_jericho_summary():
    """Last reading from the Jericho Beach weather CSV. Cached 10 min."""
    try:
        url = "https://jsca.bc.ca/main/downld02.txt"
        res = cached_fetch_url(url)
        csv_raw = res.content.decode('utf-8')
        lines = csv_raw.splitlines()
        csv_fixed = '\n'.join(lines[3:])
        df = pd.read_csv(io.StringIO(csv_fixed), header=None, sep=r'\s+')
        last = df.iloc[-1]
        # Column positions: 7 = Wind Speed, 8 = Wind Dir, 10 = Wind Hi Speed
        wind_speed = float(last.iloc[7]) if pd.notna(last.iloc[7]) else None
        wind_dir = str(last.iloc[8]) if pd.notna(last.iloc[8]) else None
        return {'direction': wind_dir, 'wind_kts': wind_speed,
                'wave_m': None, 'raw': f"{wind_dir} {wind_speed}kts"}
    except Exception as e:
        print(f"Jericho summary failed: {e}")
        return None


@st.cache_data(ttl=900, show_spinner=False)
def _fetch_howe_sound_summary():
    """Howe Sound marine forecast — current period wind from GPT-parsed CSV."""
    try:
        from fetch_gonogo import _get_howe_sound_forecast_rows
        rows = _get_howe_sound_forecast_rows()
        if not rows:
            return None
        r = rows[0]
        wind_kts = r.get('max_wind_speed') or r.get('wind_speed')
        period = r.get('time') or 'now'
        return {'direction': None, 'wind_kts': wind_kts,
                'wave_m': None, 'raw': f"forecast {period}: {wind_kts}kts"}
    except Exception as e:
        print(f"Howe Sound forecast summary failed: {e}")
        return None


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

    # ── Marine station overlay (wind / wave at nearby weather stations) ──
    # Each station is rendered as a unicode wind-arrow glyph (1/2/3 arrows
    # by intensity, pointing downwind) instead of a plain dot, colored by
    # the same wind-speed buckets used elsewhere.
    from wind_utils import _color_for_speed, wind_arrow_glyph

    station_lats, station_lons, station_labels, station_hovers, station_colors = [], [], [], [], []
    for s in MARINE_STATIONS:
        try:
            if s['kind'] == 'buoy':
                summary = _fetch_buoy_summary(s['buoy_id'])
            elif s['kind'] == 'jericho':
                summary = _fetch_jericho_summary()
            elif s['kind'] == 'howe_forecast':
                summary = _fetch_howe_sound_summary()
            else:
                summary = None
        except Exception as e:
            print(f"Station {s['name']} fetch failed: {e}")
            summary = None

        if summary is None:
            continue

        d = summary.get('direction') or ''
        w_kts = summary.get('wind_kts')
        w_m = summary.get('wave_m')
        try:
            speed_for_color = float(w_kts) if w_kts is not None else 0
        except (TypeError, ValueError):
            speed_for_color = 0

        arrows = wind_arrow_glyph(d, w_kts)
        kts_text = f"{speed_for_color:.0f}kts" if w_kts is not None else "—"
        # Glyph + speed (and wave height when available) on stacked lines.
        # Mapbox text-field renders "\n" as a line break.
        label_lines = [arrows, kts_text]
        if w_m is not None:
            label_lines.append(f"{w_m * 100:.0f}cm")
        label = "\n".join(label_lines)

        hover = (
            f"<b>{s['name']}</b><br>"
            f"Wind: {d + ' ' if d else ''}{kts_text}"
            + (f"<br>Wave: {w_m * 100:.0f} cm" if w_m is not None else "")
            + "<extra></extra>"
        )

        station_lats.append(s['lat'])
        station_lons.append(s['lon'])
        station_labels.append(label)
        station_hovers.append(hover)
        station_colors.append(_color_for_speed(speed_for_color))

    if station_lats:
        fig.add_trace(go.Scattermapbox(
            lat=station_lats, lon=station_lons,
            mode='text',  # arrow glyphs replace the dot entirely
            text=station_labels,
            textposition='middle center',
            textfont=dict(size=15, color=station_colors),
            name='Marine stations',
            hovertemplate=station_hovers,
        ))

    # Current marker — Google Maps-style blue dot with white halo.
    # Coordinates rendered below the marker.
    if lat is not None and lon is not None:
        speed_str = f"{speed:.1f}" if speed is not None else "?"

        # White halo (drawn first so the blue dot sits on top of it)
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode='markers',
            marker=dict(size=22, color='#ffffff', opacity=0.95),
            hoverinfo='skip',
            showlegend=False,
        ))
        # Blue inner dot (Google Maps blue) + coords label below
        fig.add_trace(go.Scattermapbox(
            lat=[lat], lon=[lon],
            mode='markers+text',
            marker=dict(size=14, color='#4285F4'),
            text=[f"{lat:.5f}, {lon:.5f}"],
            textposition='bottom center',
            textfont=dict(size=11, color='#1a73e8'),
            name='Current',
            hovertemplate=(
                f"<b>{DEVICE_NAME}</b><br>"
                f"Last seen: {last_seen_str}<br>"
                f"Speed: {speed_str} kph<br>"
                f"{lat:.5f}, {lon:.5f}<extra></extra>"
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
