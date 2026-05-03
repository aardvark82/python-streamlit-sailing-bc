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
import math
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

# 1NCE SIM identifiers for the FMM13A
ONENCE_BASE = "https://api.1nce.com/management-api/v1"
SIM_ICCID = "8988228066622971526"
SIM_IMSI = "901405122971526"
SIM_MSISDN = "882285123112021"

# Reference area — used as a fallback center if the boat has no fix and as
# the anchor point for the auto-zoom calculation.
WEST_VAN_LAT, WEST_VAN_LON = 49.327, -123.156
CENTER_LAT, CENTER_LON = WEST_VAN_LAT, WEST_VAN_LON


def _haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance between two lat/lon points in kilometers."""
    R = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


def _zoom_for_distance(d_km):
    """Pick a Mapbox zoom level whose visible width covers ~2x d_km, so the
    boat sits at the center with West Vancouver comfortably in view."""
    if d_km is None:
        return 11.0
    target_width_m = max(2 * d_km * 1000, 500)  # never go below ~500m of view
    # Rough Mapbox zoom math: m/pixel ≈ 156543 * cos(lat) / 2^Z;
    # for a ~800px viewport at Vancouver latitude (cos ≈ 0.66) the visible
    # width at zoom 0 is 800 * 156543 * 0.66 ≈ 8.27e7 m.
    width_at_z0 = 800 * 156543 * 0.66
    z = math.log2(width_at_z0 / target_width_m)
    return max(3.0, min(16.0, z))


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


def _walk_for_bytes(obj, depth=0):
    """Recursively search a JSON dict for a counter field that looks like
    a 'bytes received' total. Returns (value, key_path) on first hit."""
    BYTE_KEYS = (
        'traffic_in', 'traffic_in_total', 'traffic_received', 'traffic_total',
        'bytes_received', 'bytes_received_total',
        'bytes_in', 'bytes_total', 'bytes',
        'rx', 'rx_bytes', 'received_bytes', 'received',
    )
    if depth > 5 or not isinstance(obj, dict):
        return None, None
    for k, v in obj.items():
        if k in BYTE_KEYS and isinstance(v, (int, float)) and v >= 0:
            return int(v), k
    for k, v in obj.items():
        if isinstance(v, dict):
            sub, path = _walk_for_bytes(v, depth + 1)
            if sub is not None:
                return sub, f"{k}.{path}"
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            sub, path = _walk_for_bytes(v[0], depth + 1)
            if sub is not None:
                return sub, f"{k}[0].{path}"
    return None, None


def _sum_logs_recv(records):
    """Flespi /devices/{id}/logs returns an array of events. Each connection
    close (event_code == 301) carries 'recv' and 'send' byte counters.
    Sum 'recv' across every 301 event to approximate lifetime received
    traffic from this log window."""
    total = 0
    matched = 0
    for rec in records:
        if not isinstance(rec, dict):
            continue
        if rec.get('event_code') != 301:
            continue
        v = rec.get('recv')
        if isinstance(v, (int, float)) and v >= 0:
            total += int(v)
            matched += 1
    return total, matched


@st.cache_data(ttl=600, show_spinner=False)
def _fetch_device_traffic_bytes(device_id, _cache_buster=4):
    """Fetch lifetime received traffic for the flespi DEVICE.

    Tried strategies, in order:
      1. /devices/{id}                — flat fields (rarely populated)
      2. /devices/{id}/messages?count=0 — aggregate counter (rarely populated)
      3. /devices/{id}/logs            — sum 'recv' across event_code 301
                                         (connection-close) entries
    The logs strategy is the workhorse — connection events carry per-session
    recv/send byte counts and the log retention is months.
    Returns (bytes_or_None, raw_dict_for_debug). Cached 10 minutes."""
    if device_id is None:
        return None, {'error': 'no device id'}

    debug = {'tried': [], 'permission_denied': False}

    def _record_entry(url, status, **extra):
        e = {'url': url, 'status': status}
        e.update(extra)
        debug['tried'].append(e)
        return e

    # --- Strategy 1 & 2: flat-field byte counters ---
    for url in (
        f"{FLESPI_BASE}/devices/{device_id}",
        f"{FLESPI_BASE}/devices/{device_id}/messages?count=0",
    ):
        try:
            r = requests.get(url, headers=_flespi_headers(), timeout=15)
            if r.status_code == 403:
                debug['permission_denied'] = True
            _record_entry(url, r.status_code)
            if r.status_code != 200:
                continue
            data = r.json() or {}
            debug.setdefault('last_response_flat', data)
            for item in data.get('result') or [data]:
                if not isinstance(item, dict):
                    continue
                bytes_val, path = _walk_for_bytes(item)
                if bytes_val is not None and bytes_val > 0:
                    debug['matched_field'] = path
                    debug['matched_url'] = url
                    return bytes_val, debug
        except Exception as e:
            _record_entry(url, 'exception', error=str(e))

    # --- Strategy 3: sum recv across connection-close events in /logs ---
    logs_url = f"{FLESPI_BASE}/devices/{device_id}/logs?count=1000"
    try:
        r = requests.get(logs_url, headers=_flespi_headers(), timeout=20)
        if r.status_code == 403:
            debug['permission_denied'] = True
        _record_entry(logs_url, r.status_code)
        if r.status_code == 200:
            data = r.json() or {}
            records = data.get('result') or []
            total_recv, matched_count = _sum_logs_recv(records)
            debug['logs_close_events'] = matched_count
            debug['logs_total_records'] = len(records)
            if matched_count > 0:
                debug['matched_field'] = f"sum(recv) over {matched_count} close events"
                debug['matched_url'] = logs_url
                return total_recv, debug
            debug['last_response_logs_sample'] = records[:3]
    except Exception as e:
        _record_entry(logs_url, 'exception', error=str(e))

    return None, debug


def _coerce_float(v):
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


_UNIT_TO_BYTES = {
    'B': 1, 'BYTE': 1, 'BYTES': 1,
    'KB': 1024,
    'MB': 1024 * 1024,
    'GB': 1024 * 1024 * 1024,
    'TB': 1024 ** 4,
}


@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_1nce_sim_usage(iccid, _cache_buster=3):
    """Query 1NCE 'Get SIM usage' endpoint. The output is limited to the
    last 6 months per the docs. Response shape (real, observed):
        {"stats": [
            {"date": "2026-05-03", "data": {"volume":"0.033158","volume_tx":"...",
                                            "volume_rx":"...","traffic_type":{"unit":"MB"}},
                                  "sms":  {...}},
            ...,
            {"date": "TOTAL", "data": {...}, "sms": {...}}
        ]}
    Returns (total_bytes, daily_breakdown_list, raw_debug). Cached 1 hour."""
    debug = {'tried': []}
    try:
        token = st.secrets["1nce_bearer_token"]
    except (KeyError, FileNotFoundError):
        return None, [], {'error': '1nce_bearer_token missing in secrets'}

    url = f"{ONENCE_BASE}/sims/{iccid}/usage"
    headers = {
        'Authorization': f'Bearer {token}',
        'Accept': 'application/json',
    }
    try:
        r = requests.get(url, headers=headers, timeout=20)
        debug['tried'].append({'url': url, 'status': r.status_code})
        if r.status_code != 200:
            try:
                debug['body'] = r.json()
            except Exception:
                debug['body'] = r.text[:500]
            return None, [], debug

        data = r.json() or {}
        debug['raw'] = data

        stats = data.get('stats') or data.get('result') or data.get('usage_records') or []
        if isinstance(stats, dict):
            stats = [stats]

        breakdown = []
        total_bytes = None
        for entry in stats:
            if not isinstance(entry, dict):
                continue
            date = str(entry.get('date') or '?')
            d = entry.get('data') or {}
            if not isinstance(d, dict):
                continue
            unit = ((d.get('traffic_type') or {}).get('unit') or 'MB').upper()
            scale = _UNIT_TO_BYTES.get(unit, 1024 * 1024)
            vol = _coerce_float(d.get('volume')) or 0.0
            tx = _coerce_float(d.get('volume_tx'))
            rx = _coerce_float(d.get('volume_rx'))

            bytes_total = vol * scale
            bytes_tx = tx * scale if tx is not None else None
            bytes_rx = rx * scale if rx is not None else None

            if date == 'TOTAL':
                total_bytes = int(bytes_total)
            else:
                breakdown.append({
                    'period': date,
                    'total': bytes_total,
                    'tx': bytes_tx,
                    'rx': bytes_rx,
                })

        # Fallback if there's no TOTAL row: sum the per-day totals.
        if total_bytes is None and breakdown:
            total_bytes = int(sum(b['total'] for b in breakdown))
        if total_bytes is None:
            return None, [], debug
        return total_bytes, breakdown, debug
    except Exception as e:
        debug['tried'].append({'url': url, 'error': str(e)})
        return None, [], debug


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

    # ── Map zoom (auto-default + manual ±2 buttons) ──
    # Default zoom = whatever it takes to fit the boat AND every marine
    # weather station on screen, with a small margin. User clicks of +/-
    # override and persist for the rest of the session.
    if lat is not None and lon is not None:
        anchor_points = [(WEST_VAN_LAT, WEST_VAN_LON)] + [
            (s['lat'], s['lon']) for s in MARINE_STATIONS
        ]
        max_dist_km = max(
            _haversine_km(lat, lon, alat, alon) for alat, alon in anchor_points
        )
        # 15% padding so the outermost station never sits exactly on the edge.
        auto_zoom = _zoom_for_distance(max_dist_km * 1.15)
    else:
        auto_zoom = 11.0
    st.session_state.setdefault('alex_map_zoom', auto_zoom)
    z1, z2, _ = draw.columns([0.4, 0.4, 4])
    if z1.button("🔍+", key='alex_zoom_in', help="Zoom map in"):
        st.session_state['alex_map_zoom'] = min(
            st.session_state['alex_map_zoom'] + 2, 16)
    if z2.button("🔎−", key='alex_zoom_out', help="Zoom map out"):
        st.session_state['alex_map_zoom'] = max(
            st.session_state['alex_map_zoom'] - 2, 3)

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

    # Plotly Scattermapbox.textfont.color is scalar per trace, so render one
    # trace per station to preserve per-point colors. Use markers+text rather
    # than text-only — pure 'mode=text' doesn't reliably render on the
    # open-street-map basemap. The colored dot anchors the label and
    # reinforces the speed bucket visually.
    for slat, slon, slabel, shover, scolor in zip(
        station_lats, station_lons, station_labels, station_hovers, station_colors
    ):
        fig.add_trace(go.Scattermapbox(
            lat=[slat], lon=[slon],
            mode='markers+text',
            marker=dict(size=12, color=scolor, opacity=0.95),
            text=[slabel],
            textposition='bottom center',
            # Wind value rendered in bold black for max legibility on any
            # basemap; the dot itself keeps the speed-bucket color.
            textfont=dict(size=14, color='#000000', family='Open Sans Bold'),
            name='Marine stations',
            hovertemplate=shover,
            showlegend=False,
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
            zoom=st.session_state.get('alex_map_zoom', auto_zoom),
            # No `bounds` — let the user pan freely.
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

    # ── Lifetime traffic for the flespi DEVICE ──
    # Pulled from /gw/devices/{id}* endpoints rather than channels — the
    # token consistently has device scope and the device payload exposes
    # per-device counters directly.
    rt_col, _spacer = draw.columns([0.3, 4])
    if rt_col.button("🔄", key='alex_retry_traffic',
                      help="Re-fetch flespi device traffic (clears the 10-min cache)"):
        try:
            _fetch_device_traffic_bytes.clear()
        except Exception:
            pass

    try:
        device_bytes, device_debug = _fetch_device_traffic_bytes(device_id)
    except Exception as e:
        print(f"Device traffic display failed: {e}")
        device_bytes, device_debug = None, {'error': str(e)}

    if device_bytes is not None:
        kb = device_bytes / 1024
        if kb >= 1024 * 1024:
            traffic_str = f"{kb / (1024 * 1024):.2f} GB"
        elif kb >= 1024:
            traffic_str = f"{kb / 1024:.2f} MB"
        else:
            traffic_str = f"{kb:.1f} KB"
        draw.caption(
            f"📡 Total traffic on flespi device {DEVICE_IMEI}: "
            f"**{traffic_str}** "
            f"(from `{device_debug.get('matched_field', '?')}`)"
        )
    elif device_debug.get('permission_denied'):
        draw.caption(
            f"🔒 Flespi token lacks device-read ACL — can't read total "
            f"traffic for device {DEVICE_IMEI}. Widen the token at "
            f"[flespi.io](https://flespi.io) → Tokens → edit."
        )
        with draw.expander("🔍 flespi raw debug"):
            draw.json(device_debug)
    else:
        draw.caption(
            f"📡 Total traffic for device {DEVICE_IMEI}: "
            f"not reported by API — raw response below."
        )
        with draw.expander("🔍 flespi device raw response (for debugging)"):
            draw.json(device_debug)

    # ── 1NCE SIM usage (last 6 months max) ──
    sim_rt_col, _spacer2 = draw.columns([0.3, 4])
    if sim_rt_col.button("🔄", key='alex_retry_sim',
                          help="Re-fetch 1NCE SIM usage (clears the 1-hour cache)"):
        try:
            _fetch_1nce_sim_usage.clear()
        except Exception:
            pass

    try:
        sim_total_bytes, sim_breakdown, sim_debug = _fetch_1nce_sim_usage(SIM_ICCID)
    except Exception as e:
        sim_total_bytes, sim_breakdown, sim_debug = None, [], {'error': str(e)}

    if sim_total_bytes is not None:
        kb = sim_total_bytes / 1024
        if kb >= 1024 * 1024:
            sim_str = f"{kb / (1024 * 1024):.2f} GB"
        elif kb >= 1024:
            sim_str = f"{kb / 1024:.2f} MB"
        else:
            sim_str = f"{kb:.1f} KB"
        draw.caption(
            f"📶 1NCE SIM ({SIM_ICCID}) — last 6 months: **{sim_str}**"
        )
        if sim_breakdown:
            try:
                draw.dataframe(pd.DataFrame([
                    {
                        'Period': b['period'],
                        'Total (KB)': f"{b['total'] / 1024:.1f}" if b['total'] else '–',
                        'TX (KB)': f"{b['tx'] / 1024:.1f}" if b['tx'] is not None else '–',
                        'RX (KB)': f"{b['rx'] / 1024:.1f}" if b['rx'] is not None else '–',
                    }
                    for b in sim_breakdown
                ]))
            except Exception:
                pass
    else:
        draw.caption(
            f"📶 1NCE SIM usage: parsing returned 0 records — "
            f"check the raw response below."
        )

    # Always surface the raw 1NCE response in an expander so the parser can
    # be aligned to the actual schema if it ever drifts.
    with draw.expander("🔍 1NCE raw response (for debugging)"):
        draw.json(sim_debug)
