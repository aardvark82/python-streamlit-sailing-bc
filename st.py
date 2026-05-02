#  > pip install -r requirements.txt
#  > python -m streamlit run st.py
# http://localhost:8501/
# http://python-app-sailing-bc-nckqtfynerhhf26ujtt5u6

import io
import re

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote

from streamlit_autorefresh import st_autorefresh

from utils import cached_fetch_url, cached_fetch_url_live, prettydate, displayStreamlitDateTime
from fetch_forecast import (
    display_marine_forecast_for_url,
    display_summary_marine_forecast_for_url,
    openAIFetchForecastForURL,
    standardize_wind_direction,
    clean_wind_speed,
)
from fetch_beach import display_beach_quality_for_sandy_cove
from fetch_weather import display_weather_info, display_clear_skies_html
from fetch_tides import display_point_atkinson_tides
from wind_utils import create_arrow_html
from fetch_gonogo import display_gonogo_sidebar, display_gonogo_page, display_kiosk_page
from fetch_whales import display_whales_page
from fetch_whales2 import display_whales2_page
from fetch_alex import display_alex_page
from pathlib import Path

# Read version from VERSION file
_version_file = Path(__file__).parent / "VERSION"
APP_VERSION = int(_version_file.read_text().strip()) if _version_file.exists() else 0

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="data_refresher")

# Trim Streamlit's default ~6rem top padding so pages start near the top of
# the viewport on desktop and mobile. The kiosk page overrides this further.
st.markdown(
    """
    <style>
        .block-container {
            padding-top: 1.2rem !important;
            padding-bottom: 2rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Constants ---
URL_FORECAST_HOWESOUND = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400'
URL_FORECAST_SOUTH_NANAIMO = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14305'
URL_FORECAST_NORTH_NANAIMO = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14301'
VANCOUVER_LAT, VANCOUVER_LON = 49.32, -123.16
SQUAMISH_LAT, SQUAMISH_LON = 49.7, -123.16
LIONSBAY_LAT, LIONSBAY_LON = 49.45, -123.16

URL_MAP = {
    "Howe Sound": URL_FORECAST_HOWESOUND,
    "South of Nanaimo": URL_FORECAST_SOUTH_NANAIMO,
    "North of Nanaimo": URL_FORECAST_NORTH_NANAIMO,
}


def displayWindWarningIfNeeded(wind_speed, container=None):
    """Display a warning badge if wind speed exceeds 9 knots."""
    draw = container or st

    try:
        if isinstance(wind_speed, (pd.DataFrame, pd.Series)):
            wind_speed = pd.to_numeric(wind_speed, errors='coerce')
            if isinstance(wind_speed, pd.Series):
                wind_speed = wind_speed.iloc[0]

        if isinstance(wind_speed, str):
            wind_speed = float(wind_speed)

        if wind_speed > 9:
            draw.badge("Wind warning", color="orange")
    except (ValueError, TypeError) as e:
        print(f"Error processing wind speed: {e}")


# ──────────────────────────────────────────────
# Page functions (one per nav item)
# ──────────────────────────────────────────────

def page_gonogo():
    try:
        display_gonogo_page(container=st, page_links=PAGE_LINKS)
    except Exception as e:
        st.error(f"Failed to load Go/No-Go: {e}")


def page_dashboard():
    try:
        display_weather_info(container=st, lat=VANCOUVER_LAT, long=VANCOUVER_LON, title="Weather")
    except Exception as e:
        st.error(f"Failed to load weather: {e}")

    try:
        display_clear_skies_html(container=st, title="Clear Skies")
    except Exception as e:
        st.error(f"Failed to load clear skies: {e}")

    try:
        display_summary_marine_forecast_for_url(draw=st, url=URL_FORECAST_HOWESOUND, title="Howe Sound")
        display_summary_marine_forecast_for_url(draw=st, url=URL_FORECAST_SOUTH_NANAIMO, title="South of Nanaimo")
        display_summary_marine_forecast_for_url(draw=st, url=URL_FORECAST_NORTH_NANAIMO, title="North of Nanaimo")
    except Exception as e:
        st.error(f"Failed to load marine forecast summary: {e}")

    try:
        display_beach_quality_for_sandy_cove(draw=st, title="Beach water quality Sandy Cove")
    except Exception as e:
        st.error(f"Failed to load beach quality: {e}")

    try:
        display_point_atkinson_tides(container=st)
    except Exception as e:
        st.error(f"Failed to load tides: {e}")


def page_jericho():
    try:
        parseJerichoWindHistory(container=st)
    except Exception as e:
        st.error(f"Failed to load Jericho wind data: {e}")


def page_halibut():
    try:
        refreshBuoy('46146', 'Halibut Bank', container=st,
                    forecast_url=URL_FORECAST_SOUTH_NANAIMO, forecast_title='South of Nanaimo')
    except Exception as e:
        st.error(f"Failed to load Halibut Bank buoy: {e}")


def page_english_bay():
    try:
        refreshBuoy('46304', 'English Bay', container=st,
                    forecast_url=URL_FORECAST_SOUTH_NANAIMO, forecast_title='South of Nanaimo')
    except Exception as e:
        st.error(f"Failed to load English Bay buoy: {e}")


def page_atkinson():
    try:
        refreshBuoy('WSB', 'Point Atkinson', container=st,
                    forecast_url=URL_FORECAST_SOUTH_NANAIMO, forecast_title='South of Nanaimo')
    except Exception as e:
        st.error(f"Failed to load Point Atkinson buoy: {e}")


def page_pamrocks():
    try:
        refreshBuoy('WAS', 'Pam Rocks', container=st,
                    forecast_url=URL_FORECAST_HOWESOUND, forecast_title='Howe Sound')
    except Exception as e:
        st.error(f"Failed to load Pam Rocks buoy: {e}")


def page_whales():
    try:
        display_whales_page(container=st)
    except Exception as e:
        st.error(f"Failed to load whale tracker: {e}")


def page_whales2():
    try:
        display_whales2_page(container=st)
    except Exception as e:
        st.error(f"Failed to load whale tracker 2: {e}")


def page_alex():
    try:
        display_alex_page(container=st)
    except Exception as e:
        st.error(f"Failed to load Alex Location: {e}")


def page_forecast():
    region = st.selectbox("Region", ["Howe Sound", "South of Nanaimo", "North of Nanaimo"])
    try:
        display_marine_forecast_for_url(draw=st, url=URL_MAP[region], title=region)
    except Exception as e:
        st.error(f"Failed to load {region} forecast: {e}")


def page_beach():
    try:
        display_beach_quality_for_sandy_cove(draw=st, title="Beach water quality Sandy Cove")
    except Exception as e:
        st.error(f"Failed to load beach quality: {e}")


def page_tides():
    try:
        display_point_atkinson_tides(container=st)
    except Exception as e:
        st.error(f"Failed to load tides: {e}")


def page_squamish():
    try:
        display_weather_info(container=st, lat=SQUAMISH_LAT, long=SQUAMISH_LON, title="Squamish")
    except Exception as e:
        st.error(f"Failed to load Squamish weather: {e}")


def page_lionsbay():
    try:
        display_weather_info(container=st, lat=LIONSBAY_LAT, long=LIONSBAY_LON, title="Lions Bay")
    except Exception as e:
        st.error(f"Failed to load Lions Bay weather: {e}")


def page_kiosk():
    try:
        # _pg_gonogo is defined later at module scope; safe to reference here
        # because this function is only called via pg.run() after all pages
        # are constructed.
        display_kiosk_page(home_page=_pg_gonogo)
    except Exception as e:
        st.error(f"Failed to load kiosk mode: {e}")


def parseJerichoWindHistory(container=None):
    container.subheader("Jericho Beach Wind History")
    draw = container or st

    url = "https://jsca.bc.ca/main/downld02.txt"
    res = cached_fetch_url(url)

    csv_raw = res.content.decode('utf-8')
    lines = csv_raw.splitlines()
    csv_fixed = '\n'.join(lines[3:])

    df = pd.read_csv(io.StringIO(csv_fixed), header=None, sep=r'\s+')

    df.columns = [
        'Date', 'Time', 'Temp Out', 'Temp Hi', 'Temp Low', 'Hum Out',
        'Dew Pt.', 'Wind Speed', 'Wind Dir', 'Wind Run', 'Wind Hi Speed',
        'Wind Hi Dir', 'Wind Chill', 'Heat Index', 'THW Index', 'Bar',
        'Rain', 'Rain Rate', 'Heat D-D', 'Cool D-D', 'In Temp', 'In Hum',
        'In Dew', 'In Heat', 'In EMC', 'In Air Density', 'Wind Samp',
        'Wind TX', 'IS Recept.', 'Arc Int',
    ]

    # Combine date and time columns
    df['datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], utc=False)
    df = df.drop(columns=['Date', 'Time'])
    cols = df.columns.tolist()
    df = df[['datetime'] + cols[:-1]]

    last_row = df.iloc[-1]
    prev_row = df.iloc[-2] if len(df) > 1 else last_row

    # Wind trend: compare current speed to previous reading
    try:
        current_speed = float(last_row['Wind Speed'])
        prev_speed = float(prev_row['Wind Speed'])
        wind_delta = current_speed - prev_speed
        wind_trend = f"{wind_delta:+.0f}" if wind_delta != 0 else None
    except (ValueError, TypeError):
        wind_trend = None

    displayWindWarningIfNeeded(last_row['Wind Hi Speed'], container=draw)
    displayStreamlitDateTime(
        last_row['datetime'], draw, label="Last reading",
        source_url=url, source_label='jsca.bc.ca csv',
    )

    col1, col2, col3 = draw.columns(3)
    col1.metric(label="Wind Speed", value=last_row['Wind Speed'], delta=wind_trend, delta_color="inverse")
    col2.metric(label="Wind High", value=last_row['Wind Hi Speed'])
    col3.metric(label="Wind Direction", value=last_row['Wind Dir'])
    col3.markdown(create_arrow_html(last_row['Wind Dir'], last_row['Wind Hi Speed']),
                  unsafe_allow_html=True)

    col1, col2, col3 = draw.columns(3)
    col1.metric(label="Bar", value=last_row['Bar'])
    col2.metric(label="Rain", value=last_row['Rain'])
    col3.metric(label="Temperature", value=last_row['Temp Out'])

    import plotly.express as px

    df_tail = df.tail(24)
    fig = px.line(df_tail, x='datetime', y=['Wind Speed', 'Wind Hi Speed'],
                  title="Jericho Wind History (Last 12 Hours)")
    fig.update_yaxes(range=[0, 30], title="Speed (knots)")
    fig.add_hline(y=15, line_dash="dot", line_color="red")

    vancouver_tz = pytz.timezone('America/Vancouver')
    current_time = datetime.now(vancouver_tz)
    current_time_ts = current_time.timestamp() * 1000

    fig.add_vline(
        x=current_time_ts,
        line_width=2, line_dash="dash", line_color="red",
        annotation_text="Now", annotation_position="top right"
    )
    draw.plotly_chart(fig, width='stretch')
    with draw.expander("Raw Data (Last 12 Hours)"):
        st.dataframe(df.tail(24))


def drawMapWithBuoy(container=None, buoy=None):
    latlong = None
    if buoy == '46146':
        latlong = pd.DataFrame({'latitude': [49.34], 'longitude': [-123.72]})
    if buoy == '46304':
        latlong = pd.DataFrame({'latitude': [49.300], 'longitude': [-123.360]})
    if buoy == 'WSB':
        latlong = pd.DataFrame({'latitude': [49.330], 'longitude': [-123.2646]})
    if buoy == 'WAS':
        latlong = pd.DataFrame({'latitude': [49.49], 'longitude': [-123.3]})
    container.map(latlong, zoom=10)


@st.cache_data(ttl=144600)
def get_buoy_observation_from_cf(base_url, headers, buoy_id, timestamp_str):
    encoded_key_wind = quote(f"{buoy_id}_wind_{timestamp_str}", safe='')
    encoded_key_dir = quote(f"{buoy_id}_direction_{timestamp_str}", safe='')
    encoded_key_wave = quote(f"{buoy_id}_wave_{timestamp_str}", safe='')

    r_speed = requests.get(f"{base_url}/values/{encoded_key_wind}", headers=headers)
    r_dir = requests.get(f"{base_url}/values/{encoded_key_dir}", headers=headers)
    r_wave = requests.get(f"{base_url}/values/{encoded_key_wave}", headers=headers)

    speed = float(r_speed.text) if r_speed.status_code == 200 else 0.0
    direction = r_dir.text if r_dir.status_code == 200 else "N/A"
    try:
        wave = float(r_wave.text) if r_wave.status_code == 200 else None
    except ValueError:
        wave = None

    return speed, direction, wave


def get_resolved_namespace_id(account_id, api_token, name_or_id):
    if "storage/kv/namespaces/" not in name_or_id:
        headers = {"Authorization": f"Bearer {api_token}"}
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            namespaces = response.json().get("result", [])
            for ns in namespaces:
                if ns.get("title") == name_or_id:
                    return ns.get("id")
        except requests.exceptions.RequestException as e:
            print(f"Error resolving namespace ID: {e}")
            return name_or_id
    return name_or_id


@st.cache_data(ttl=600)
def cached_kv_list(_container, buoy_id, api_token, account_id, namespace_id):
    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        params = {'prefix': f"{buoy_id}_wind_", 'limit': 1000}
        return requests.get(f"{base_url}/keys", params=params, headers=headers)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        _container.error("Could not load historical data")
    return None


# ──────────────────────────────────────────────
# Merged wind chart: past 3 days (buoy) + next 3 days (forecast)
# ──────────────────────────────────────────────

_COMPASS_DEGREES = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}


def _fetch_buoy_wind_history_df(container, buoy_id, days_back=3):
    """Fetch buoy wind history from Cloudflare KV.
    Returns DataFrame with columns [timestamp, wind_speed, direction, wave_height] or None."""
    try:
        account_id = st.secrets["cloudflare_account_id"]
        namespace_id_raw = st.secrets["cloudflare_namespace_id"]
        api_token = st.secrets["cloudflare_api_token"]
    except KeyError:
        if container:
            container.warning("Cloudflare secrets not configured")
        return None

    namespace_id = get_resolved_namespace_id(account_id, api_token, namespace_id_raw)
    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        response = cached_kv_list(container, buoy_id, api_token, account_id, namespace_id)
        if response is None or response.status_code != 200:
            return None
        data = response.json()
        if not data.get("success", False):
            return None

        all_keys = [item["name"] for item in data.get("result", [])]
        cutoff = datetime.now(pytz.timezone('America/Vancouver')) - pd.Timedelta(days=days_back)
        data_points = []

        for key in all_keys:
            if not key.startswith(f"{buoy_id}_wind_"):
                continue
            timestamp_str = key.replace(f"{buoy_id}_wind_", "")
            try:
                timestamp = datetime.fromisoformat(timestamp_str)
                if timestamp < cutoff:
                    continue
                wind_value, direction, wave_height = get_buoy_observation_from_cf(
                    base_url, headers, buoy_id, timestamp_str)
                data_points.append({
                    'timestamp': timestamp,
                    'wind_speed': wind_value,
                    'direction': direction,
                    'wave_height': wave_height,
                })
            except Exception as e:
                print(f"Error processing key {key}: {e}")

        if not data_points:
            return None
        return pd.DataFrame(data_points).sort_values('timestamp').reset_index(drop=True)
    except Exception as e:
        print(f"Error fetching historical data: {e}")
        return None


def _parse_forecast_timestamps(time_labels, now_van):
    """Convert forecast text labels (e.g. 'Monday afternoon', 'tonight', 'overnight')
    into Vancouver-local datetimes. First row anchors to now. Unparseable labels step
    forward 8 hours from the previous timestamp. Output is always monotonically increasing."""
    weekday_map = {
        'mon': 0, 'monday': 0,
        'tue': 1, 'tuesday': 1, 'tues': 1,
        'wed': 2, 'wednesday': 2,
        'thu': 3, 'thursday': 3, 'thurs': 3,
        'fri': 4, 'friday': 4,
        'sat': 5, 'saturday': 5,
        'sun': 6, 'sunday': 6,
    }
    # Period → local hour-of-day
    period_hours = [
        ('early morning', 5),
        ('late tonight', 23),
        ('overnight', 2),     # special-cased: rolls to next day
        ('morning', 8),
        ('afternoon', 14),
        ('evening', 20),
        ('tonight', 20),
        ('night', 22),
    ]

    vancouver_tz = pytz.timezone('America/Vancouver')
    timestamps = []
    last_dt = None

    for i, raw in enumerate(time_labels):
        label = str(raw).strip().lower() if raw is not None else ''
        if i == 0:
            dt = now_van
        else:
            target_date = None
            for key, day_num in weekday_map.items():
                if key in label:
                    days_ahead = (day_num - now_van.weekday()) % 7
                    target_date = (now_van + timedelta(days=days_ahead)).date()
                    break
            if target_date is None:
                if 'tomorrow' in label:
                    target_date = (now_van + timedelta(days=1)).date()
                elif 'today' in label or 'tonight' in label:
                    target_date = now_van.date()

            hour = None
            rolls_next_day = False
            for period, h in period_hours:
                if period in label:
                    hour = h
                    if period == 'overnight':
                        rolls_next_day = True
                    break

            if target_date is not None and hour is not None:
                if rolls_next_day:
                    target_date = target_date + timedelta(days=1)
                dt = vancouver_tz.localize(datetime(
                    target_date.year, target_date.month, target_date.day, hour, 0, 0))
            else:
                dt = (last_dt or now_van) + timedelta(hours=8)

        # Strict monotonic — forecast rows must march forward
        if last_dt is not None and dt <= last_dt:
            dt = last_dt + timedelta(hours=6)

        timestamps.append(dt)
        last_dt = dt

    return timestamps


def _fetch_forecast_wind_df(url):
    """Fetch GPT-parsed marine forecast and attach real timestamps.
    Returns DataFrame [timestamp, time_label, wind speed, max wind speed, wind direction]
    or None."""
    if not url:
        return None
    try:
        chatgpt_forecast = openAIFetchForecastForURL(url=url)
        if not chatgpt_forecast:
            return None
        text = chatgpt_forecast.replace('```csv', '').replace('```', '')
        df = pd.read_csv(io.StringIO(text), sep=',', on_bad_lines='skip')
        df = df.dropna(how='all').reset_index(drop=True)
        df.columns = df.columns.str.strip().str.lower()

        if 'wind_direction' in df.columns:
            df['wind direction'] = df['wind_direction'].astype(str).str.lower().apply(standardize_wind_direction)
        elif 'wind direction' in df.columns:
            df['wind direction'] = df['wind direction'].astype(str).str.lower().apply(standardize_wind_direction)
        else:
            df['wind direction'] = None

        if 'wind speed' in df.columns:
            df['wind speed'] = df['wind speed'].apply(clean_wind_speed)
        else:
            df['wind speed'] = 0

        if 'max wind speed' in df.columns:
            df['max wind speed'] = df['max wind speed'].apply(clean_wind_speed)
        else:
            df['max wind speed'] = df['wind speed']

        if 'time' not in df.columns:
            return None
        df['time_label'] = df['time'].astype(str)

        now_van = datetime.now(pytz.timezone('America/Vancouver'))
        df['timestamp'] = _parse_forecast_timestamps(df['time_label'].tolist(), now_van)
        return df
    except Exception as e:
        print(f"Error fetching forecast wind: {e}")
        return None


def plot_merged_wind_chart(container, buoy_id, forecast_url, forecast_title):
    """Plot past 3 days of buoy wind + next 3 days of forecast wind on one chart.
    Returns the buoy past-history DataFrame (for reuse by the wave chart) or None."""
    import plotly.graph_objects as go

    past_df = _fetch_buoy_wind_history_df(container, buoy_id)
    forecast_df = _fetch_forecast_wind_df(forecast_url)

    if (past_df is None or past_df.empty) and (forecast_df is None or forecast_df.empty):
        container.warning(f"No wind data available for buoy {buoy_id}")
        return past_df

    now_van = datetime.now(pytz.timezone('America/Vancouver'))
    x_min = now_van - timedelta(days=3)
    x_max = now_van + timedelta(days=3)

    fig = go.Figure()

    # Past: buoy observations as arrow markers
    if past_df is not None and not past_df.empty:
        df = past_df.copy()
        df['degree'] = df['direction'].map(_COMPASS_DEGREES).fillna(0)
        df['rotation'] = (180 - df['degree']) % 360
        fig.add_trace(go.Scatter(
            x=df['timestamp'], y=df['wind_speed'],
            mode='markers', name='Observed (Buoy)',
            marker=dict(
                symbol='arrow-up', size=14, angle=df['rotation'],
                color='#1f77b4', line=dict(width=1, color='DarkSlateGrey'),
            ),
            customdata=df['direction'],
            hovertemplate="<b>Observed</b><br>%{x}<br>Speed: %{y:.1f} kts<br>Dir: %{customdata}<extra></extra>",
        ))

    # Future: forecast line + gust line + direction arrows (where known)
    if forecast_df is not None and not forecast_df.empty:
        fdf = forecast_df.copy()

        fig.add_trace(go.Scatter(
            x=fdf['timestamp'], y=fdf['wind speed'],
            mode='lines', name='Forecast',
            line=dict(color='#ff7f0e', width=3, dash='dot'),
            customdata=fdf['time_label'],
            hovertemplate="<b>Forecast</b><br>%{customdata}<br>Speed: %{y:.0f} kts<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=fdf['timestamp'], y=fdf['max wind speed'],
            mode='lines', name='Forecast Gust',
            line=dict(color='#ff7f0e', width=2, dash='dash'),
            opacity=0.5,
            customdata=fdf['time_label'],
            hovertemplate="<b>Gust</b><br>%{customdata}<br>Speed: %{y:.0f} kts<extra></extra>",
        ))

        dir_df = fdf[fdf['wind direction'].isin(_COMPASS_DEGREES.keys())].copy()
        if not dir_df.empty:
            dir_df['degree'] = dir_df['wind direction'].map(_COMPASS_DEGREES)
            dir_df['rotation'] = (180 - dir_df['degree']) % 360
            fig.add_trace(go.Scatter(
                x=dir_df['timestamp'], y=dir_df['wind speed'],
                mode='markers', name='Forecast Direction',
                marker=dict(
                    symbol='arrow-up', size=14, angle=dir_df['rotation'],
                    color='#ff7f0e', line=dict(width=1, color='DarkSlateGrey'),
                ),
                customdata=dir_df['wind direction'],
                hovertemplate="<b>Forecast dir</b>: %{customdata}<extra></extra>",
                showlegend=False,
            ))

    # Blue "Now" vertical line — use add_shape + add_annotation manually because
    # add_vline with tz-aware datetime has historically hit Plotly's
    # "int + datetime" bug on some versions.
    now_hour = now_van.hour % 12 or 12
    now_label = f"Now · {now_van.strftime('%a')} {now_hour}:{now_van.strftime('%M %p')} PDT"
    now_x = pd.Timestamp(now_van)
    fig.add_shape(
        type="line",
        x0=now_x, x1=now_x, y0=0, y1=40,
        line=dict(color="blue", width=2),
        xref="x", yref="y",
    )
    fig.add_annotation(
        x=now_x, y=0,
        text=now_label,
        showarrow=False,
        yshift=-20,
        font=dict(size=11, color="blue"),
        xref="x", yref="y",
    )
    fig.add_hline(y=15, line_dash="dot", line_color="red", opacity=0.4)

    # Tide overlay (orange dotted line) on a secondary y-axis (0-5m, no label).
    # Sampled hourly across the visible window so the curve is smooth without
    # being expensive.
    try:
        from fetch_gonogo import _get_tide_data, _tide_at
        _tide_extremes_df, tide_x_ts, tide_y_h = _get_tide_data()
        if tide_x_ts is not None and tide_y_h is not None:
            tide_x_dt = []
            tide_y_m = []
            cursor = x_min
            step = timedelta(hours=1)
            while cursor <= x_max:
                h = _tide_at(tide_x_ts, tide_y_h, cursor)
                if h is not None:
                    tide_x_dt.append(pd.Timestamp(cursor))
                    tide_y_m.append(h)
                cursor += step
            if tide_x_dt:
                fig.add_trace(go.Scatter(
                    x=tide_x_dt, y=tide_y_m,
                    mode='lines', name='Tide (m)',
                    line=dict(color='#ff7f0e', width=2, dash='dot'),
                    yaxis='y2',
                    hovertemplate="Tide: %{y:.2f}m<extra></extra>",
                ))
    except Exception as e:
        print(f"Tide overlay skipped: {e}")

    fig.update_layout(
        title=f"Wind · Past 3 days + Forecast ({forecast_title}) — Buoy {buoy_id}",
        xaxis_title="",
        yaxis_title="Wind Speed (knots)",
        yaxis=dict(range=[0, 40]),
        # Secondary axis for tide overlay — 0-5m, hidden label per user request.
        yaxis2=dict(
            range=[0, 5],
            overlaying='y',
            side='right',
            showgrid=False,
            showticklabels=False,
            title=None,
        ),
        xaxis=dict(range=[x_min, x_max]),
        hovermode='closest',
        legend=dict(orientation='h', y=-0.15),
        margin=dict(l=40, r=20, t=50, b=60),
    )

    container.plotly_chart(fig, width='stretch')
    return past_df


def plot_wave_history_chart(container, past_df, buoy_id):
    """Plot wave-height history from a past-buoy DataFrame (skips if no wave data)."""
    if past_df is None or past_df.empty:
        return
    df_waves = past_df.dropna(subset=['wave_height']).copy()
    if df_waves.empty:
        return

    import plotly.express as px
    df_waves['wave_height_cm'] = df_waves['wave_height'] * 100
    now_van = datetime.now(pytz.timezone('America/Vancouver'))
    three_days_ago = now_van - timedelta(days=3)

    fig_wave = px.line(df_waves,
                       x='timestamp', y='wave_height_cm',
                       title=f'Wave Height Over Last 3 Days - Buoy {buoy_id}',
                       labels={'wave_height_cm': 'Wave Height (cm)', 'timestamp': 'Time'})
    fig_wave.update_xaxes(range=[three_days_ago, now_van])
    fig_wave.update_yaxes(range=[0, 200])
    fig_wave.add_hline(y=33, line_dash="dot", line_color="green")
    fig_wave.add_hline(y=75, line_dash="dot", line_color="orange")
    fig_wave.add_hline(y=100, line_dash="dot", line_color="red")
    container.plotly_chart(fig_wave, width='stretch')


def record_buoy_data_history(buoy, container, wind_speed, direction, wave_height):
    current_time = datetime.now(pytz.timezone('America/Vancouver'))
    current_time = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0)
    timestamp = current_time.isoformat(timespec='minutes')

    try:
        account_id = st.secrets["cloudflare_account_id"]
        namespace_id_raw = st.secrets["cloudflare_namespace_id"]
        api_token = st.secrets["cloudflare_api_token"]
    except KeyError:
        print("Cloudflare secrets missing")
        return

    namespace_id = get_resolved_namespace_id(account_id, api_token, namespace_id_raw)

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    @st.cache_data(ttl=1800)
    def store_buoy_data_cached(base_url, headers, buoy, timestamp, wind_speed, direction, wave_height):
        try:
            key_wind = f"{buoy}_wind_{timestamp}"
            key_dir = f"{buoy}_direction_{timestamp}"
            key_wave = f"{buoy}_wave_{timestamp}"
            requests.put(f"{base_url}/values/{quote(key_wind, safe='')}", headers=headers, data=str(wind_speed))
            requests.put(f"{base_url}/values/{quote(key_dir, safe='')}", headers=headers, data=str(direction))
            if wave_height is not None:
                requests.put(f"{base_url}/values/{quote(key_wave, safe='')}", headers=headers, data=str(wave_height))
        except Exception as e:
            print(f"Error storing data in Cloudflare KV: {e}")
        return 0

    store_buoy_data_cached(base_url, headers, buoy, timestamp, wind_speed, direction, wave_height)


def refreshBuoy(buoy='46146', title='Halibut Bank - 46146', container=None,
                forecast_url=None, forecast_title=''):
    draw = container or st
    url = f'https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html?mapID=02&siteID=14305&stationID={buoy}'

    # Live data — short 3-min cache so new observations surface quickly
    res = cached_fetch_url_live(url)
    soup = BeautifulSoup(res.content, 'html.parser')
    table = soup.find('table', class_='table')
    time = soup.find('span', class_='issuedTime').string
    rows = table.tbody.find_all('tr')

    data_wind = data_pressure = data_wave_height = data_airtemp = data_waveperiod = data_watertemp = 'N/A'
    data_wind = rows[0].find_all('td')[0].text.strip()

    def parse_wind_data(wind_text):
        """Parse wind text to extract direction and highest speed."""
        if not isinstance(wind_text, str) or not wind_text.strip():
            return None, 0
        parts = wind_text.strip().split()
        if not parts:
            return None, 0
        direction = parts[0]
        numbers = [int(num) for num in re.findall(r'\d+', wind_text)]
        highest_speed = max(numbers) if numbers else 0
        return direction, highest_speed

    if buoy in ('46146', '46304'):
        data_wave_height = rows[1].find_all('td')[0].text.strip() + 'm'
        data_airtemp = rows[1].find_all('td')[1].text.strip() + '°C'
        data_waveperiod = rows[2].find_all('td')[0].text.strip() + 's'
        data_watertemp = rows[2].find_all('td')[1].text.strip() + '°C'

    draw.subheader('Weather Data for ' + title + ' - ' + buoy)

    winds = re.findall(r'\d+', data_wind)
    highest_wind = int(winds[0]) if winds else 0
    displayWindWarningIfNeeded(highest_wind, container=draw)
    displayStreamlitDateTime(time, draw, label="Issued", source_url=url,
                              source_label='weather.gc.ca buoy')
    draw.text(data_wind)

    waves = re.findall(r"[-+]?\d*\.\d+|\d+", data_wave_height)
    highest_wave = float(waves[0]) if waves else 0.0
    if highest_wave >= 1:
        draw.badge("Wave warning", color="orange")

    # Split wind text ("NW 25 gust 30") into direction + highest speed for display
    parts = data_wind.strip().split() if data_wind and isinstance(data_wind, str) else []
    wind_direction = parts[0] if len(parts) > 0 else "N/A"
    wind_numbers = re.findall(r'\d+', data_wind or '')
    wind_speed_display = f"{max(int(n) for n in wind_numbers)} kts" if wind_numbers else "N/A"

    if data_wave_height == 'N/A':
        col1, col2 = draw.columns(2)
        col1.metric("Wind Speed", wind_speed_display)
        col2.metric("Wind Direction", wind_direction)
        col2.markdown(create_arrow_html(wind_direction, wind_speed_display),
                      unsafe_allow_html=True)
    else:
        r1c1, r1c2, r1c3, r1c4 = draw.columns(4)
        r1c1.metric("Wind Speed", wind_speed_display)
        r1c2.metric("Wind Direction", wind_direction)
        r1c2.markdown(create_arrow_html(wind_direction, wind_speed_display),
                      unsafe_allow_html=True)
        r1c3.metric("Wave Height", data_wave_height)
        r1c4.metric("Air Temp", data_airtemp)

        r2c1, r2c2, r2c3 = draw.columns(3)
        r2c1.metric("Water Temp", data_watertemp)
        r2c2.metric("Wave Period", data_waveperiod)
        r2c3.metric("Pressure", data_pressure)

    direction, wind_speed = parse_wind_data(data_wind)
    wave_val_for_record = highest_wave if data_wave_height != 'N/A' and waves else None

    record_buoy_data_history(buoy, container, wind_speed, direction, wave_val_for_record)
    past_df = plot_merged_wind_chart(container, buoy, forecast_url, forecast_title)
    plot_wave_history_chart(container, past_df, buoy)
    with draw.expander("Map"):
        drawMapWithBuoy(container=st, buoy=buoy)


# ──────────────────────────────────────────────
# Shared sidebar + st.navigation entrypoint
# ──────────────────────────────────────────────

with st.sidebar:
    st.badge(f"v{APP_VERSION}", color="blue")
    st.caption("Auto-refresh every 5 minutes")

display_gonogo_sidebar()

# Build st.Page objects — keep references for page_link usage
_pg_gonogo = st.Page(page_gonogo, title="Go / No-Go", icon="🚤", default=True)
_pg_dashboard = st.Page(page_dashboard, title="Dashboard", icon="📊")
_pg_jericho = st.Page(page_jericho, title="Jericho Wind", icon="💨")
_pg_halibut = st.Page(page_halibut, title="Halibut Bank", icon="🔵")
_pg_english_bay = st.Page(page_english_bay, title="English Bay", icon="🔵")
_pg_atkinson = st.Page(page_atkinson, title="Pt Atkinson", icon="🔵")
_pg_pamrocks = st.Page(page_pamrocks, title="Pam Rocks", icon="🔵")
_pg_whales = st.Page(page_whales, title="Whale boats", icon="🐋")
_pg_whales2 = st.Page(page_whales2, title="Whale boats 2", icon="🐋")
_pg_alex = st.Page(page_alex, title="Alex Location", icon="📍")
_pg_forecast = st.Page(page_forecast, title="Marine Forecast", icon="🌊")
_pg_tides = st.Page(page_tides, title="Tides", icon="🌊")
_pg_beach = st.Page(page_beach, title="Beach", icon="🏖️")
_pg_squamish = st.Page(page_squamish, title="Squamish W", icon="⛰️")
_pg_lionsbay = st.Page(page_lionsbay, title="Lions Bay W", icon="⛰️")
_pg_kiosk = st.Page(page_kiosk, title="Kiosk Mode", icon="📺")

# Store page_link targets so page_gonogo can reference them
PAGE_LINKS = {
    'Dashboard': _pg_dashboard,
    'Marine_Forecast': _pg_forecast,
    'Halibut_Bank': _pg_halibut,
    'English_Bay': _pg_english_bay,
    'Tides': _pg_tides,
}

pages = {
    "Conditions": [_pg_gonogo, _pg_dashboard],
    "Live Data": [_pg_alex, _pg_pamrocks, _pg_whales, _pg_whales2, _pg_jericho, _pg_english_bay, _pg_atkinson, _pg_halibut],
    "Forecast & Tides": [_pg_forecast, _pg_tides, _pg_beach],
    "Regional Weather": [_pg_squamish, _pg_lionsbay],
    "Display": [_pg_kiosk],
}

pg = st.navigation(pages)
pg.run()
