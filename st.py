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
from datetime import datetime
from urllib.parse import quote

from streamlit_autorefresh import st_autorefresh

from utils import cached_fetch_url, prettydate, displayStreamlitDateTime
from fetch_forecast import display_marine_forecast_for_url
from fetch_forecast import display_summary_marine_forecast_for_url
from fetch_beach import display_beach_quality_for_sandy_cove
from fetch_weather import display_weather_info
from fetch_tides import display_point_atkinson_tides
from wind_utils import create_arrow_html

# Auto-refresh every 5 minutes
st_autorefresh(interval=300000, key="data_refresher")


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


def headerbox():
    from fetch_weather import display_clear_skies_html

    URL_forecast_howesound = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400'
    URL_forecast_south_of_nanaimo = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14305'
    URL_forecast_north_of_nanaimo = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14301'

    VANCOUVER_LAT = 49.32
    VANCOUVER_LON = -123.16
    SQUAMISH_LAT = 49.7
    SQUAMISH_LON = -123.16
    LIONSBAY_LAT = 49.45
    LIONSBAY_LON = -123.16

    # Sidebar navigation
    with st.sidebar:
        st.title("Sailing BC")
        page = st.radio("Navigate", [
            "Dashboard",
            "Jericho Wind",
            "Halibut Bank",
            "Pt Atkinson",
            "Pam Rocks",
            "Forecast",
            "Beach",
            "Tides",
            "Squamish",
            "Lions Bay",
        ], label_visibility="collapsed")
        st.divider()
        st.badge("v21", color="blue")
        st.caption("Auto-refresh every 5 minutes")

    draw = st

    if page == "Dashboard":
        try:
            display_weather_info(container=draw, lat=VANCOUVER_LAT, long=VANCOUVER_LON, title="Weather")
        except Exception as e:
            draw.error(f"Failed to load weather: {e}")

        try:
            display_clear_skies_html(container=draw, title="Clear Skies")
        except Exception as e:
            draw.error(f"Failed to load clear skies: {e}")

        try:
            display_summary_marine_forecast_for_url(draw=draw, url=URL_forecast_howesound, title="Howe Sound")
            display_summary_marine_forecast_for_url(draw=draw, url=URL_forecast_south_of_nanaimo, title="South of Nanaimo")
            display_summary_marine_forecast_for_url(draw=draw, url=URL_forecast_north_of_nanaimo, title="North of Nanaimo")
        except Exception as e:
            draw.error(f"Failed to load marine forecast summary: {e}")

        try:
            display_beach_quality_for_sandy_cove(draw=draw, title="Beach water quality Sandy Cove")
        except Exception as e:
            draw.error(f"Failed to load beach quality: {e}")

        try:
            display_point_atkinson_tides(container=draw)
        except Exception as e:
            draw.error(f"Failed to load tides: {e}")

    elif page == "Jericho Wind":
        try:
            parseJerichoWindHistory(container=draw)
        except Exception as e:
            draw.error(f"Failed to load Jericho wind data: {e}")

    elif page == "Halibut Bank":
        try:
            refreshBuoy('46146', 'Halibut Bank', container=draw)
        except Exception as e:
            draw.error(f"Failed to load Halibut Bank buoy: {e}")

    elif page == "Pt Atkinson":
        try:
            refreshBuoy('WSB', 'Point Atkinson', container=draw)
        except Exception as e:
            draw.error(f"Failed to load Point Atkinson buoy: {e}")

    elif page == "Pam Rocks":
        try:
            refreshBuoy('WAS', 'Pam Rocks', container=draw)
        except Exception as e:
            draw.error(f"Failed to load Pam Rocks buoy: {e}")

    elif page == "Forecast":
        region = st.selectbox("Region", ["Howe Sound", "South of Nanaimo", "North of Nanaimo"])
        url_map = {
            "Howe Sound": URL_forecast_howesound,
            "South of Nanaimo": URL_forecast_south_of_nanaimo,
            "North of Nanaimo": URL_forecast_north_of_nanaimo,
        }
        try:
            display_marine_forecast_for_url(draw=draw, url=url_map[region], title=region)
        except Exception as e:
            draw.error(f"Failed to load {region} forecast: {e}")

    elif page == "Beach":
        try:
            display_beach_quality_for_sandy_cove(draw=draw, title="Beach water quality Sandy Cove")
        except Exception as e:
            draw.error(f"Failed to load beach quality: {e}")

    elif page == "Tides":
        try:
            display_point_atkinson_tides(container=draw)
        except Exception as e:
            draw.error(f"Failed to load tides: {e}")

    elif page == "Squamish":
        try:
            display_weather_info(container=draw, lat=SQUAMISH_LAT, long=SQUAMISH_LON, title="Squamish")
        except Exception as e:
            draw.error(f"Failed to load Squamish weather: {e}")

    elif page == "Lions Bay":
        try:
            display_weather_info(container=draw, lat=LIONSBAY_LAT, long=LIONSBAY_LON, title="Lions Bay")
        except Exception as e:
            draw.error(f"Failed to load Lions Bay weather: {e}")


def parseJerichoWindHistory(container=None):
    container.subheader("Jericho Beach Wind History")
    draw = container or st

    url = "https://jsca.bc.ca/main/downld02.txt"
    container.write('https://jsca.bc.ca/services/weather/ -  data from csv ' + url)

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

    displayWindWarningIfNeeded(last_row['Wind Hi Speed'], container=draw)
    displayStreamlitDateTime(last_row['datetime'], draw)

    col1, col2, col3 = draw.columns(3)
    col1.metric(label="Wind Speed", value=last_row['Wind Speed'])
    col2.metric(label="Wind High", value=last_row['Wind Hi Speed'])
    col3.metric(label="Wind Direction", value=last_row['Wind Dir'])
    col2.markdown(create_arrow_html(last_row['Wind Dir'], last_row['Wind Hi Speed']), unsafe_allow_html=True)

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
    draw.plotly_chart(fig, use_container_width=True)
    draw.dataframe(df.tail(24))


def drawMapWithBuoy(container=None, buoy=None):
    latlong = None
    if buoy == '46146':
        latlong = pd.DataFrame({'latitude': [49.34], 'longitude': [-123.72]})
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


def plot_historical_buoy_data(container, buoy_id):
    """Fetch and plot historical wind and wave data from Cloudflare KV."""
    try:
        account_id = st.secrets["cloudflare_account_id"]
        namespace_id_raw = st.secrets["cloudflare_namespace_id"]
        api_token = st.secrets["cloudflare_api_token"]
    except KeyError:
        container.warning("Cloudflare secrets not configured")
        return

    namespace_id = get_resolved_namespace_id(account_id, api_token, namespace_id_raw)

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    try:
        response = cached_kv_list(container, buoy_id, api_token, account_id, namespace_id)
        if response is None:
            container.error("Could not fetch cached list")
            return

        if response.status_code != 200:
            container.error(f"Cloudflare API error: {response.status_code} - {response.text}")
            return

        try:
            data = response.json()
            if not data.get("success", False):
                container.error(f"Cloudflare API error: {data.get('errors')}")
                return

            all_keys = [item["name"] for item in data.get("result", [])]
            three_days_ago = datetime.now(pytz.timezone('America/Vancouver')) - pd.Timedelta(days=3)
            data_points = []
        except requests.exceptions.JSONDecodeError as e:
            container.error(f"Failed to parse Cloudflare response: {str(e)}")
            container.code(response.text[:500])
            return

        for key in all_keys:
            if key.startswith(f"{buoy_id}_wind_"):
                timestamp_str = key.replace(f"{buoy_id}_wind_", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp >= three_days_ago:
                        wind_value, direction, wave_height = get_buoy_observation_from_cf(
                            base_url, headers, buoy_id, timestamp_str)
                        data_points.append({
                            'timestamp': timestamp,
                            'wind_speed': wind_value,
                            'direction': direction,
                            'wave_height': wave_height
                        })
                except Exception as e:
                    print(f"Error processing key {key}: {e}")

        if data_points:
            df = pd.DataFrame(data_points).sort_values('timestamp')

            min_wind = df['wind_speed'].min()
            max_wind = df['wind_speed'].max()
            container.info(f"Min wind speed: {min_wind} knots, Max wind speed: {max_wind} knots")

            direction_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            df['degree'] = df['direction'].map(direction_map).fillna(0)
            df['rotation'] = (180 - df['degree']) % 360

            import plotly.express as px

            fig_wind = px.scatter(df,
                                  x='timestamp', y='wind_speed',
                                  title=f'Wind Speed and Direction Over Last 3 Days - Buoy {buoy_id}',
                                  labels={'wind_speed': 'Wind Speed (knots)', 'timestamp': 'Time'})

            now_van = datetime.now(pytz.timezone('America/Vancouver'))
            fig_wind.update_xaxes(range=[three_days_ago, now_van])
            fig_wind.update_yaxes(range=[0, 40])
            fig_wind.add_hline(y=15, line_dash="dot", line_color="red")

            fig_wind.update_traces(
                marker=dict(
                    symbol='arrow-up', size=14,
                    angle=df['rotation'],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertemplate="<br>".join([
                    "Time: %{x}", "Speed: %{y:.1f} knots", "Direction: %{customdata}"
                ]),
                customdata=df['direction']
            )
            container.plotly_chart(fig_wind, use_container_width=True)

            df_waves = df.dropna(subset=['wave_height']).copy()
            if not df_waves.empty:
                df_waves['wave_height_cm'] = df_waves['wave_height'] * 100
                fig_wave = px.line(df_waves,
                                   x='timestamp', y='wave_height_cm',
                                   title=f'Wave Height Over Last 3 Days - Buoy {buoy_id}',
                                   labels={'wave_height_cm': 'Wave Height (cm)', 'timestamp': 'Time'})
                fig_wave.update_xaxes(range=[three_days_ago, now_van])
                fig_wave.update_yaxes(range=[0, 200])
                fig_wave.add_hline(y=33, line_dash="dot", line_color="green")
                fig_wave.add_hline(y=75, line_dash="dot", line_color="orange")
                fig_wave.add_hline(y=100, line_dash="dot", line_color="red")
                container.plotly_chart(fig_wave, use_container_width=True)
        else:
            container.warning(f"No data available for buoy {buoy_id} in the selected period")

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        container.error("Could not load historical data")


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


def refreshBuoy(buoy='46146', title='Halibut Bank - 46146', container=None):
    draw = container or st
    url = f'https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html?mapID=02&siteID=14305&stationID={buoy}'

    res = cached_fetch_url(url)
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

    if buoy == '46146':
        data_wave_height = rows[1].find_all('td')[0].text.strip() + 'm'
        data_airtemp = rows[1].find_all('td')[1].text.strip() + '°C'
        data_waveperiod = rows[2].find_all('td')[0].text.strip() + 's'
        data_watertemp = rows[2].find_all('td')[1].text.strip() + '°C'

    draw.subheader('Weather Data for ' + title + ' - ' + buoy)
    draw.write(url)

    winds = re.findall(r'\d+', data_wind)
    highest_wind = int(winds[0]) if winds else 0
    displayWindWarningIfNeeded(highest_wind, container=draw)
    displayStreamlitDateTime(time, draw)
    draw.text(data_wind)

    waves = re.findall(r"[-+]?\d*\.\d+|\d+", data_wave_height)
    highest_wave = float(waves[0]) if waves else 0.0
    if highest_wave >= 1:
        draw.badge("Wave warning", color="orange")

    if data_wave_height == 'N/A':
        draw.metric("Wind", data_wind)
    else:
        col1, col2, col3 = draw.columns(3)
        col1.metric("Wind", data_wind)

        parts = data_wind.strip().split() if data_wind and isinstance(data_wind, str) else []
        wind_direction = parts[0] if len(parts) > 0 else "N/A"
        wind_speed_str = parts[1] if len(parts) > 1 else "0"

        col1.markdown(create_arrow_html(wind_direction, wind_speed_str), unsafe_allow_html=True)
        col2.metric("Wave Height", data_wave_height)
        col3.metric("Air Temp", data_airtemp)
        col1.metric("Water Temp", data_watertemp)
        col2.metric("Wave Period", data_waveperiod)
        col3.metric("Pressure", data_pressure)

    direction, wind_speed = parse_wind_data(data_wind)
    wave_val_for_record = highest_wave if data_wave_height != 'N/A' and waves else None

    record_buoy_data_history(buoy, container, wind_speed, direction, wave_val_for_record)
    plot_historical_buoy_data(container, buoy)
    drawMapWithBuoy(container=draw, buoy=buoy)


# Initialize application
headerbox()
