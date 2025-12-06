#  > pip install -r requirements.txt
#  > python -m streamlit run st.py
# http://localhost:8501/
# http://python-app-sailing-bc-nckqtfynerhhf26ujtt5u6
from http.client import responses

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pytz
import numpy as np
import pandas as pd

from datetime import datetime
import pytz

from timeago import format as timeago_format

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# Call the component to set up the auto-refresh
# interval is in milliseconds, so 15 minutes = 15 * 60 * 1000 = 900000
st_autorefresh(interval=300000, key="data_refresher")


def cached_fetch_url(url):
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return response


def prettydate(d):
    now_vancouver = datetime.now(pytz.timezone('America/Vancouver'))
    return timeago_format(d, now_vancouver)

from dateutil import parser
from dateutil.tz import gettz

def displayStreamlitDateTime(datetime, container=None):
    """ accepts a string or datetime object, tries its best at recognizing/parsing it, and displays it in Streamlit format."""
    draw = container
    if isinstance(datetime,str):
        tzinfos = {"PDT": gettz("America/Vancouver")    ,
                   "PST": gettz("America/Vancouver"),
        }
        print("Parsing time ", datetime)
        datetime_van = parser.parse(datetime, tzinfos=tzinfos)
        datetime_van = datetime_van

    else:
        datetime_van = datetime.replace(tzinfo=gettz('America/Vancouver'))

    draw.title(prettydate(datetime_van))
    draw.text(datetime_van)

# Selector
from fetch_forecast import display_marine_forecast_for_url
from fetch_forecast import display_beach_quality_for_sandy_cove
from fetch_forecast import display_weather_info
from fetch_forecast import display_summary_marine_forecast_for_url


def headerbox():

    tab10, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab1 = st.tabs([
        "Weather",
        "Jericho",
        "Halibut Bank",
        "Pt Atkinson",
        "Pam Rocks",
        "Howe Sound",
        "S Nanaimo",
        "N Nanaimo",
        "Beach",
        'Tides'
    ])
    # Example coordinates for West Vancouver
    VANCOUVER_LAT = 49.32
    VANCOUVER_LON = -123.16

    URL_forecast_howesound = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400'
    URL_forecast_south_of_nanaimo = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14305'
    URL_forecast_north_of_nanaimo = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14301'

#Main tab
    display_weather_info(container=tab10, lat=VANCOUVER_LAT, long=VANCOUVER_LON, title="Weather")
    display_summary_marine_forecast_for_url(draw=tab10, url=URL_forecast_howesound , title="Howe Sound")
    display_summary_marine_forecast_for_url(draw=tab10, url=URL_forecast_south_of_nanaimo , title="Howe Sound")
    display_summary_marine_forecast_for_url(draw=tab10, url=URL_forecast_north_of_nanaimo , title="Howe Sound")
    display_beach_quality_for_sandy_cove(draw=tab10, title="🏖️ Beach water quality Sandy Cove")
    display_point_atkinson_tides(container=tab10)

    parseJerichoWindHistory(container=tab2)
    refreshBuoy('46146','Halibut Bank', container=tab3)
    refreshBuoy('WSB', 'Point Atkinson', container=tab4)
    refreshBuoy('WAS', 'Pam Rocks', container=tab5)


    display_marine_forecast_for_url(draw=tab6, url=URL_forecast_howesound, title="Howe Sound")
    display_marine_forecast_for_url(draw=tab7, url=URL_forecast_south_of_nanaimo, title="South of Nanaimo")
    display_marine_forecast_for_url(draw=tab8, url=URL_forecast_north_of_nanaimo, title="North of Nanaimo")
    display_beach_quality_for_sandy_cove(draw=tab9, title="🏖️ Beach water quality Sandy Cove")


    display_point_atkinson_tides(container=tab1)
    st.badge("v20", color="blue")
    st.write("This application will auto-refresh every 15 minutes.")


def displayWindWarningIfNeeded(wind_speed, container=None):
    """ above 9 knots """
    if container:
        draw = container
    else:
        draw = st

    try:
        # Convert to numeric if it's a DataFrame/Series
        if isinstance(wind_speed, (pd.DataFrame, pd.Series)):
            wind_speed = pd.to_numeric(wind_speed, errors='coerce')
            if isinstance(wind_speed, pd.Series):
                wind_speed = wind_speed.iloc[0]  # Get the first value if it's a Series

        # Convert to float if it's a string
        if isinstance(wind_speed, str):
            wind_speed = float(wind_speed)

        warning_wind = (wind_speed > 9)
        if warning_wind:
            draw.badge("Wind warning", color="orange")
    except (ValueError, TypeError) as e:
        print(f"Error processing wind speed: {e}")
        return


from fetch_tides import display_point_atkinson_tides


def parseJerichoWindHistory(container = None):

    container.subheader("Jericho Beach Wind History")

    if container:
        draw = container
    else:
        draw = st
    # https://jsca.bc.ca/main/downld02.txt
    url = "https://jsca.bc.ca/main/downld02.txt"

    container.write('https://jsca.bc.ca/services/weather/ -  data from csv '+ url)

    res = cached_fetch_url(url)

    # stupid csv file as 2 first rows as column headers with columns 0,1,13,14 first line missing, fix this
    csv_raw = res.content.decode('utf-8')
    lines = csv_raw.splitlines()
    csv_fixed = '\n'.join(lines[3:]) # drop the 1st 3 rows that is '--------------' as it messes with separator
   # csv_fixed = '\n'.join(lines[:1] + lines[2:])  # drop the 3rd (now 2nd) row that is '--------------' as it messes with separator

   #
    #st.write(csv_fixed)
    #print(csv_fixed)
    import io

    df = pd.read_csv(io.StringIO(csv_fixed) ,header=None, delim_whitespace=True)

    print(f"Number of columns: {len(df.columns)}")
    df.columns = ['Date',
                  'Time',
                  'Temp Out',
                  'Temp Hi',
                  'Temp Low',
                  'Hum Out',
                  'Dew Pt.',
                  'Wind Speed',
                  'Wind Dir',
                  'Wind Run',
                  'Wind Hi Speed',
                  'Wind Hi Dir',
                  'Wind Chill',
                  'Heat Index',
                  'THW Index',
                  'Bar',
                  'Rain',
                  'Rain Rate',
                  'Heat D-D',
                  'Cool D-D',
                  'In Temp',
                  'In Hum',
                  'In Dew',
                  'In Heat',
                  'In EMC',
                  'In Air Density',
                  'Wind Samp',
                  'Wind TX',
                  'IS Recept.',
                  'Arc Int',
                  ]  # adjust the number of columns as needed

# combine 1st and 2nd column date and time, drop first 2 columns
    # Combine date and time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1], utc=False)

    # If you want to drop the original date and time columns
    df = df.drop(df.columns[[0, 1]], axis=1)
    # If you want to move the datetime column to the front
    cols = df.columns.tolist()
    df = df[['datetime'] + cols[:-1]]  # excluding the last column since it's datetime

    displayWindWarningIfNeeded(df.iloc[-1, 9], container=draw)

#display time
    datetime_last_measurement = df.iloc[-1, 0]  # -1 for last row, 1 for second column (0-based index)
    displayStreamlitDateTime(datetime_last_measurement, draw)

    # display last values
    temp_out = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)
    wind_speed = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)
    temp_out = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)

    col1, col2, col3 = draw.columns(3)
    col1.metric(label="Wind Speed",     value=df.iloc[-1, 6])
    col2.metric(label="Wind High",      value=df.iloc[-1, 9])
    col3.metric(label="Wind Direction",      value=df.iloc[-1, 7])
    from fetch_forecast import create_arrow_html
    col2.markdown(create_arrow_html(df.iloc[-1, 7],df.iloc[-1, 9]), unsafe_allow_html=True)

    col1, col2, col3 = draw.columns(3)
    col1.metric(label="Bar",            value=df.iloc[-1, 14])
    col2.metric(label="Rain",           value=df.iloc[-1, 15])
    col3.metric(label="Temperature",    value=df.iloc[-1, 1])

    # display graph of last 12 hours (24 entries)
    import plotly.express as px

    df_tail = df.tail(24) 
    fig = px.line(df_tail, x='datetime', y=['Wind Speed', 'Wind Hi Speed'],
                  title="Jericho Wind History (Last 12 Hours)")
    fig.update_yaxes(range=[0, 30], title="Speed (knots)")
    fig.add_hline(y=15, line_dash="dot", line_color="red")

    draw.plotly_chart(fig, use_container_width=True)

    draw.dataframe(df.tail(24))


def headerboxMenuDeprecated():
    with st.popover("Select a Buoy"):
        c = st.container(border = True)

        buoy = c.selectbox('Choose a Buoy', ['46146', 'WSB', 'WAS'])
        c.write('46146 = Halibut Bank')
        c.write('WSB = Point Atkinson')
        c.write('WAS = Pam Rocks')
        title = 'N/A'

        if buoy == '46146':
            title = 'Halibut Bank'
        if buoy == 'WSB':
            title = 'Point Atkinson'
        if buoy == 'WAS':
            title = 'Pam Rocks'

    refreshBuoy(buoy = buoy, title = title)


def drawMapWithBuoy(container = None, buoy = None):
#Halibut Bank - 46146
    latlong =  None
    if buoy == '46146':
        latlong = pd.DataFrame({
            'latitude': [49.34],
            'longitude': [-123.72]
        })
    if buoy == 'WSB':
        latlong = pd.DataFrame({
            'latitude': [49.330],
            'longitude': [-123.2646]
        })
    if buoy == 'WAS':
        latlong = pd.DataFrame({
            'latitude': [49.49],
            'longitude': [-123.3]
        })
    ## Create a map with the data
    container.map(latlong, zoom=10)


from urllib.parse import quote

@st.cache_data(ttl=144600)
def get_buoy_observation_from_cf(base_url, headers, buoy_id, timestamp_str):
    # Cloudflare keys must be URL-encoded in the path
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
    # Check if the provided namespace_id is actually the name of a namespace
    if "storage/kv/namespaces/" not in name_or_id:
        # Attempt to resolve it as a namespace name
        headers = {"Authorization": f"Bearer {api_token}"}
        url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces"
        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            
            namespaces = response.json().get("result", [])
            
            for ns in namespaces:
                if ns.get("title") == name_or_id:
                    return ns.get("id")
        
        except requests.exceptions.RequestException as e:
            print(f"Error resolving namespace ID: {e}")
            return name_or_id # Fallback to original if resolution fails or error occurs
    return name_or_id # Fallback to original if resolution fails or error occurs

@st.cache_data(ttl=600)
def cached_kv_list(_container, buoy_id, api_token, account_id, namespace_id):
    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    headers = {"Authorization": f"Bearer {api_token}"}
    try:
        # Get keys with prefix for this buoy
        # We use _wind_ keys to discover available timestamps
        params = {
            'prefix': f"{buoy_id}_wind_",
            'limit': 1000
        }

        response = requests.get(f"{base_url}/keys", params=params, headers=headers)
        return response

    except Exception as e:
        print(f"Error fetching historical data: {e}")
        _container.error("Could not load historical data")

    return None


def plot_historical_buoy_data(container, buoy_id):
    """Fetch and plot historical wind and wave data from Cloudflare KV for a specific buoy"""
    try:
        account_id = st.secrets["cloudflare_account_id"]
        namespace_id_raw = st.secrets["cloudflare_namespace_id"]
        api_token = st.secrets["cloudflare_api_token"]
    except KeyError:
        container.warning("Cloudflare secrets not configured (cloudflare_account_id, cloudflare_namespace_id, cloudflare_api_token)")
        return

    # Resolve namespace ID if the secret contains the name
    namespace_id = get_resolved_namespace_id(account_id, api_token, namespace_id_raw)

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    # Fetch all keys from last 3 days
    try:
        # Get keys with prefix for this buoy
        # We use _wind_ keys to discover available timestamps

        response = cached_kv_list(container, buoy_id, api_token, account_id, namespace_id)
        if response is None:
            container.error(f"Could not fetch cached list")

        if response.status_code != 200:
            container.error(f"Cloudflare API error: {response.status_code} - {response.text}")
            return

        try:
            data = response.json()
            if not data.get("success", False):
                container.error(f"Cloudflare API error: {data.get('errors')}")
                return
            
            all_keys = [item["name"] for item in data.get("result", [])]

            # Filter keys for wind data from last 3 days
            three_days_ago = datetime.now(pytz.timezone('America/Vancouver')) - pd.Timedelta(days=3)

            data_points = []
            print (f"Cloudflare keys found = {len(all_keys)}")
        except requests.exceptions.JSONDecodeError as e:
            container.error(f"Failed to parse Cloudflare response: {str(e)}")
            container.write("Raw response:")
            container.code(response.text[:500])
            return

        for key in all_keys:

            if key.startswith(f"{buoy_id}_wind_"):
                timestamp_str = key.replace(f"{buoy_id}_wind_", "")
                try:
                    timestamp = datetime.fromisoformat(timestamp_str)
                    if timestamp >= three_days_ago:
                        # Fetch all values
                        wind_value, direction, wave_height = get_buoy_observation_from_cf(base_url, headers, buoy_id, timestamp_str)
                        
                        data_points.append({
                            'timestamp': timestamp,
                            'wind_speed': wind_value,
                            'direction': direction,
                            'wave_height': wave_height
                        })
                except Exception as e:
                    print(f"Error processing key {key}: {e}")

        if data_points:
            # Create DataFrame
            df = pd.DataFrame(data_points)
            df = df.sort_values('timestamp')

            # --- Plot Wind ---
            min_wind = df['wind_speed'].min()
            max_wind = df['wind_speed'].max()
            container.info(f"Min wind speed: {min_wind} knots, Max wind speed: {max_wind} knots")

            # Map directions to degrees for Plotly arrows
            direction_map = {
                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
            }
            df['degree'] = df['direction'].map(direction_map).fillna(0)
            # Plotly uses CCW rotation from Up.
            # We want arrows to point with the wind.
            # (180 - degree) converts Meteo (CW from N) to Plotly (CCW from Up)
            df['rotation'] = (180 - df['degree']) % 360

            import plotly.express as px
            fig_wind = px.scatter(df,
                                x='timestamp',
                                y='wind_speed',
                                title=f'Wind Speed and Direction Over Last 3 Days - Buoy {buoy_id}',
                                labels={'wind_speed': 'Wind Speed (knots)',
                                        'timestamp': 'Time'})

            # Set x-axis range to show last 3 days even if data is sparse
            now_van = datetime.now(pytz.timezone('America/Vancouver'))
            fig_wind.update_xaxes(range=[three_days_ago, now_van])
            fig_wind.update_yaxes(range=[0, 40])
            fig_wind.add_hline(y=15, line_dash="dot", line_color="red")

            # Add direction information to hover text and use arrow markers
            fig_wind.update_traces(
                marker=dict(
                    symbol='arrow-up',
                    size=14,
                    angle=df['rotation'],
                    line=dict(width=1, color='DarkSlateGrey')
                ),
                hovertemplate="<br>".join([
                    "Time: %{x}",
                    "Speed: %{y:.1f} knots",
                    "Direction: %{customdata}"
                ]),
                customdata=df['direction']
            )

            container.plotly_chart(fig_wind, use_container_width=True)

            # --- Plot Waves ---
            # Filter out rows where wave_height is None for the plot
            df_waves = df.dropna(subset=['wave_height']).copy()
            if not df_waves.empty:
                # Convert meters to cm
                df_waves['wave_height_cm'] = df_waves['wave_height'] * 100
                
                fig_wave = px.line(df_waves,
                                    x='timestamp',
                                    y='wave_height_cm',
                                    title=f'Wave Height Over Last 3 Days - Buoy {buoy_id}',
                                    labels={'wave_height_cm': 'Wave Height (cm)',
                                            'timestamp': 'Time'})
                
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

    # Store in Cloudflare KV
    current_time = datetime.now(pytz.timezone('America/Vancouver'))
    # Truncate minutes to nearest 30 minutes
    current_time = current_time.replace(minute=current_time.minute // 30 * 30, second=0, microsecond=0)
    timestamp = current_time.isoformat(timespec='minutes')

    try:
        account_id = st.secrets["cloudflare_account_id"]
        namespace_id_raw = st.secrets["cloudflare_namespace_id"]
        api_token = st.secrets["cloudflare_api_token"]
    except KeyError:
        print("Cloudflare secrets missing")
        return

    # Resolve namespace ID if the secret contains the name
    namespace_id = get_resolved_namespace_id(account_id, api_token, namespace_id_raw)

    base_url = f"https://api.cloudflare.com/client/v4/accounts/{account_id}/storage/kv/namespaces/{namespace_id}"
    headers = {"Authorization": f"Bearer {api_token}"}

    @st.cache_data(ttl=1800)
    def store_buoy_data_cached(base_url, headers, buoy, timestamp, wind_speed, direction, wave_height):
        try:
            # Store wind data - keys must be URL encoded for the API endpoint
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


def refreshBuoy(buoy = '46146', title = 'Halibut Bank - 46146', container = None):
    if container:
        draw = container
    else:
        draw = st
    url = f'https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html?mapID=02&siteID=14305&stationID={buoy}'

    res = cached_fetch_url(url)

    soup = BeautifulSoup(res.content, 'html.parser')

    tables = soup.find_all('table')
    table = soup.find('table', class_='table')
    # parsing
    time = soup.find('span', class_='issuedTime').string

    # --- debug ---
    print('TABLE:')
    #(table.tbody.find_all('tr'))
    for row in table.tbody.find_all('tr'):
        columns = row.find_all('td')
        # print('COLUMNS LENGTH:', len(columns))

    # --- debug end ---

    rows = table.tbody.find_all('tr')
    print('TABLE LENGTH:', len(rows))

    data_wind = data_pressure = data_wave_height = data_airtemp = data_waveperiod = data_watertemp = 'N/A'

    data_wind = rows[0].find_all('td')[0].text.strip()

    # parse wind speed from wind data. take the gust value or highest value. also extract direction. Here are some examples of potential values
    # Here are some examples of potential values N19 gusts 24          WNW3         E7gusts10
    def parse_wind_data(wind_text):
        """
        Parse wind data text to extract direction and highest speed value.
        Returns tuple of (direction, highest_speed)

        Examples:
        'N 19 gusts 24' -> ('N', 24)
        'WNW 3' -> ('WNW', 3)
        'E 7 gusts 10' -> ('E', 10)
        """
        if not isinstance(wind_text, str) or not wind_text.strip():
            return None, 0

        # Split the string into parts
        parts = wind_text.strip().split()
        if not parts:
            return None, 0

        # First part is typically the direction
        direction = parts[0]

        # Extract all numbers from the string
        import re
        numbers = [int(num) for num in re.findall(r'\d+', wind_text)]

        # Get the highest number (either gust or regular speed)
        highest_speed = max(numbers) if numbers else 0

        return direction, highest_speed


    if buoy == '46146':
        data_wave_height = rows[1].find_all('td')[0].text.strip() + 'm'
        data_airtemp = rows[1].find_all('td')[1].text.strip() + '°C'

        data_waveperiod = rows[2].find_all('td')[0].text.strip() + 's'
        data_watertemp = rows[2].find_all('td')[1].text.strip() + '°C'

    if buoy == 'WSB': # Point Atkinson
        None

    if buoy == 'WAS': # Pam Rocks
        None

    # data_wave_height = row.parent.find_all('td')[1]  # last cell in the row
        #data_airtemp = row.parent.find_all('td')[3]  # last cell in the row
        #data_waveperiod = row.parent.find_all('td')[2]  # last cell in the row
        #data_watertemp = row.parent.find_all('td')[5]  # last cell in the row

    draw.subheader('Weather Data for '+ title + ' - ' + buoy)

    draw.write(url)

    import re
    winds = re.findall(r'\d+', data_wind)
    highest_wind = 0
    if winds:
        highest_wind = int(winds[0])
    displayWindWarningIfNeeded(highest_wind, container=draw)
    displayStreamlitDateTime(time, draw)

    draw.text(data_wind )

    # Improved wave parsing
    waves = re.findall(r"[-+]?\d*\.\d+|\d+", data_wave_height)
    highest_wave = 0.0
    if waves:
        highest_wave = float(waves[0])
    warning_wave = (highest_wave>=1)

    if (warning_wave):
        draw.badge("Wave warning", color="orange")

    if data_wave_height == 'N/A':
        draw.metric("Wind", data_wind )

    else:
        col1, col2, col3 = draw.columns(3)
        col1.metric("Wind", data_wind )
        # Safe parsing of wind data
        try:
            parts = data_wind.strip().split() if data_wind and isinstance(data_wind, str) else []
            wind_direction = parts[0] if len(parts) > 0 else "N/A"
            wind_speed = parts[1] if len(parts) > 1 else "0"
        except (AttributeError, IndexError) as e:
            print(f"Error parsing wind data: {e}")
            wind_direction = "N/A"
            wind_speed = "0"
        from fetch_forecast import create_arrow_html
        col1.markdown(create_arrow_html(wind_direction, wind_speed), unsafe_allow_html=True)

        col2.metric("Wave Height", data_wave_height)
        col3.metric("Air Temp", data_airtemp)
        col1.metric("Water Temp", data_watertemp)
        col2.metric("Wave Period", data_waveperiod)
        col3.metric("Pressure", data_pressure)


    direction, wind_speed = parse_wind_data(data_wind)
    
    # Parse wave height for recording (use highest_wave from above or None if N/A)
    # data_wave_height string e.g. "0.6m" or "N/A"
    wave_val_for_record = highest_wave if data_wave_height != 'N/A' and waves else None
    
    record_buoy_data_history(buoy, container, wind_speed, direction, wave_val_for_record)
    # Fetch and plot historical data
    plot_historical_buoy_data(container, buoy)

    # st.code(soup) # debug HTML
    drawMapWithBuoy(container=draw, buoy=buoy)



# initialize application
headerbox()
