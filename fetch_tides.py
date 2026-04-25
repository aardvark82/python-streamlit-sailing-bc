import shutil
import os
import re
import io
import subprocess
import glob as glob_module

import streamlit as st
import requests
import numpy as np
import pandas as pd
import pytz
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from scipy.interpolate import CubicSpline

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select

# --- Tide data source configuration ---
# Enable exactly one primary source. Selenium is the default and most reliable.
USE_SELENIUM = True          # Fetch tide CSV via headless Chrome (recommended for Streamlit Cloud)
USE_BEAUTIFULSOUP = False    # Scrape tide table HTML directly (no JS needed, but less data)
USE_CHAT_GPT = False         # Send HTML to GPT-4o for parsing (costs API credits)
USE_STORMGLASS = False       # Use Stormglass.io tide API (free tier limited)

CANADA_GOVERNMENT_TIDE_POINT_ATKINSON = "https://www.tides.gc.ca/en/stations/07795"

download_dir = os.path.abspath("temp_downloads")


def cached_fetch_url(url):
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return response


@st.cache_data(ttl=1800)
def beautifulSoupFetchTidesForURL(url):
    """Scrape tide extremes table from tides.gc.ca using BeautifulSoup."""
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, "html.parser")

    def classify_tide(height):
        return "high" if float(height) > 2.0 else "low"

    def parse_time(date_str, time_str):
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        dt_utc = dt + timedelta(hours=7)
        return dt_utc.isoformat() + "+00:00"

    data = {"data": []}
    tide_sections = soup.select("div.tide-table")

    for section in tide_sections:
        date_header = section.find_previous("h3")
        if not date_header:
            continue
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_header.text)
        if not date_match:
            continue
        date = date_match.group(0)

        rows = section.select("table tbody tr")
        for row in rows:
            cols = row.find_all("td")
            if len(cols) < 2:
                continue
            time_pdt = cols[0].text.strip()
            height_m = cols[1].text.strip()
            try:
                height_val = float(height_m)
            except ValueError:
                continue
            tide_type = classify_tide(height_val)
            timestamp = parse_time(date, time_pdt)
            data["data"].append({
                "height": height_val,
                "time": timestamp,
                "type": tide_type
            })

    return data


@st.cache_resource(show_spinner=False)
def get_chromium_version() -> str:
    try:
        result = subprocess.run(['chromium', '--version'], capture_output=True, text=True)
        return result.stdout.split()[1]
    except Exception as e:
        return str(e)


@st.cache_resource(show_spinner=False)
def get_chromedriver_path() -> str:
    return shutil.which('chromedriver')


@st.cache_resource(show_spinner=False)
def get_chromedriver_version() -> str:
    try:
        result = subprocess.run(['chromedriver', '--version'], capture_output=True, text=True)
        return result.stdout.split()[1]
    except Exception as e:
        return str(e)


@st.cache_resource(show_spinner=False)
def get_webdriver_options() -> Options:
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--disable-features=NetworkService")
    options.add_argument("--window-size=1920x1080")
    options.add_argument("--disable-features=VizDisplayCompositor")
    options.add_argument('--ignore-certificate-errors')
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    os.makedirs(download_dir, exist_ok=True)
    options.add_experimental_option("prefs", {
        "download.default_directory": download_dir,
        "download.prompt_for_download": False,
        "download.directory_upgrade": True,
        "safebrowsing.enabled": True
    })

    return options


def get_webdriver_service() -> Service:
    return Service(executable_path=get_chromedriver_path())


def seleniumGetTidesFromURL(url):
    """Fetch tide CSV data from tides.gc.ca using Selenium headless Chrome."""
    import time as time_module

    options = get_webdriver_options()
    service = get_webdriver_service()
    with webdriver.Chrome(options=options, service=service) as driver:
        try:
            url = "https://www.tides.gc.ca/en/stations/07795"
            driver.get(url)
            time_module.sleep(1)

            select_element = driver.find_element(By.ID, "export-select")
            select = Select(select_element)
            select.select_by_value("Predictions")

            export_button = driver.find_element(By.ID, "export_button")
            export_button.click()
            time_module.sleep(3)

            files = glob_module.glob(os.path.join(download_dir, "*.csv"))
            if not files:
                raise Exception("No CSV file was downloaded")

            latest_file = max(files, key=os.path.getctime)
            with open(latest_file, 'r') as file:
                csv_content = file.read()

            return csv_content

        except Exception as e:
            print(f"Error: {e}")

        finally:
            if driver:
                driver.quit()

    return None


@st.cache_data(ttl=1800)
def openAIFetchTidesForURL(url):
    """Send tide HTML to GPT-4o for parsing into structured JSON."""
    openai_api_key = st.secrets["OpenAI_key"]
    if openai_api_key is None:
        raise ValueError("OpenAI API key is not set in environment variables.")

    chat_gpt_msg = '''
    Return just the JSON

    https://www.tides.gc.ca/en/stations/07795. The content of the URL HTML is pasted below.
    Return it in a JSON format that copies the following structure but use the new values from the URL):
    data = {'data': [{'height': 0.5114702042385154, 'time': '2025-05-02T11:16:00+00:00', 'type': 'low'},
                     {'height': 0.8861352612764213, 'time': '2025-05-02T15:04:00+00:00', 'type': 'high'},
                     {'height': -2.339105387160864, 'time': '2025-05-02T22:49:00+00:00', 'type': 'low'},
                     {'height': 1.5511464556228052, 'time': '2025-05-03T06:52:00+00:00', 'type': 'high'},
                     {'height': 0.4471043324619499, 'time': '2025-05-03T12:40:00+00:00', 'type': 'low'},
                     {'height': 0.6249055747798654, 'time': '2025-05-03T15:55:00+00:00', 'type': 'high'},
                     {'height': -2.0247559153700725, 'time': '2025-05-03T23:44:00+00:00', 'type': 'low'},
                     {'height': 1.4711212002069232, 'time': '2025-05-04T07:51:00+00:00', 'type': 'high'},
                     {'height': 0.2401116280506163, 'time': '2025-05-04T14:23:00+00:00', 'type': 'low'},
                     {'height': 0.3448168313068383, 'time': '2025-05-04T17:10:00+00:00', 'type': 'high'},
                     {'height': -1.6944775406122297, 'time': '2025-05-05T00:43:00+00:00', 'type': 'low'},
                     {'height': 1.4152651862548065, 'time': '2025-05-05T08:44:00+00:00', 'type': 'high'}]}

                     '''

    def beautifulSoupFetchTidesSectionForChatGPTForURL(url):
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            response = requests.get(url, headers=headers, timeout=25)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            hourly_table = soup.find('table', class_='hourlytable')
            return str(hourly_table) if hourly_table else "Predictions section not found"
        except requests.RequestException as e:
            return f"Error fetching data: {str(e)}"

    html = beautifulSoupFetchTidesSectionForChatGPTForURL(url)
    chat_gpt_msg = chat_gpt_msg + "This is the HTML " + html

    url_api = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    data = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are an expert meteorologist."},
            {"role": "user", "content": chat_gpt_msg}
        ]
    }

    response = requests.post(url_api, headers=headers, json=data)

    if response.status_code == 200:
        return response
    elif response.status_code == 429:
        return response
    else:
        print("Error:", response.status_code, response.text)
        return response


@st.cache_data(ttl=14400)
def fetch_tide_extremes_selenium():
    """Fetch tide extremes via Selenium and return list of dicts with Height/Time/type.
    Cached for 4 hours since tide predictions are precomputed and don't change often.
    Returns None on failure."""
    try:
        csv_text = seleniumGetTidesFromURL('https://www.tides.gc.ca/en/stations/07795')
        if not csv_text:
            return None

        # Same subsampling as display_point_atkinson_tides to keep extrema detection tractable
        csv_lines = csv_text.splitlines()
        csv_subsampled = '\n'.join(csv_lines[::20])
        csv_lines2 = csv_subsampled.splitlines()
        halfway_point = len(csv_lines2) // 3
        csv_half_subsampled = '\n'.join(csv_lines2[:halfway_point])

        # processCSVResponseToJSONSelenium requires a container for error reporting.
        # Use a silent no-op container so gonogo callers don't get unexpected UI output.
        class _Silent:
            def error(self, *a, **kw):
                pass
        return processCSVResponseToJSONSelenium(_Silent(), csv_half_subsampled)
    except Exception as e:
        print(f"fetch_tide_extremes_selenium error: {e}")
        return None


@st.cache_data(ttl=14400)
def stormglassFetchTidesPointAtkinson():
    """Fetch tide extremes from Stormglass API for Point Atkinson."""
    try:
        lat = 49.3304
        lon = -123.2646

        # Stormglass API key from secrets (never hardcode)
        api_key = st.secrets["stormglass_key"]

        base_url = "https://api.stormglass.io/v2/tide/extremes/point"
        vancouver_tz = pytz.timezone('America/Vancouver')
        now = datetime.now(vancouver_tz)
        start_date = now - timedelta(days=1)
        end_date = now + timedelta(days=2)

        params = {
            'lat': lat,
            'lng': lon,
            'start': start_date.strftime('%Y-%m-%d'),
            'end': end_date.strftime('%Y-%m-%d')
        }
        headers = {'Authorization': api_key}

        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        return response.json()

    except Exception as e:
        print(f"Error fetching tide data: {e}")
        return None


def process_tide_data(data, container=None, use_chat_gpt=False):
    """Parse tide data from various sources into a standard DataFrame."""
    predictions = []

    if 'data' not in data:
        predictions = data
    else:
        for prediction in data['data']:
            dt = pd.to_datetime(prediction['time'])
            dt = dt.tz_convert('America/Vancouver')
            predictions.append({
                'Time (PDT)& Date': dt,
                'Height': float(prediction['height'])
            })

    if USE_STORMGLASS:
        for i in range(len(predictions)):
            predictions[i]['Height'] = predictions[i]['Height'] + 2.64
            predictions[i]['Time (PDT)& Date'] = predictions[i]['Time (PDT)& Date'] + pd.Timedelta(hours=0, minutes=18)

    tide_df = pd.DataFrame(predictions)
    tide_df = tide_df.sort_values('Time (PDT)& Date', ignore_index=True)
    return tide_df


def display_tide_table_text(tide_df, container=None):
    draw = container or st
    draw.markdown("---")

    if not tide_df.empty:
        display_df = tide_df.copy()
        display_df['Time'] = display_df['datetime'].dt.strftime('%I:%M %p')
        display_df['Date'] = display_df['datetime'].dt.strftime('%A, %b %d')
        display_df['Height (m)'] = display_df['Height'].round(2)

        table_df = display_df[['Date', 'Time', 'Height (m)']].copy()
        styled_df = table_df.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': '#e1e4e8'
        }).hide(axis='index')

        draw.dataframe(styled_df, width='stretch')
        draw.markdown("---")


def parse_tide_datetime(time_str):
    """Parse datetime string from tide data."""
    try:
        dt = pd.to_datetime(time_str)
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')
        vancouver_tz = pytz.timezone('America/Vancouver')
        return dt.tz_convert(vancouver_tz)
    except Exception as e:
        print(f"Error parsing datetime: {e}")
        return pd.NaT


def create_smooth_tides(df):
    """Create a smooth tide curve via cubic spline interpolation."""
    times = df['datetime']
    base_time = times.iloc[0]

    x = [(t.timestamp() - base_time.timestamp()) / 3600 for t in times]
    y = df['Height'].values

    x_smooth = np.linspace(min(x), max(x), 200)
    cs = CubicSpline(x, y, bc_type='natural')
    y_smooth = cs(x_smooth)

    times_smooth = pd.Series([
        base_time + pd.Timedelta(seconds=h * 3600)
        for h in x_smooth
    ])

    return pd.DataFrame({
        'datetime': times_smooth,
        'Height': y_smooth
    })


def extract_meters(height_str):
    try:
        return float(height_str.split('m')[0].strip())
    except (ValueError, AttributeError):
        return None


def create_natural_tide_chart(tide_df, container=None):
    """Create an interactive Plotly tide chart with interpolated curve."""
    import plotly.graph_objects as go

    draw = container or st

    if USE_CHAT_GPT:
        draw.badge("From ChatGPT")
    elif USE_SELENIUM:
        draw.badge("From Selenium")
    elif USE_STORMGLASS:
        draw.badge("From Stormglass.io")

    tide_df = tide_df.rename(columns={'Time (PDT)& Date': 'datetime'})
    tide_df['datetime'] = tide_df['datetime'].apply(parse_tide_datetime)
    tide_df['Height'] = tide_df['Height'].astype(str).apply(extract_meters)

    if tide_df['Height'].isnull().all():
        draw.error("No valid height data available")
        return

    tide_df = tide_df.dropna(subset=['Height', 'datetime'])

    if len(tide_df) < 2:
        draw.error("Not enough valid tide data points for interpolation")
        return

    if tide_df['Height'].isnull().any():
        tide_df['Height'] = tide_df['Height'].fillna(method='ffill')

    smooth_tide_df = create_smooth_tides(tide_df)

    min_time = tide_df['datetime'].min()
    max_time = tide_df['datetime'].max()

    if pd.isna(min_time) or pd.isna(max_time):
        draw.error("Invalid time range in tide data")
        return

    vancouver_tz = pytz.timezone('America/Vancouver')
    full_index = pd.date_range(start=min_time, end=max_time, freq='15min')
    if full_index.tz is None:
        full_index = full_index.tz_localize(vancouver_tz)

    tide_interpolated = pd.DataFrame(index=full_index)
    x_timestamps = tide_df['datetime'].astype(np.int64) // 10 ** 9
    x_new_timestamps = full_index.astype(np.int64) // 10 ** 9
    tide_interpolated['Height'] = np.interp(
        x=x_new_timestamps,
        xp=x_timestamps,
        fp=tide_df['Height'].values
    )

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=smooth_tide_df['datetime'],
        y=smooth_tide_df['Height'],
        name='Tide Level',
        line=dict(color='#2E86C1', width=3),
        fill='tozeroy',
        fillcolor='rgba(46, 134, 193, 0.2)'
    ))

    fig.add_trace(go.Scatter(
        x=tide_df['datetime'],
        y=tide_df['Height'],
        mode='markers+text',
        name='Measured Points',
        text=[f"{t.strftime('%I:%M %p')}<br><b>{h:.2f}m</b>" for t, h in
              zip(tide_df['datetime'], tide_df['Height'])],
        textposition=['top center' if i % 2 == 0 else 'bottom center'
                      for i in range(len(tide_df))],
        textfont=dict(size=10, color='#2E86C1', family='Arial Black'),
        texttemplate='%{text}',
        dy=20,
        marker=dict(size=8, color='#1A5276', symbol='circle')
    ))

    fig.update_layout(
        title={'text': 'Tide Levels at Point Atkinson', 'y': 0.95, 'x': 0.5, 'xanchor': 'center', 'yanchor': 'top'},
        xaxis_title="Time",
        yaxis_title="Height (meters)",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)', zeroline=False),
        yaxis=dict(showgrid=True, gridwidth=1, gridcolor='rgba(128,128,128,0.2)',
                   zeroline=True, zerolinewidth=2, zerolinecolor='rgba(128,128,128,0.5)')
    )

    current_time = datetime.now(vancouver_tz)
    current_time_ts = current_time.timestamp() * 1000

    fig.add_vline(
        x=current_time_ts, line_width=2, line_dash="dash", line_color="red",
        annotation_text="Current Time", annotation_position="top right"
    )

    draw.plotly_chart(fig, width='stretch')

    col1, col2, col3 = draw.columns(3)

    # Match the chart exactly by interpolating the smooth (cubic-spline) curve.
    # The earlier `tide_interpolated` grid relied on .astype(np.int64) which can
    # mis-handle tz-aware DatetimeIndex on some pandas versions, producing
    # values shifted by hours from what the chart shows.
    smooth_x_seconds = np.array(
        [pd.Timestamp(t).timestamp() for t in smooth_tide_df['datetime']]
    )
    smooth_y_heights = np.asarray(smooth_tide_df['Height'].values, dtype=float)
    current_ts = current_time.timestamp()
    if smooth_x_seconds.size and smooth_x_seconds[0] <= current_ts <= smooth_x_seconds[-1]:
        current_height = float(np.interp(current_ts, smooth_x_seconds, smooth_y_heights))
    else:
        # Outside curve range — clamp to nearest endpoint
        current_height = float(
            smooth_y_heights[0] if current_ts < smooth_x_seconds[0]
            else smooth_y_heights[-1]
        )

    # Determine rising or falling by checking the next tide point
    tide_direction = ""
    if 'datetime' in tide_df.columns:
        try:
            next_tide = tide_df[tide_df['datetime'] > current_time].iloc[0]
            if next_tide['Height'] > current_height:
                tide_direction = "Rising"
            else:
                tide_direction = "Falling"
        except (IndexError, KeyError):
            pass

    col1.metric("Current Tide Level", f"{current_height:.2f}m", delta=tide_direction, delta_color="off")

    if 'datetime' in tide_df.columns:
        try:
            next_tide = tide_df[tide_df['datetime'] > current_time].iloc[0]
            time_diff = next_tide['datetime'] - current_time
            col2.metric(
                "Next Tide",
                f"{next_tide['Height']:.2f}m",
                f"in {time_diff.total_seconds() // 3600:.0f}h {(time_diff.total_seconds() // 60 % 60):.0f}m"
            )
        except (IndexError, KeyError):
            col2.metric("Next Tide", "No data available", "")
    else:
        col2.metric("Next Tide", "No data available", "")

    if 'Height' in tide_df.columns:
        daily_range = tide_df['Height'].max() - tide_df['Height'].min()
        col3.metric("Daily Tide Range", f"{daily_range:.2f}m")
    else:
        col3.metric("Daily Tide Range", "No data available")

    with draw.expander("Tide Table"):
        display_tide_table_text(tide_df=tide_df, container=st)


def displayErrorWithResponseIfNeeded(container=None, response=None):
    if not response:
        container.warning("No response.")
        return None

    if isinstance(response, dict):
        if hasattr(response, 'status_code'):
            if response.status_code == 402:
                container.warning("API quota exceeded. Using cached data if available.")
                return None
            if response.status_code == 500:
                container.warning("Internal Server Error. Try again later.")
                return None
            if response.status_code == 503:
                container.warning("Service Unavailable. Please try again later.")
                return None
            if response.status_code != 200:
                container.error(f"Failed to fetch tide data. Status code: {response.status_code}")
                return None

    return None


def processResponseToJSONStormglass(container=None, response=None):
    displayErrorWithResponseIfNeeded(container, response)
    if isinstance(response, (dict, list)):
        return response
    elif hasattr(response, 'json'):
        return response.json()
    else:
        if container:
            container.error("Invalid response format")
        return None


def find_local_extrema(df):
    """Keep only local maxima and minima from supersampled tide CSV."""
    height_array = df['height'].values

    maxima_indices = []
    for i in range(1, len(height_array) - 1):
        if height_array[i - 1] < height_array[i] > height_array[i + 1]:
            maxima_indices.append(i)

    minima_indices = []
    for i in range(1, len(height_array) - 1):
        if height_array[i - 1] > height_array[i] < height_array[i + 1]:
            minima_indices.append(i)

    extrema_indices = sorted(maxima_indices + minima_indices)
    df_extrema = df.iloc[extrema_indices].copy()
    df_extrema['type'] = ['high' if i in maxima_indices else 'low' for i in extrema_indices]

    return df_extrema


def processCSVResponseToJSONSelenium(container=None, _csv=None):
    """Parse Selenium-downloaded CSV into JSON tide data."""
    if not _csv:
        container.error("No CSV data received")
        return None

    _csv_no_timezone = _csv.replace(' PDT', '').replace(' PST', '')
    df = pd.read_csv(io.StringIO(_csv_no_timezone), on_bad_lines='skip', sep=',', skipinitialspace=True)

    df.columns = df.columns.str.replace('ï»¿', '')

    df['datetime'] = pd.to_datetime(df[df.columns[0]])

    pacific = pytz.timezone('America/Vancouver')
    df['datetime'] = df['datetime'].apply(lambda dt: pacific.localize(dt).isoformat())

    if 'predictions (m)' in df.columns:
        df.rename(columns={'predictions (m)': 'height'}, inplace=True)
    elif 'Predicted (m)' in df.columns:
        df.rename(columns={'Predicted (m)': 'height'}, inplace=True)
    elif 'prediction (m)' in df.columns:
        df.rename(columns={'prediction (m)': 'height'}, inplace=True)
    else:
        if container:
            container.error("Could not find predictions column. Available columns: " + ", ".join(df.columns.tolist()))
        df['height'] = df['observations (m)'] - df['Observations minus predictions (m)']

    df = df.dropna(subset=['height'])

    if len(df) < 3:
        if container:
            container.error(f"Not enough valid data points (only {len(df)} found)")
        return None

    df = find_local_extrema(df)

    json_result = []
    for _, row in df.iterrows():
        json_result.append({
            'Height': round(float(row['height']), 4),
            'Time (PDT)& Date': row['datetime'],
            'type': row['type']
        })

    return json_result


def processResponseToJSONOpenAI(container=None, response=None):
    """Parse ChatGPT response into JSON tide data."""
    import json

    displayErrorWithResponseIfNeeded(container, response)

    if USE_CHAT_GPT:
        data_txt = response.json()['choices'][0]['message']['content']
        data_txt = data_txt.strip()

        if data_txt.startswith('```json'):
            data_txt = data_txt.replace('```json', '', 1)
        if data_txt.startswith('```'):
            data_txt = data_txt.replace('```', '', 1)
        if data_txt.endswith('```'):
            data_txt = data_txt[:-3]
        data_txt = data_txt.strip()

        try:
            return json.loads(data_txt)
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Raw content:", data_txt)
    return None


def display_point_atkinson_tides(container=None, title="🌊Tides for Point Atkinson"):
    """Main entry point: fetch and display tide data for Point Atkinson."""
    draw = container or st
    draw.subheader(title)

    data = None
    draw.write(CANADA_GOVERNMENT_TIDE_POINT_ATKINSON)

    # Selenium fetch can take 5–10 seconds — show progress so the page doesn't look frozen.
    progress = draw.progress(0, text="Loading tide data…")

    try:
        if USE_BEAUTIFULSOUP:
            draw.badge("USE_BEAUTIFULSOUP")
            progress.progress(20, text="Scraping tides.gc.ca…")
            data = beautifulSoupFetchTidesForURL("https://www.tides.gc.ca/en/stations/07795")

        if USE_SELENIUM:
            progress.progress(15, text="Launching headless Chrome…")
            _csv = seleniumGetTidesFromURL('https://www.tides.gc.ca/en/stations/07795')
            progress.progress(60, text="Downloaded predictions, parsing CSV…")
            csv_lines = _csv.splitlines()
            csv_subsampled = '\n'.join(csv_lines[::20])
            csv_lines2 = csv_subsampled.splitlines()
            halfway_point = len(csv_lines2) // 3
            csv_half_subsampled = '\n'.join(csv_lines2[:halfway_point])
            progress.progress(80, text="Extracting tide extrema…")
            data = processCSVResponseToJSONSelenium(draw, csv_half_subsampled)

        if USE_CHAT_GPT:
            draw.badge("USE_CHAT_GPT")
            progress.progress(40, text="Calling OpenAI to parse forecast…")
            response = openAIFetchTidesForURL("https://www.tides.gc.ca/en/stations/07795")
            progress.progress(80, text="Parsing GPT response…")
            data = processResponseToJSONOpenAI(draw, response)

        if USE_STORMGLASS:
            draw.badge("USE_STORMGLASS")
            progress.progress(40, text="Querying Stormglass API…")
            response = stormglassFetchTidesPointAtkinson()
            if 'errors' in response:
                if 'key' in response['errors']:
                    error_msg = response['errors']['key']
                    if error_msg == 'API quota exceeded':
                        draw.error("Stormglass API quota exceeded.")
            else:
                data = processResponseToJSONStormglass(draw, response)

        if data:
            progress.progress(90, text="Building chart…")
            tide_data = process_tide_data(data, draw, use_chat_gpt=USE_CHAT_GPT)
            progress.progress(100, text="Done")
            progress.empty()
            if not isinstance(data, type(None)):
                create_natural_tide_chart(tide_data, draw)
            else:
                draw.error("Unable to fetch tide data. Please try again later.")
        else:
            progress.empty()
            draw.error("No data from fetch_tides.")
    except Exception:
        progress.empty()
        raise
