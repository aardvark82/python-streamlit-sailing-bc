


import streamlit as st
import requests
from bs4 import BeautifulSoup
import pytz
import numpy as np
import pandas as pd

from datetime import datetime
import pytz

USE_BEAUTIFULSOUP = False #otherwise use stormglass.io API
USE_CHAT_GPT = False #otherwise use stormglass.io API
USE_STORMGLASS = True #otherwise use stormglass.io API

MAKE_LIVE_REQUESTS_STORMGLASS = True
API_KEY_STORMGLASS_IO = '4b108f2a-27f4-11f0-88e2-0242ac130003-4b109010-27f4-11f0-88e2-0242ac130003'

CANADA_GOVERNMENT_TIDE_POINT_ATKINSON = "https://www.tides.gc.ca/en/stations/07795"

import re
from datetime import datetime, timedelta

#@st.cache_data(ttl=1800)  # Cache for 1/2 hours
def beautifulSoupFetchTidesForURL(url):
    # Download the page (you may need headers if blocked)
    # url = "https://www.tides.gc.ca/en/stations/07795"
    # Add headers to mimic a browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()  # Raise an exception for bad status codes

    soup = BeautifulSoup(response.content, "html.parser")

    # Prepare regex and helpers
    def classify_tide(height):
        return "high" if float(height) > 2.0 else "low"

    def parse_time(date_str, time_str):
        dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
        # Convert PDT (UTC-7) to UTC
        dt_utc = dt + timedelta(hours=7)
        return dt_utc.isoformat() + "+00:00"

    print("URL accessed:", url)
    print("Response status:", response.status_code)

    data = {"data": []}

    # Find the 7-day tide table block
    tide_sections = soup.select("div.tide-table")  # Usually there are multiple day tables

    for section in tide_sections:
        date_header = section.find_previous("h3")
        if not date_header:
            continue

        # Extract date (e.g., '2025-05-04 (Sun)' -> '2025-05-04')
        date_match = re.search(r"\d{4}-\d{2}-\d{2}", date_header.text)
        if not date_match:
            continue
        date = date_match.group(0)

        # Extract rows of time + height in meters
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

    print(data)
    return data


#@st.cache_data(ttl=1800)  # Cache for 1/2 hours
def openAIFetchTidesForURL(url):
    res = ''

    import json
    import os
    print("Calling OpenAI...")
    openai_api_key = st.secrets["OpenAI_key"] # put yout api key here
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
    # option 1 fetch the entire page - Too many tokens
    # response = requests.get(url, timeout=25)
    # response.raise_for_status()
    # html = response.text

    #option 2 fetch only the HTML section we need using Beauitful Soup and pass it to ChatGPT for processing
    def beautifulSoupFetchTidesSectionForChatGPTForURL(url):
        try:
            # Add headers to mimic a browser request
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            response = requests.get(url, headers=headers, timeout=25)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Find the forecast section - typically in a div with specific class or id
            # Find the hourly predictions table
            hourly_table = soup.find('table', class_='hourlytable')

            if hourly_table:
                return str(hourly_table)
            else:
                return "Predictions section not found"


        except requests.RequestException as e:
            return f"Error fetching data: {str(e)}"

    html = beautifulSoupFetchTidesSectionForChatGPTForURL(url)

    # EXTRACT THE SECTION 7 DAYS AND HOURLY TO REDUCE TOKENS

    chat_gpt_msg = chat_gpt_msg + "This is the HTML " + html
    url_api = "https://api.openai.com/v1/chat/completions"

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }

    data = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert meteorologist ."
            },
            {
                "role": "user",
                "content": chat_gpt_msg
            }
        ]
    }

    response = requests.post(url_api, headers=headers, json=data)

    # Check if the request was successful
    if response.status_code == 200:
        print("Response from OpenAI:", response.json())
        print('\n')
        print(response.json()['choices'][0]['message']['content'])
        res = response.json()['choices'][0]['message']['content']
    elif response.status_code == 429:
        print("Response from OpenAI:", response.json())
        print('\n')
        print(response.json()['choices'][0]['message']['content'])
        res = response.json()['choices'][0]['message']['content']
    else:
        print("Error:", response.status_code, response.text)
        res=("Error:", response.status_code, response.text)

    return response




@st.cache_data(ttl=14400)  # Cache for 4 hours
def stormglassFetchTidesPointAtkinson():

    """Fetch tide data for Point Atkinson from Stormglass API"""
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    import pytz
    import json

    try:
        # Point Atkinson coordinates
        lat = 49.3304
        lon = -123.2646

        if MAKE_LIVE_REQUESTS_STORMGLASS:
            # Stormglass API configuration
            base_url = "https://api.stormglass.io/v2/tide/extremes/point"
            api_key = st.secrets["stormglass_key"]

            vancouver_tz = pytz.timezone('America/Vancouver')
            now = datetime.now(vancouver_tz)
            
            # Get 4 days before and 4 days after current time
            start_date = now - timedelta(days=1)
            end_date = now + timedelta(days=2)

            params = {
                'lat': lat,
                'lng': lon,
                'start': start_date.strftime('%Y-%m-%d'),
                'end': end_date.strftime('%Y-%m-%d')
            }

            headers = {
                'Authorization': api_key
            }

            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            return response.json()

        else:
            #container.warning("Using stub data for tide data")
            # Use the stub data when not making live requests
            response = {  # 2 days of data, 4 points per day
                "data": [
                    {"height": 1.6281896122083954, "time": "2025-04-29T02:59:00+00:00", "type": "high"},
                    {"height": 0.09792586003904245, "time": "2025-04-29T08:16:00+00:00", "type": "low"},
                    {"height": 1.374891079134516, "time": "2025-04-29T13:10:00+00:00", "type": "high"},
                    {"height": -2.7679098753738876, "time": "2025-04-29T20:27:00+00:00", "type": "low"},
                    {"height": 1.716952818959448, "time": "2025-04-30T03:55:00+00:00", "type": "high"},
                    {"height": 0.32124120920566857, "time": "2025-04-30T09:10:00+00:00", "type": "low"},
                    {"height": 1.2659443022297774, "time": "2025-04-30T13:45:00+00:00", "type": "high"},
                    {"height": -2.755521186344923, "time": "2025-04-30T21:11:00+00:00", "type": "low"}
                ]
            }
            # 2 days of data, 4 points per day
            response = {'data': [{'height': 1.6432596903918761, 'time': '2025-05-02T05:51:00+00:00', 'type': 'high'},
                             {'height': 0.5114702000679019, 'time': '2025-05-02T11:16:00+00:00', 'type': 'low'},
                             {'height': 0.8861352640591091, 'time': '2025-05-02T15:04:00+00:00', 'type': 'high'},
                             {'height': -2.339105387349193, 'time': '2025-05-02T22:49:00+00:00', 'type': 'low'},
                             {'height': 1.551146455775455, 'time': '2025-05-03T06:52:00+00:00', 'type': 'high'},
                             {'height': 0.4471043214497481, 'time': '2025-05-03T12:40:00+00:00', 'type': 'low'},
                             {'height': 0.624905587275962, 'time': '2025-05-03T15:55:00+00:00', 'type': 'high'},
                             {'height': -2.0247559154532104, 'time': '2025-05-03T23:44:00+00:00', 'type': 'low'}
                             ]
                    }
            return response
    except Exception as e:
        print(f"Error fetching tide data: {e}")
        return None


def process_tide_data(data, container=None, use_chat_gpt=False):

        print (data)

        # Convert predictions to pandas DataFrame
        predictions = []

        dt = None

        for prediction in data['data']:
            dt = pd.to_datetime(prediction['time'])
            dt = dt.tz_convert('America/Vancouver')
            predictions.append({
                'Time (PDT)& Date': dt,
                'Height': float(prediction['height'])
            })

    # correct to stormio data
        if USE_STORMGLASS:
        # add +2.94m to all tides - stormglass gets it wrong
            for i in range(len(predictions)):
               predictions[i]['Height'] = predictions[i]['Height'] + 2.94
               predictions[i]['Time (PDT)& Date'] = predictions[i]['Time (PDT)& Date'] + pd.Timedelta(hours=0, minutes=18)

        # Create DataFrame and sort by time
        tide_df = pd.DataFrame(predictions)
        tide_df = tide_df.sort_values('Time (PDT)& Date', ignore_index=True)
        print(tide_df)
        return tide_df


def display_tide_table_text(tide_df, container=None):

    if container:
        draw = container
    else:
        draw = st

    draw.markdown("---")

    # Display tide table at the top
    if not tide_df.empty:
        # Format the dataframe for display
        display_df = tide_df.copy()
        display_df['Time'] = display_df['datetime'].dt.strftime('%I:%M %p')
        display_df['Date'] = display_df['datetime'].dt.strftime('%A, %b %d')
        display_df['Height (m)'] = display_df['Height'].round(2)

        # Select and order columns for display
        table_df = display_df[['Date', 'Time', 'Height (m)']].copy()

        # Style the dataframe
        styled_df = table_df.style.set_properties(**{
            'background-color': 'white',
            'color': 'black',
            'border-color': '#e1e4e8'
        }).hide(axis='index')

        # Display the table
        draw.dataframe(styled_df, use_container_width=True)

        # Add some space after the table
        draw.markdown("---")

def parse_tide_datetime(time_str):
    """Parse datetime string from tide data"""
    import pandas as pd
    from datetime import datetime
    import pytz

    try:
        # Parse the datetime string
        dt = pd.to_datetime(time_str)

        # Make sure it's timezone aware and convert to Vancouver time
        if dt.tzinfo is None:
            dt = dt.tz_localize('UTC')

        vancouver_tz = pytz.timezone('America/Vancouver')
        dt = dt.tz_convert(vancouver_tz)

        return dt
    except Exception as e:
        print(f"Error parsing datetime: {e}")
        return pd.NaT


# After creating tide_df, add interpolation:
from scipy.interpolate import CubicSpline
def create_smooth_tides(df):
    # Use existing timezone-aware timestamps
    times = df['datetime']
    base_time = times.iloc[0]

    # Convert to seconds since base_time using timestamp() method
    x = [(t.timestamp() - base_time.timestamp()) / 3600 for t in times]
    y = df['Height'].values

    # Create more points for smooth curve
    x_smooth = np.linspace(min(x), max(x), 200)  # 200 points for smooth curve

    # Cubic spline interpolation
    cs = CubicSpline(x, y, bc_type='natural')
    y_smooth = cs(x_smooth)

    # Convert back to timestamps while preserving timezone
    # Create timedelta objects and add them to the base_time
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
        # Extract the number before 'm'
        meters = float(height_str.split('m')[0].strip())
        return meters
    except (ValueError, AttributeError) as e:
        print(f"Error parsing height from value: {height_str}")
        return None


def create_natural_tide_chart(tide_df, container=None):
    if container:
        draw = container
    else:
        draw = st

    #### Time cleaning

    # Cleanup columns
    print("----------------------------------------------------------------------------")
    print("ALL TIDES")
    print("----------------------------------------------------------------------------")
    print(tide_df)

    print("Columns in tide_df:", tide_df.columns)
    # The datetime is already in the Time column, so we'll use that directly
    # First convert to datetime
    tide_df = tide_df.rename(columns={'Time (PDT)& Date': 'datetime'})

    tide_df['datetime'] = tide_df['datetime'].apply(parse_tide_datetime)
    print("----------------------------------------------------------------------------")
    print("ALL TIDES CLEAN TIME")
    print("----------------------------------------------------------------------------")
    print(tide_df)

    #### Height cleaning
    # Clean the height data - remove any 'm' or other units if present
    # Debug: Print the data types and check for any non-numeric values
    print("Height column data:", tide_df['Height'])


    tide_df['Height'] = tide_df['Height'].astype(str).apply(extract_meters)

    # After parsing heights, check if we have valid data
    if tide_df['Height'].isnull().all():
        draw.error("No valid height data available")
        return

    # Remove any remaining null values before interpolation
    tide_df = tide_df.dropna(subset=['Height', 'datetime'])

    if len(tide_df) < 2:
        draw.error("Not enough valid tide data points for interpolation")
        return

    if tide_df['Height'].isnull().any():
        # Optionally, report or handle NAs here
        # For now, let's forward-fill them (or use .dropna())
        tide_df['Height'] = tide_df['Height'].fillna(method='ffill')
    print("----------------------------------------------------------------------------")
    print("ALL TIDES CLEAN HEIGHT")
    print("----------------------------------------------------------------------------")
    print(tide_df)

    # Create interpolated dataframe
    smooth_tide_df = create_smooth_tides(tide_df)

    # Resample to 15-minute intervals
    min_time = tide_df['datetime'].min()
    max_time = tide_df['datetime'].max()

    if pd.isna(min_time) or pd.isna(max_time):
        draw.error("Invalid time range in tide data")
        return

    # Create timezone-aware date range
    full_index = pd.date_range(
        start=min_time,
        end=max_time,
        freq='15min'
    )

    # Make the index timezone aware if it isn't already
    vancouver_tz = pytz.timezone('America/Vancouver')
    if full_index.tz is None:
        full_index = full_index.tz_localize(vancouver_tz)

    # Create interpolated series
    tide_interpolated = pd.DataFrame(index=full_index)

    # Convert to timestamps for interpolation
    x_timestamps = tide_df['datetime'].astype(np.int64) // 10 ** 9
    x_new_timestamps = full_index.astype(np.int64) // 10 ** 9

    # Perform interpolation
    tide_interpolated['Height'] = np.interp(
        x=x_new_timestamps,
        xp=x_timestamps,
        fp=tide_df['Height'].values
    )
    print("----------------------------------------------------------------------------")
    print("smooth tide_interpolated")
    print("----------------------------------------------------------------------------")
    print(smooth_tide_df)

    # Create the visualization
    draw.subheader("🌊 Point Atkinson Tide Chart")

    # Use Plotly for better interactivity
    import plotly.graph_objects as go

    # Before creating the figure, ensure both dataframes have the same timezone
    pacific_tz = pytz.timezone('America/Los_Angeles')

    fig = go.Figure()
    print("----------------------------------------------------------------------------")
    print(" tide_df BEFORE DRAW")
    print("----------------------------------------------------------------------------")
    print(tide_df)
    print("----------------------------------------------------------------------------")
    print(" smooth_tide_df BEFORE DRAW")
    print("----------------------------------------------------------------------------")
    print(smooth_tide_df)

    # Add the smooth tide line
    fig.add_trace(go.Scatter(
        x=smooth_tide_df['datetime'],
        y=smooth_tide_df['Height'],
        name='Tide Level',
        line=dict(color='#2E86C1', width=3),
        fill='tozeroy',  # Fill to zero
        fillcolor='rgba(46, 134, 193, 0.2)'  # Light blue fill
    ))

    # Add actual data points with spaced, bold, red labels
    fig.add_trace(go.Scatter(
        x=tide_df['datetime'],
        y=tide_df['Height'],
        mode='markers+text',
        name='Measured Points',
        text=[f"{t.strftime('%I:%M %p')}<br><b>{h:.2f}m</b>" for t, h in
              zip(tide_df['datetime'], tide_df['Height'])],
        textposition=['top center' if i % 2 == 0 else 'bottom center'
                      for i in range(len(tide_df))],
        textfont=dict(
            size=10,
            color='#2E86C1',  # Medium-bright blue that's readable on both light and dark backgrounds
            family='Arial Black'
        ),
        texttemplate='%{text}',
        dy=20,  # 10 pixels up or down
        marker=dict(
            size=8,
            color='#1A5276',
            symbol='circle'
        )
    ))
    # Customize the layout
    fig.update_layout(
        title={
            'text': 'Tide Levels at Point Atkinson',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Time",
        yaxis_title="Height (meters)",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(128,128,128,0.5)'
        )
    )

    # Add current time marker
    vancouver_tz = pytz.timezone('America/Vancouver')
    current_time = datetime.now(vancouver_tz)
    current_time_ts = current_time.timestamp() * 1000  # multiply by 1000 for milliseconds

    fig.add_vline(
        x=current_time_ts,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text="Current Time",
        annotation_position="top right"
    )

    # Show the plot in Streamlit
    draw.plotly_chart(fig, use_container_width=True)

    # Add tide statistics
    col1, col2, col3 = draw.columns(3)

    current_height = np.interp(
        current_time.timestamp(),
        tide_interpolated.index.astype(np.int64) // 10 ** 9,
        tide_interpolated['Height']
    )

    col1.metric(
        "Current Tide Level",
        f"{current_height:.2f}m",
    )

    # Check if we have the tide data columns before trying to display next tide info
    if 'datetime' in tide_df.columns:
        try:
            next_tide = tide_df[tide_df['datetime'] > current_time].iloc[0]
            time_diff = (next_tide['datetime'] - current_time)

            # Use Height only since we don't have Type information
            col2.metric(
                "Next Tide",
                f"{next_tide['Height']:.2f}m",
                f"in {time_diff.total_seconds() // 3600:.0f}h {(time_diff.total_seconds() // 60 % 60):.0f}m"
            )
        except (IndexError, KeyError):
            col2.metric(
                "Next Tide",
                "No data available",
                ""
            )
    else:
        col2.metric(
            "Next Tide",
            "No data available",
            ""
        )

    if 'Height' in tide_df.columns:
        daily_range = tide_df['Height'].max() - tide_df['Height'].min()
        col3.metric(
            "Daily Tide Range",
            f"{daily_range:.2f}m"
        )
    else:
        col3.metric(
            "Daily Tide Range",
            "No data available"
        )

    if USE_CHAT_GPT:
        draw.badge("From ChatGPT")
    else:
        draw.badge("From Stormglass.io")

    display_tide_table_text(tide_df=tide_df, container=draw) # debug



def displayErrorWithResponseIfNeeded(container = None, response = None):
    if not response:
        error_msg = ("No response.")
        container.warning(error_msg)
        return None

    if hasattr(response, 'status_code'):  # Check if response object has status_code attribute
        if response.status_code == 402:
            error_msg = ("API quota exceeded. Please wait for quota reset or check your subscription. "
                         "Using cached data if available.")
            if container:
                container.warning(error_msg)
            return None

        if response.status_code == 500:
            error_msg = ("Internal Server Error – We had a problem with our server. Try again later..")
            if container:
                container.warning(error_msg)
            return None

        if response.status_code == 503:
            error_msg = ("Service Unavailable – We’re temporarily offline for maintenance. Please try again later.")
            if container:
                container.warning(error_msg)
            return None

        if response.status_code != 200:
            error_msg = f"Failed to fetch tide data. Status code: {response.status_code}"
            if container:
                container.error(error_msg)
            return None

    if isinstance(response, dict) and 'data' in response:
        data = response['data']
        # container.text(data)

    return None


def processResponseToJSONStormglass(container = None, response = None):
    data = None
    displayErrorWithResponseIfNeeded(container, response)
    if isinstance(response, (dict, list)):
        data = response
    # If response is a requests Response object
    elif hasattr(response, 'json'):
        data = response.json()
    else:
        if container:
            container.error("Invalid response format")
        return None

    return data


def processResponseToJSONOpenAI(container = None, response = None):
    displayErrorWithResponseIfNeeded(container, response)

    if USE_CHAT_GPT:
        import json
        data_txt = (response.json()['choices'][0]['message']['content'])
        # Clean up the text before parsing
        # Remove any leading/trailing whitespace
        data_txt = data_txt.strip()

        # If the string starts with a backtick block (common in ChatGPT responses), remove it
        if data_txt.startswith('```json'):
            data_txt = data_txt.replace('```json', '', 1)
        if data_txt.startswith('```'):
            data_txt = data_txt.replace('```', '', 1)
        if data_txt.endswith('```'):
            data_txt = data_txt[:-3]

        # Remove any leading/trailing whitespace again after cleaning
        data_txt = data_txt.strip()

        try:
            data = json.loads(data_txt)
            return data
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON: {e}")
            print("Raw content:", data_txt)
            # Fallback to empty data structure
            data = {"data": []}
    return None



# Modify your displayPointAtkinsonTides function to use the new visualization
def displayPointAtkinsonTides(container=None, title="Point Atkinson"):
    if container:
        draw = container
    else:
        draw = st

    container.subheader("Tides for "+title)

    # Fetch the tide data
    if USE_BEAUTIFULSOUP:
        draw.badge("USE_BEAUTIFULSOUP")
    if USE_CHAT_GPT:
        draw.badge("USE_CHAT_GPT")
    if USE_STORMGLASS:
        draw.badge("USE_STORMGLASS")

    data = None
    draw.write(CANADA_GOVERNMENT_TIDE_POINT_ATKINSON)

    if USE_BEAUTIFULSOUP:
        response = beautifulSoupFetchTidesForURL("https://www.tides.gc.ca/en/stations/07795")
        data = response
        # response is a json

    if USE_CHAT_GPT:
        response = openAIFetchTidesForURL("https://www.tides.gc.ca/en/stations/07795")
        # response is a response with a json
        data = processResponseToJSONOpenAI(draw, response)

    if USE_STORMGLASS:
        response = stormglassFetchTidesPointAtkinson()
        if 'errors' in response:
            if 'key' in response['errors']:
                error_msg = response['errors']['key']
                if error_msg == 'API quota exceeded':
                    draw.error("Stormglass API quota exceeded. Please wait for quota reset or check your subscription.")
        else:
            data = processResponseToJSONStormglass(draw, response)

        # response is a response
    if data:
        tide_data = process_tide_data(data, draw, use_chat_gpt=USE_CHAT_GPT)
        if not isinstance(data, type(None)):
            # Create the natural tide chart
            create_natural_tide_chart(tide_data, draw)
        else:
            draw.error("Unable to fetch tide data. Please try again later.")

    else:
        draw.error("No data from fetch_tides.")

