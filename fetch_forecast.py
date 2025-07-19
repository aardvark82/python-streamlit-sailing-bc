from dataclasses import replace

import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
import re

from streamlit import container
from timeago import format as timeago_format

import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd


def cached_fetch_url(url):
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return response

@st.cache_data(ttl=1800)
def openAIFetchForecastForURL(url):
    res = ''

    import json
    import os

    openai_api_key = st.secrets["OpenAI_key"] # put yout api key here
    if openai_api_key is None:
        raise ValueError("OpenAI API key is not set in environment variables.")

    response = cached_fetch_url(url)
    response.raise_for_status()

    chat_gpt_msg = ("Make it short and just the table. "
                    "Parse this forecast and extract a table with the following columns: time, wind speed, max wind speed, wind direction. "
                    "Make it a CSV."
                    "The first few words before the first occurence of 'becoming' describe the current conditions with a time of now and should be the 1st entry in the table")
    chat_gpt_msg = chat_gpt_msg + response.text
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
    else:
        print("Error:", response.status_code, response.text)
        res=("Error:", response.status_code, response.text)

    return res



import pdfplumber
import io
import re

@st.cache_data(ttl=1800)
def fetch_water_quality_for_url(url, title):
    try:
        st.write(f"Fetching data for: {title}")
        response = requests.get(url)
        response.raise_for_status()

        ecoli_sample1 = None
        ecoli_sample2 = None
        time_measurement = None

        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                # 1. Extract top-right timestamp using regex
                text = page.extract_text()
                date_match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)\b', text)
                if date_match and not time_measurement:
                    time_str = date_match.group(1)
                    tz = pytz.timezone("America/Vancouver")
                    naive_dt  = datetime.strptime(time_str, "%m/%d/%Y %I:%M %p")
                    time_measurement = tz.localize(naive_dt)

                # 2. Extract tables and look for the BWV-04-01 row
                tables = page.extract_tables()
                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
                    print(df)
                    if "Sample Name" in df.columns and "Ecoli MPN/100mLs" in df.columns:
                        match1 = df[df["Sample Name"].str.strip() == "BWV-04-01"]
                        if not match1.empty:
                            ecoli_sample1 = match1.iloc[0]["Ecoli MPN/100mLs"]
                    if "Sample Name" in df.columns and "Ecoli MPN/100mLs" in df.columns:
                        match2 = df[df["Sample Name"].str.strip() == "BWV-04-02"]
                        if not match2.empty:
                            ecoli_sample2 = match2.iloc[0]["Ecoli MPN/100mLs"]

        return ecoli_sample1, ecoli_sample2, time_measurement

    except Exception as e:
        st.error(f"Failed to fetch or parse water quality data from {url}: {e}")
        return None, None

@st.cache_data(ttl=1800)
def fetch_marine_forecast_for_url(url, title):
    """
       Fetches marine forecast data from a given URL, parsing wind warnings and forecast details.
       Extracts issue date, wind conditions, and other weather information.
       Returns a dictionary containing the processed forecast data with error handling.
       """

    try:
        response = cached_fetch_url(url)

        soup = BeautifulSoup(response.content, 'html.parser')
        if not soup:
            return {
                'error': True,
                'message': "Could not parse the weather page content",
                'details': "The page content was empty or invalid",
                'suggestion': "Please try refreshing the page"
            }

        # display Wind Warning
        # Check for wind warnings at the top of the page
        wind_warning_in_effect = False
        strong_wind_warning_in_effect = False

        # Look for warning text in the header section
        warning_banner = soup.find('div', id='warningBanner')
        if warning_banner:
            warning_text = warning_banner.text.lower().strip()
            strong_wind_warning_in_effect = "strong wind warning" in warning_text #extra NTSB character
            wind_warning_in_effect = "wind warning" in warning_text and not strong_wind_warning_in_effect

        # Find the main forecast content
        forecast_content = soup.find('div', id='forecast-content')
        if not forecast_content:
            return {
                'error': True,
                'message': "Could not find forecast information",
                'details': "The forecast section was not found on the page",
                'suggestion': "The website structure might have changed or the service might be temporarily down"
            }

        # Find the issue date
        issue_dt = forecast_content.find('span', class_='text-info')
        issue_date = None
        if issue_dt:
            issue_date_str = issue_dt.text.strip()
            match = re.match(r'Issued\s+(\d{1,2}:\d{2}\s+(?:AM|PM)\s+\w+\s+\d{2}\s+\w+\s+\d{4})', issue_date_str)
            if match:
                try:
                    date_str = match.group(1)
                    # Clean up the string - remove "Issued", &nbsp; and extra spaces
                    date_str = date_str.replace('Issued', '').replace('&nbsp;', ' ').strip()
                    
                    # Split and parse
                    parts = date_str.split()
                    if len(parts) == 6:  # Should be: ["10:30", "AM", "PDT", "03", "May", "2025"]
                        time_part, ampm, tz, day, month, year = parts
                        # Combine them in a format that strptime can handle
                        formatted_date_str = f"{time_part} {ampm} {day} {month} {year}"
                        issue_date = datetime.strptime(formatted_date_str, '%I:%M %p %d %B %Y')
                        
                        vancouver_tz = pytz.timezone('America/Vancouver')
                        if issue_date.tzinfo is None:
                            issue_date = vancouver_tz.localize(issue_date)
                        else:
                            issue_date = issue_date.astimezone(vancouver_tz)

                except ValueError as e:
                    # Instead of printing, return the error
                    container.error = {
                        'error': True,
                        'message': f"Could not parse date: {date_str}",
                        'details': str(e),
                        'suggestion': "The date format might have changed"
                    }
                    return None, None, None, None

        # Find the period of coverage (subtitle)
        period_elem = forecast_content.find('span', class_='periodOfCoverage')
        period_coverage = period_elem.text.strip() if period_elem else ""

        # Find any warnings
        warning_elem = forecast_content.find('span', class_='text-danger')
        warning = warning_elem.text.strip() if warning_elem else ""

        # Find the forecast text
        forecast_elem = forecast_content.find('span', class_='textSummary')
        forecast_text = forecast_elem.text.strip() if forecast_elem else ""

        if not forecast_text:
            return {
                'error': True,
                'message': "No forecast text found",
                'details': "The forecast text section was empty",
                'suggestion': "Please check back later for updated forecast"
            }

        title = title  # This appears to be static based on the page URL

        return {
            'error': False,
            'issue_date': issue_date,
            'title': title,
            'subtitle': period_coverage,
            'warning': warning,
            'forecast': forecast_text,
            'wind_warning': wind_warning_in_effect,
            'strong_wind_warning': strong_wind_warning_in_effect,
        }

    except requests.Timeout:
        return {
            'error': True,
            'message': "Connection timeout",
            'details': "The weather service is taking too long to respond",
            'suggestion': "Please try again in a few minutes"
        }
    except requests.RequestException as e:
        return {
            'error': True,
            'message': "Network error",
            'details': str(e),
            'suggestion': "Check your internet connection and try again"
        }
    except Exception as e:
        return {
            'error': True,
            'message': "Unexpected error",
            'details': str(e),
            'suggestion': "Please try again later or report this issue"
        }


import re
from datetime import datetime, timedelta


def parse_wind_forecast(forecast_text):
    # Split into time segments
    segments = forecast_text.split('then')

    # Initialize list to store parsed data
    wind_data = []

    # Regular expressions for parsing
    wind_pattern = r'(?P<direction>[A-Za-z]+)\s+(?:inflow|outflow)?\s*(?P<speed>\d+)\s*to\s*(?P<speed_high>\d+)'

    for segment in segments:
        segment = segment.strip()

        # Extract time period
        time_period = ""
        if "evening" in segment.lower():
            time_period = "Evening"
        elif "afternoon" in segment.lower():
            time_period = "Afternoon"
        elif "morning" in segment.lower():
            time_period = "Morning"
        elif "overnight" in segment.lower():
            time_period = "Overnight"
        elif "late overnight" in segment.lower():
            time_period = "Late Overnight"

        # Extract wind information
        if "light" in segment.lower():
            wind_data.append({
                "time_period": time_period,
                "direction": "Variable",
                "speed": "Light",
                "speed_range": "<15"
            })
        else:
            match = re.search(wind_pattern, segment)
            if match:
                wind_data.append({
                    "time_period": time_period,
                    "direction": match.group("direction"),
                    "speed_range": f"{match.group('speed')}-{match.group('speed_high')}",
                    "speed": f"{match.group('speed')}-{match.group('speed_high')}"
                })

    return wind_data


def standardize_wind_direction(direction):
    if pd.isna(direction):  # Handle NaN values
        return direction

    if not isinstance(direction, str):  # Handle non-string values
        return direction

    direction = direction.lower().strip()

    # Dictionary mapping various forms to standard abbreviations
    direction_map = {
        'north': 'N',
        'north northeasterly': 'NNE',
        'northeast': 'NE',
        'east northeasterly': 'ENE',
        'east': 'E',
        'east southeasterly': 'ESE',
        'southeast': 'SE',
        'south southeasterly': 'SSE',
        'south': 'S',
        'south southwesterly': 'SSW',
        'southwest': 'SW',
        'west southwesterly': 'WSW',
        'west': 'W',
        'west northwesterly': 'WNW',
        'northwest': 'NW',
        'north northwesterly': 'NNW',
        'northerly': 'N',
        'northeasterly': 'NE',
        'easterly': 'E',
        'southeasterly': 'SE',
        'southerly': 'S',
        'southwesterly': 'SW',
        'westerly': 'W',
        'northwesterly': 'NW',
        'northerly outflow': 'N',
        'southerly inflow': 'S'
    }

    return direction_map.get(direction, direction)


def create_arrow_html(direction, wind_speed = ''):
    # Convert cardinal directions to degrees
    direction_degrees = {
        'N': 0,
        'NNE': 22.5,
        'NE': 45,
        'ENE': 67.5,
        'E': 90,
        'ESE': 112.5,
        'SE': 135,
        'SSE': 157.5,
        'S': 180,
        'SSW': 202.5,
        'SW': 225,
        'WSW': 247.5,
        'W': 270,
        'WNW': 292.5,
        'NW': 315,
        'NNW': 337.5
    }
    degree = direction_degrees.get(direction.upper(), 1000)
    if degree == 1000:
        return '<div style="text-align: center;"><div style="width: 20px; height: 20px;background-color: #808080; border-radius: 50%; display: inline-block;"></div></div>'
    else:
        # Handle different wind speed types
        if isinstance(wind_speed, (int, float)):
            wind_speed_value = int(wind_speed)
        else:
            # Convert string to number if possible
            wind_speed = str(wind_speed).split()[0]  # Get first part of string
            wind_speed_value = int(float(wind_speed)) if wind_speed.replace('.', '').strip().isdigit() else 0

        arrow_count = max(0, int(wind_speed_value / 5))

        if arrow_count == 0:
            return '<div style="text-align: center;"><div style="width: 20px; height: 20px; background-color: #1f77b4; border-radius: 50%; display: inline-block;"></div></div>'

        arrow_html = f'<div style="text-align: center; white-space: nowrap;">'
        for _ in range(arrow_count):
            arrow_html += f'<div style="width: 0; height: 0; border-left: 10px solid transparent; border-right: 10px solid transparent; border-bottom: 30px solid #1f77b4; display: inline-block; transform: rotate({180+degree}deg); margin: 0 5px;"></div>'
        arrow_html += '</div>'
        return arrow_html


def display_beach_quality_for_url(container=None, url='', title=''):
    if container is None:
        container = st

    ecoli_sample1, ecoli_sample2, time_measurement = fetch_water_quality_for_url(url='https://www.vch.ca/en/Documents/VCH-beach-route3.pdf', title=title)

    relative_date = timeago_format(time_measurement, datetime.now(pytz.timezone('America/Vancouver')))

    container.header(f"Issued {relative_date}")
    if not ecoli_sample1:
        ecoli_sample1 = 'not found'
    if not ecoli_sample2:
        ecoli_sample2 = 'not found'

    col1, col2, col3, col4 = container.columns(4)
    col1.text( 'Sandy cove station 1 BWV-04-01')
    col2.badge(ecoli_sample1, color='green')
    if int(ecoli_sample1) > 200:
        col2.badge( ecoli_sample1, color='orange')
    if int(ecoli_sample1) > 400:
        col2.badge( ecoli_sample1, color='red')

    col3.text( 'Sandy cove station 2 BWV-04-02')
    col4.badge(ecoli_sample2, color='green')
    if int(ecoli_sample2) > 200:
        col4.badge( ecoli_sample2, color='orange')
    if int(ecoli_sample2) > 400:
        col4.badge( ecoli_sample2, color='red')




def display_marine_forecast_for_url(container=None, url='', title=''):
    if container is None:
        container = st

    result = fetch_marine_forecast_for_url(url, title)
    issue_date = result['issue_date']
    title = result['title']
    subtitle = result['subtitle']
    forecast = result['forecast']
    wind_warning = result['wind_warning']
    strong_wind_warning = result['strong_wind_warning']

    container.subheader("Marine Forecast for "+title)
    container.write(url)

    relative_date = timeago_format(issue_date, datetime.now(pytz.timezone('America/Vancouver')))
    container.header(f"Issued {relative_date}")

    if wind_warning:
        container.badge("wind warning in effect", color="orange")
    if strong_wind_warning:
        container.badge("wind warning in effect", color="red")

    # Display the structured wind table
    chatgpt_forecast = openAIFetchForecastForURL(url=url)

    import io
    # Create StringIO object from the CSV string
    chatgpt_forecast = chatgpt_forecast.replace('```csv','')
    chatgpt_forecast = chatgpt_forecast.replace('```','')
    # Read CSV from StringIO
    csv_stringio = io.StringIO(chatgpt_forecast)
    df = pd.read_csv(csv_stringio, sep=',', on_bad_lines='skip')
    # Clean up the dataframe
    df = df.dropna(how='all')  # Remove empty rows
    df = df.reset_index(drop=True)  # Reset index after dropping rows

    # Apply the standardization to the wind direction column
    df['wind direction'] = df['wind direction'].str.lower().apply(standardize_wind_direction)

    df['wind speed'] = df['wind speed'].fillna(0)
    df['max wind speed'] = df['max wind speed'].fillna('<5')

    print(*df)

    col1, col2, col3, col4 = container.columns(4)
    col1.badge( df['time'].iloc[0], color='red')
    col2.metric("Wind Speed", df['wind speed'].iloc[0])
    col3.metric("Wind High", df['max wind speed'].iloc[0])

    wind_direction = df['wind direction'].iloc[0]
    col4.metric("Direction", wind_direction)
    col1.markdown(create_arrow_html(wind_direction,df['wind speed'].iloc[0] ), unsafe_allow_html=True)


    col21, col22, col23, col24 = container.columns(4)
    col21.badge( df['time'].iloc[1], color='red')
    col22.metric("Wind Speed", df['wind speed'].iloc[1])
    col23.metric("Wind High", df['max wind speed'].iloc[1])

    wind_direction = df['wind direction'].iloc[1]
    col24.metric("Direction", wind_direction)
    col21.markdown(create_arrow_html(wind_direction,df['wind speed'].iloc[1]), unsafe_allow_html=True)

    container.dataframe(df)
    container.badge("chatGPT forecast")

    container.divider()


    if issue_date:
        # Display relative and absolute dates
        container.caption(f"({issue_date.strftime('%Y-%m-%d %I:%M %p %Z')})")

        # Display title and subtitle
        container.subheader(title)
        container.write(subtitle)

        # Display forecast with bold numbers
        if forecast:
            # Bold all numbers in the forecast
            bold_forecast = re.sub(r'(\d+(?:\.\d+)?)', r'**\1**', forecast)
            container.markdown("""
            <div style="padding: 1em; border-radius: 5px; background-color: #f0f2f6;">
                {}
            </div>
            """.format(bold_forecast), unsafe_allow_html=True)
    else:
        container.error("Unable to fetch "+title+" marine forecast")
    container.badge("BeautifulSoup forecast")


    return None
