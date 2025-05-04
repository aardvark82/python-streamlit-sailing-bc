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

URL_forecast_howesound = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400'

def openAIParseForecastForURL(container, url):
    res = ''

    import json
    import os

    openai_api_key = st.secrets["OpenAI_key"] # put yout api key here
    if openai_api_key is None:
        raise ValueError("OpenAI API key is not set in environment variables.")

    response = requests.get(url, timeout=10)
    response.raise_for_status()

    chat_gpt_msg = "Make it short and just the table. Parse this forecast and extract a table with the following columns: time, wind speed, max wind speed, wind direction. "
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



def fetch_howe_sound_forecast():
    url = URL_forecast_howesound
    return fetch_marine_forecast_for_url(url)

def fetch_marine_forecast_for_url(url):
    url = URL_forecast_howesound

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

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

        title = "Howe Sound"  # This appears to be static based on the page URL

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


def display_howe_sound_forecast(container=None):
    return display_marine_forecast_for_url(container=container, url=URL_forecast_howesound)

def display_marine_forecast_for_url(container=None, url=''):
    if container is None:
        container = st

    result = fetch_howe_sound_forecast()
    issue_date = result['issue_date']
    title = result['title']
    subtitle = result['subtitle']
    forecast = result['forecast']
    wind_warning = result['wind_warning']
    strong_wind_warning = result['strong_wind_warning']

    if wind_warning:
        container.badge("wind warning in effect", color="orange")
    if strong_wind_warning:
        container.badge("wind warning in effect", color="red")

    # Display the structured wind table
    chatgpt_forecast = openAIParseForecastForURL(container=container, url=url)
    container.badge("chatGPT forecast")
    container.markdown(chatgpt_forecast)

    container.badge("BeautifulSoup forecast")

    if issue_date:
        # Display relative and absolute dates


        relative_date = timeago_format(issue_date, datetime.now(pytz.timezone('America/Vancouver')))
        container.header(f"Issued {relative_date}")
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
        container.error("Unable to fetch Howe Sound marine forecast")

    container.error

    return None