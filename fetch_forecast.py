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

def fetch_howe_sound_forecast():
    url = "https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400"

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
            'forecast': forecast_text
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



def display_howe_sound_forecast(container=None):
    if container is None:
        container = st

    result = fetch_howe_sound_forecast()
    issue_date = result['issue_date']
    title = result['title']
    subtitle = result['subtitle']
    forecast = result['forecast']

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