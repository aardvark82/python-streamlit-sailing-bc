
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pytz
import re
from timeago import format as timeago_format

import streamlit as st
import requests
from bs4 import BeautifulSoup


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_howe_sound_forecast():
    url = "https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400"

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')
        if not soup:
            return None, None, None, None

        # Find the issue date
        issue_date = None
        issue_dt = soup.find('dl', class_='dl-horizontal')
        if issue_dt:
            issue_date_elem = issue_dt.find('dt', string=re.compile('Issued'))
            if issue_date_elem and issue_date_elem.find_next_sibling('dd'):
                issue_date_str = issue_date_elem.find_next_sibling('dd').text.strip()
                try:
                    issue_date = datetime.strptime(issue_date_str, '%Y-%m-%d %H:%M %Z')
                    issue_date = pytz.timezone('America/Vancouver').localize(issue_date)
                except ValueError:
                    pass

        # Find the forecast content
        content_div = soup.find('div', {'class': 'row'})
        if not content_div:
            return None, None, None, None

        # Extract paragraphs
        paragraphs = content_div.find_all('p')
        if len(paragraphs) < 3:
            return None, None, None, None

        # Get title, subtitle, and forecast
        title = paragraphs[0].text.strip() if paragraphs[0] else "Howe Sound"
        subtitle = paragraphs[1].text.strip() if paragraphs[1] else ""

        # Get the main forecast text
        forecast_text = None
        for p in paragraphs[2:]:
            text = p.text.strip()
            if text and not text.startswith('Issued'):
                forecast_text = text
                break

        return issue_date, title, subtitle, forecast_text

    except requests.RequestException as e:
        st.error(f"Network error: {str(e)}")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error fetching Howe Sound forecast: {str(e)}")
        return None, None, None, None

def display_howe_sound_forecast(container=None):
    if container is None:
        container = st

    issue_date, title, subtitle, forecast = fetch_howe_sound_forecast()

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