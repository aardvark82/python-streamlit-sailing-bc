import re
import io
import requests
import streamlit as st
import pandas as pd
import pytz
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
from timeago import format as timeago_format

from utils import cached_fetch_url
from wind_utils import create_arrow_html


def openAIFetchForecastForURL(url):
    """Use GPT-4o to parse a marine forecast page into a CSV table of wind data.
    The response depends on the time of day (evening vs not) so the cache key includes it."""
    vancouver_tz = pytz.timezone('America/Vancouver')
    now_pacific = datetime.now(vancouver_tz)

    # Time-bucket for the cache key: crossing 7 PM / 9 PM / 11 PM must trigger a refetch
    if now_pacific.hour >= 23:
        bucket = 'overnight'
    elif now_pacific.hour >= 21:
        bucket = 'tonight'
    elif now_pacific.hour >= 19:
        bucket = 'evening'
    else:
        bucket = 'day'

    return _openAIFetchForecastForURL_cached(url, bucket)


@st.cache_data(ttl=1800)
def _openAIFetchForecastForURL_cached(url, time_bucket):
    """Cached implementation — keyed on (url, time_bucket) so crossings trigger refresh."""
    openai_api_key = st.secrets["OpenAI_key"]
    if openai_api_key is None:
        raise ValueError("OpenAI API key is not set in environment variables.")

    response = cached_fetch_url(url)
    response.raise_for_status()

    vancouver_tz = pytz.timezone('America/Vancouver')
    now_pacific = datetime.now(vancouver_tz)
    now_str = now_pacific.strftime('%A %I:%M %p %Z')
    is_evening = now_pacific.hour >= 19  # 7 PM Pacific

    chat_gpt_msg = (
        "Make it short and just the table. "
        "Parse this forecast from marine weather canada (the section called \"Marine Forecast\") and "
        "extract a table with the following columns: time, wind speed, max wind speed, wind direction. "
        "wind speed is the first number in the wind speed string. max wind speed is the second. "
        "for example "
        "- if it says 5 to 15 knots, wind speed is 5 and max wind speed is 15. "
        "- If it says light winds, use a value of 3. "
        "- Make sure the Max (can be called Gust or gusting in the forecast) wind speed if not mentioned "
        "is the value of the wind speed, never less. "
        "Make it a CSV. "
        "The first few words describe the current conditions with a time of now and should be the 1st entry "
        "in the table. "
        f"\n\nCurrent local time is {now_str}. "
        "IMPORTANT time-aware rule for the FIRST row (current conditions): "
        "If the forecast uses a transition phrase like 'winds X becoming Y this evening' "
        "(or 'tonight' / 'overnight'), pick which phrase applies to right now: "
        f"{'it IS evening now (>= 7 PM Pacific), so use the AFTER-transition value (e.g. Y / light / strong).' if is_evening else 'it is NOT evening yet (< 7 PM Pacific), so use the BEFORE-transition value (e.g. X / 5-15).'} "
        "Apply the same logic for 'becoming ... tonight' (threshold 9 PM) and 'becoming ... overnight' (threshold 11 PM). "
        "The 'time' column for this first row should still be labeled with the current period (e.g. 'now'). "
        "\n\nHere's the forecast HTML:"
    )
    chat_gpt_msg = chat_gpt_msg + response.text

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
        return response.json()['choices'][0]['message']['content']
    else:
        print("Error:", response.status_code, response.text)
        return f"Error: {response.status_code} {response.text}"


@st.cache_data(ttl=1800)
def fetch_beautifulsoup_marine_forecast_for_url(url, title):
    """Fetch and parse marine forecast data from weather.gc.ca."""
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

        # Check for wind warnings
        wind_warning_in_effect = False
        strong_wind_warning_in_effect = False

        warning_banner = soup.find('div', id='warningBanner')
        if warning_banner:
            warning_text = warning_banner.text.lower().strip()
            strong_wind_warning_in_effect = "strong wind warning" in warning_text
            wind_warning_in_effect = "wind warning" in warning_text and not strong_wind_warning_in_effect

        forecast_content = soup.find('div', id='forecast-content')
        if not forecast_content:
            return {
                'error': True,
                'message': "Could not find forecast information",
                'details': "The forecast section was not found on the page",
                'suggestion': "The website structure might have changed or the service might be temporarily down"
            }

        # Parse issue date
        issue_dt = forecast_content.find('span', class_='text-info')
        issue_date = None
        if issue_dt:
            issue_date_str = issue_dt.text.strip()
            match = re.match(r'Issued\s+(\d{1,2}:\d{2}\s+(?:AM|PM)\s+\w+\s+\d{2}\s+\w+\s+\d{4})', issue_date_str)
            if match:
                try:
                    date_str = match.group(1)
                    date_str = date_str.replace('Issued', '').replace('&nbsp;', ' ').strip()
                    parts = date_str.split()
                    if len(parts) == 6:
                        time_part, ampm, tz, day, month, year = parts
                        formatted_date_str = f"{time_part} {ampm} {day} {month} {year}"
                        issue_date = datetime.strptime(formatted_date_str, '%I:%M %p %d %B %Y')
                        vancouver_tz = pytz.timezone('America/Vancouver')
                        if issue_date.tzinfo is None:
                            issue_date = vancouver_tz.localize(issue_date)
                        else:
                            issue_date = issue_date.astimezone(vancouver_tz)
                except ValueError as e:
                    return {
                        'error': True,
                        'message': f"Could not parse date: {date_str}",
                        'details': str(e),
                        'suggestion': "The date format might have changed"
                    }

        period_elem = forecast_content.find('span', class_='periodOfCoverage')
        period_coverage = period_elem.text.strip() if period_elem else ""

        warning_elem = forecast_content.find('span', class_='text-danger')
        warning = warning_elem.text.strip() if warning_elem else ""

        forecast_elem = forecast_content.find('span', class_='textSummary')
        forecast_text = forecast_elem.text.strip() if forecast_elem else ""

        if not forecast_text:
            return {
                'error': True,
                'message': "No forecast text found",
                'details': "The forecast text section was empty",
                'suggestion': "Please check back later for updated forecast"
            }

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


def parse_wind_forecast(forecast_text):
    """Parse wind forecast text into structured data."""
    segments = forecast_text.split('then')
    wind_data = []

    wind_pattern = r'(?P<direction>[A-Za-z]+)\s+(?:inflow|outflow)?\s*(?P<speed>\d+)\s*to\s*(?P<speed_high>\d+)'

    for segment in segments:
        segment = segment.strip()

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
    """Convert full wind direction names to standard abbreviations."""
    if pd.isna(direction) or not isinstance(direction, str):
        return direction

    direction = direction.replace(' outflow', '').strip().lower()

    direction_map = {
        'north': 'N', 'north northeasterly': 'NNE', 'northeast': 'NE',
        'east northeasterly': 'ENE', 'east': 'E', 'east southeasterly': 'ESE',
        'southeast': 'SE', 'south southeasterly': 'SSE', 'south': 'S',
        'south southwesterly': 'SSW', 'southwest': 'SW', 'west southwesterly': 'WSW',
        'west': 'W', 'west northwesterly': 'WNW', 'northwest': 'NW',
        'north northwesterly': 'NNW', 'northerly': 'N', 'northeasterly': 'NE',
        'easterly': 'E', 'southeasterly': 'SE', 'southerly': 'S',
        'southwesterly': 'SW', 'westerly': 'W', 'northwesterly': 'NW',
        'northerly outflow': 'N', 'southerly inflow': 'S', 'variable': 'V'
    }

    return direction_map.get(direction, direction)


def clean_wind_speed(x):
    """Clean and normalize wind speed values to float."""
    def extract_highest_integer(text):
        numbers_as_strings = re.findall(r'\d+', text)
        if not numbers_as_strings:
            return None
        return max(int(n) for n in numbers_as_strings)

    if pd.isna(x):
        return 0
    if isinstance(x, str) and '<' in x:
        return float(x.replace('<', ''))
    if isinstance(x, str) and 'light' in x:
        return float(2)
    if isinstance(x, str):
        nbr = extract_highest_integer(x)
        return float(nbr) if nbr else float(-1)
    return float(x)


def drawChartOfForecast(draw, df, title):
    """Draw a Plotly chart of wind forecast data."""
    import plotly.graph_objects as go

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['wind speed'],
        mode='lines+markers', name='Wind Speed',
        line=dict(color='blue', width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df['time'], y=df['max wind speed'],
        mode='lines+markers', name='Max Gust',
        line=dict(color='lightblue', width=2, dash='dash')
    ))

    fig.update_layout(
        title=f"Wind Forecast: {title}",
        yaxis_title="Knots",
        yaxis=dict(range=[0, 40]),
        hovermode="x unified"
    )
    fig.add_hline(y=15, line_dash="dot", line_color="red", annotation_text="15 knots")

    draw.plotly_chart(fig, width='stretch')


def display_summary_marine_forecast_for_url(draw=None, url='', title=''):
    """Display a summary of the marine forecast with current wind conditions."""
    df = None
    issue_date = None
    forecast = None
    subtitle = None

    with st.container(border=True):
        import time
        start_time = time.time()
        result = fetch_beautifulsoup_marine_forecast_for_url(url, title)
        elapsed_time = time.time() - start_time
        if elapsed_time > 0.1:
            draw.info(f"Forecast fetched in {elapsed_time:.2f} seconds")

        issue_date = result['issue_date']
        title = result['title']
        subtitle = result['subtitle']
        forecast = result['forecast']
        wind_warning = result['wind_warning']
        strong_wind_warning = result['strong_wind_warning']

        from utils import display_last_updated_badge
        draw.subheader(title)
        display_last_updated_badge(draw, issue_date, label="Issued")
        draw.write(url)

        if wind_warning:
            draw.badge("wind warning in effect", color="orange")
        if strong_wind_warning:
            draw.badge("wind warning in effect", color="red")

        start_time = time.time()
        chatgpt_forecast = openAIFetchForecastForURL(url=url)
        elapsed_time = time.time() - start_time
        if elapsed_time > 0.1:
            draw.info(f"Forecast fetched in {elapsed_time:.2f} seconds")

        if chatgpt_forecast:
            chatgpt_forecast = chatgpt_forecast.replace('```csv', '').replace('```', '')
            csv_stringio = io.StringIO(chatgpt_forecast)
            df = pd.read_csv(csv_stringio, sep=',', on_bad_lines='skip')
            df = df.dropna(how='all').reset_index(drop=True)
            df.columns = df.columns.str.strip().str.lower()

            try:
                if 'wind_direction' in df.columns:
                    df['wind direction'] = df['wind_direction'].str.lower().apply(standardize_wind_direction)
                elif 'wind direction' in df.columns:
                    df['wind direction'] = df['wind direction'].str.lower().apply(standardize_wind_direction)
                else:
                    df['wind direction'] = 'N/A'
            except Exception as e:
                print(f"Error applying standardization to wind direction column: {e}")
                draw.warning(f"Error applying standardization to wind direction column: {e}")

            if 'wind speed' in df.columns:
                df['wind speed'] = df['wind speed'].apply(clean_wind_speed)
            else:
                df['wind speed'] = 0

            if 'max wind speed' in df.columns:
                df['max wind speed'] = df['max wind speed'].apply(clean_wind_speed)
            else:
                df['max wind speed'] = 0

            if 'time' not in df.columns:
                df['time'] = 'N/A'

            if not df.empty:
                col1, col2, col3, col4 = draw.columns(4)
                col1.badge(f" {df['time'].iloc[0]}")
                col2.metric("Wind Speed", df['wind speed'].iloc[0])
                col3.metric("Wind High", df['max wind speed'].iloc[0])

                wind_direction = df['wind direction'].iloc[0]
                col4.metric("Direction", wind_direction)
                col1.markdown(create_arrow_html(wind_direction, df['wind speed'].iloc[0]),
                              unsafe_allow_html=True)

                if len(df) > 1:
                    col21, col22, col23, col24 = draw.columns(4)
                    col21.badge(f"{df['time'].iloc[1]}")
                    col22.metric("Wind Speed", df['wind speed'].iloc[1])
                    col23.metric("Wind High", df['max wind speed'].iloc[1])

                    wind_direction = df['wind direction'].iloc[1]
                    col24.metric("Direction", wind_direction)
                    col21.markdown(create_arrow_html(wind_direction, df['wind speed'].iloc[1]),
                                   unsafe_allow_html=True)

    return df, issue_date, forecast, subtitle


def display_table_marine_forecast_for_url(draw=None, url='', title='', df=None):
    draw.badge("chatGPT forecast table")
    draw.dataframe(df)


def display_text_marine_forecast_for_url(draw=None, url='', title='', forecast=None, issue_date=None, subtitle=None):
    draw.write(forecast)
    draw.badge("chatGPT forecast")

    if issue_date:
        draw.caption(f"({issue_date.strftime('%Y-%m-%d %H:%M %Z')})")
        draw.subheader(title)
        draw.write(subtitle)

        if forecast:
            bold_forecast = re.sub(r'(\d+(?:\.\d+)?)', r'**\1**', forecast)
            draw.markdown(
                f'<div style="padding: 1em; border-radius: 5px; background-color: #f0f2f6;">{bold_forecast}</div>',
                unsafe_allow_html=True
            )
        else:
            draw.error("Unable to fetch " + title + " marine forecast")

        draw.badge("BeautifulSoup forecast")


def display_marine_forecast_for_url(draw=None, url='', title=''):
    if draw is None:
        draw = st

    df, issue_date, forecast, subtitle = display_summary_marine_forecast_for_url(draw, url, title)
    drawChartOfForecast(draw, df, title)
    with draw.expander("Forecast Table"):
        display_table_marine_forecast_for_url(draw=st, url=url, title=title, df=df)
    with draw.expander("Forecast Raw Text"):
        display_text_marine_forecast_for_url(draw=st, url=url, title=title, forecast=forecast, issue_date=issue_date, subtitle=subtitle)
