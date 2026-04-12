import io
import re

import streamlit as st
import pandas as pd
import numpy as np
import pytz
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from utils import cached_fetch_url
from fetch_weather import fetch_from_open_weather
from fetch_forecast import (
    fetch_beautifulsoup_marine_forecast_for_url,
    openAIFetchForecastForURL,
    clean_wind_speed,
)
from fetch_tides import beautifulSoupFetchTidesForURL, process_tide_data, parse_tide_datetime, extract_meters

# --- Thresholds ---
WIND_GO = 10        # knots — ideal
WIND_CAUTION = 15   # knots — manageable but exposed in a 14ft RIB
WAVE_GO = 0.51      # meters (51 cm)
WAVE_CAUTION = 0.75 # meters
PRECIP_GO = 0.5     # mm in next 24h — light drizzle OK
PRECIP_CAUTION = 2.0
TIDE_NOGO = 1.5     # meters — very low, tough to launch at Horseshoe Bay
TIDE_CAUTION = 2.5  # meters

VANCOUVER_LAT = 49.32
VANCOUVER_LON = -123.16
URL_HOWE_SOUND = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400'


def _status(value, go_threshold, caution_threshold, higher_is_worse=True):
    if higher_is_worse:
        if value <= go_threshold:
            return 'go'
        if value <= caution_threshold:
            return 'caution'
        return 'nogo'
    else:
        if value >= caution_threshold:
            return 'go'
        if value >= go_threshold:
            return 'caution'
        return 'nogo'


_BADGE = {'go': 'green', 'caution': 'orange', 'nogo': 'red'}
_ICON = {'go': '✅', 'caution': '⚠️', 'nogo': '🔴'}


def _fetch_buoy_wind_wave(buoy_id='46146'):
    """Lightweight scrape of Halibut Bank buoy for wind & wave."""
    url = (
        'https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html'
        f'?mapID=02&siteID=14305&stationID={buoy_id}'
    )
    try:
        res = cached_fetch_url(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        table = soup.find('table', class_='table')
        if not table or not table.tbody:
            return None, None
        rows = table.tbody.find_all('tr')

        wind_text = rows[0].find_all('td')[0].text.strip()
        winds = re.findall(r'\d+', wind_text)
        max_wind = max(int(w) for w in winds) if winds else None

        wave_height = None
        if buoy_id == '46146' and len(rows) > 1:
            wave_text = rows[1].find_all('td')[0].text.strip()
            wave_nums = re.findall(r'[-+]?\d*\.\d+|\d+', wave_text)
            wave_height = float(wave_nums[0]) if wave_nums else None

        return max_wind, wave_height
    except Exception as e:
        print(f"Go/NoGo buoy fetch error: {e}")
        return None, None


def _get_current_tide_height():
    """Estimate current tide height from BeautifulSoup extremes."""
    try:
        data = beautifulSoupFetchTidesForURL("https://www.tides.gc.ca/en/stations/07795")
        if not data or not data.get('data'):
            return None, None

        tide_df = process_tide_data(data)
        tide_df = tide_df.rename(columns={'Time (PDT)& Date': 'datetime'})
        tide_df['datetime'] = tide_df['datetime'].apply(parse_tide_datetime)
        tide_df['Height'] = tide_df['Height'].astype(str).apply(extract_meters)
        tide_df = tide_df.dropna(subset=['Height', 'datetime'])

        if len(tide_df) < 2:
            return None, None

        vancouver_tz = pytz.timezone('America/Vancouver')
        now = datetime.now(vancouver_tz)

        x_ts = tide_df['datetime'].apply(lambda dt: dt.timestamp()).values
        current_h = float(np.interp(now.timestamp(), x_ts, tide_df['Height'].values))

        direction = ""
        try:
            nxt = tide_df[tide_df['datetime'] > now].iloc[0]
            direction = "Rising" if nxt['Height'] > current_h else "Falling"
        except (IndexError, KeyError):
            pass

        return current_h, direction
    except Exception as e:
        print(f"Go/NoGo tide error: {e}")
        return None, None


def _analyze_5day_windows(weather_data):
    """Find boating windows in the hourly forecast (wind + rain per half-day)."""
    if not weather_data or not weather_data.hourly_forecast:
        return []

    vancouver_tz = pytz.timezone('America/Vancouver')
    today = datetime.now(vancouver_tz).replace(hour=0, minute=0, second=0, microsecond=0)
    windows = []

    for day_offset in range(1, 6):
        day = today + timedelta(days=day_offset)
        day_name = day.strftime('%a %b %d')

        for period_name, center_h in [('8AM', 8), ('Noon', 12), ('4PM', 16)]:
            # Find the forecast entry closest to this hour (within +/- 1.5h)
            target = day.replace(hour=center_h)
            items = [
                item for item in weather_data.hourly_forecast
                if abs((datetime.fromtimestamp(item['dt']).astimezone(vancouver_tz) - target).total_seconds()) <= 5400
            ]
            if not items:
                continue

            max_wind = max(
                item['wind'].get('gust', item['wind']['speed']) * 1.94384
                for item in items
            )
            total_rain = sum(item.get('rain', {}).get('3h', 0) for item in items)

            if max_wind > WIND_CAUTION or total_rain > PRECIP_CAUTION:
                status = 'nogo'
            elif max_wind > WIND_GO or total_rain > PRECIP_GO:
                status = 'caution'
            else:
                status = 'go'

            windows.append({
                'label': f"{day_name} {period_name}",
                'status': status,
                'wind': max_wind,
                'rain': total_rain,
            })

    return windows


def display_gonogo_sidebar():
    """Render the Go/No-Go assessment in st.sidebar."""

    st.sidebar.markdown("---")
    st.sidebar.subheader("Go / No-Go")

    factors = {}
    weather = None

    # 1. Current weather (wind + precipitation)
    try:
        api_key = st.secrets["openweather_api_key"]
        weather = fetch_from_open_weather(VANCOUVER_LAT, VANCOUVER_LON, api_key)
        if weather:
            wind_kts = weather.wind_speed_now * 1.94384
            factors['wind_now'] = {
                'status': _status(wind_kts, WIND_GO, WIND_CAUTION),
                'label': f"Wind Now: {wind_kts:.0f}kts",
            }
            factors['precip'] = {
                'status': _status(weather.next_24_hours_precipitation, PRECIP_GO, PRECIP_CAUTION),
                'label': f"Rain 24h: {weather.next_24_hours_precipitation:.1f}mm",
            }
    except Exception as e:
        print(f"Go/NoGo weather error: {e}")

    # 2. Marine forecast — Howe Sound warnings + parsed wind
    try:
        forecast = fetch_beautifulsoup_marine_forecast_for_url(URL_HOWE_SOUND, "Howe Sound")
        if forecast and not forecast.get('error'):
            if forecast.get('strong_wind_warning'):
                factors['warnings'] = {'status': 'nogo', 'label': 'Strong Wind Warning!'}
            elif forecast.get('wind_warning'):
                factors['warnings'] = {'status': 'caution', 'label': 'Wind Warning'}
            else:
                factors['warnings'] = {'status': 'go', 'label': 'No Warnings'}

            # GPT-parsed Howe Sound wind forecast — use first 2 rows (now + next period)
            try:
                csv_text = openAIFetchForecastForURL(url=URL_HOWE_SOUND)
                if csv_text:
                    csv_clean = csv_text.replace('```csv', '').replace('```', '')
                    df = pd.read_csv(io.StringIO(csv_clean), sep=',', on_bad_lines='skip')
                    df = df.dropna(how='all').reset_index(drop=True)
                    df.columns = df.columns.str.strip().str.lower()

                    if 'max wind speed' in df.columns:
                        df['max wind speed'] = df['max wind speed'].apply(clean_wind_speed)
                        # Only consider current + next period for go/nogo (not the whole forecast)
                        current_wind = df['max wind speed'].iloc[:2].max() if len(df) >= 2 else df['max wind speed'].iloc[0]
                        time_label = df['time'].iloc[0] if 'time' in df.columns else "now"
                        factors['howe_wind'] = {
                            'status': _status(current_wind, WIND_GO, WIND_CAUTION),
                            'label': f"Howe Sound: {current_wind:.0f}kts ({time_label})",
                        }
            except Exception:
                pass
    except Exception as e:
        print(f"Go/NoGo forecast error: {e}")

    # 3. Halibut Bank buoy — wind + waves
    try:
        buoy_wind, buoy_wave = _fetch_buoy_wind_wave('46146')
        if buoy_wind is not None:
            factors['buoy_wind'] = {
                'status': _status(buoy_wind, WIND_GO, WIND_CAUTION),
                'label': f"Halibut Bank: {buoy_wind}kts",
            }
        if buoy_wave is not None:
            wave_cm = buoy_wave * 100
            factors['waves'] = {
                'status': _status(buoy_wave, WAVE_GO, WAVE_CAUTION),
                'label': f"Waves: {wave_cm:.0f}cm",
            }
    except Exception as e:
        print(f"Go/NoGo buoy error: {e}")

    # 4. Tide — launch feasibility at Horseshoe Bay
    try:
        tide_h, tide_dir = _get_current_tide_height()
        if tide_h is not None:
            suffix = f" ({tide_dir})" if tide_dir else ""
            factors['tide'] = {
                'status': _status(tide_h, TIDE_NOGO, TIDE_CAUTION, higher_is_worse=False),
                'label': f"Tide: {tide_h:.1f}m{suffix}",
            }
    except Exception as e:
        print(f"Go/NoGo tide error: {e}")

    # --- Overall verdict ---
    if not factors:
        st.sidebar.warning("Unable to assess conditions")
        return

    statuses = [f['status'] for f in factors.values()]
    if 'nogo' in statuses:
        overall, overall_label = 'nogo', 'NO-GO'
    elif 'caution' in statuses:
        overall, overall_label = 'caution', 'CAUTION'
    else:
        overall, overall_label = 'go', 'GO'

    # Compact summary — always visible
    nogo_count = statuses.count('nogo')
    caution_count = statuses.count('caution')
    summary_parts = []
    if nogo_count:
        summary_parts.append(f"{nogo_count} red")
    if caution_count:
        summary_parts.append(f"{caution_count} amber")
    summary_note = f"  ({', '.join(summary_parts)})" if summary_parts else ""

    st.sidebar.badge(overall_label, color=_BADGE[overall])
    st.sidebar.caption(f"{len(factors)} factors checked{summary_note}")

    # Expandable details — current conditions
    with st.sidebar.expander("Current Conditions"):
        for f in factors.values():
            st.caption(f"{_ICON[f['status']]} {f['label']}")

        st.markdown(
            "**Thresholds:** "
            f"Wind GO < {WIND_GO}kts, CAUTION < {WIND_CAUTION}kts | "
            f"Waves GO < {int(WAVE_GO * 100)}cm | "
            f"Rain GO < {PRECIP_GO}mm/24h | "
            f"Tide NO-GO < {TIDE_NOGO}m"
        )

    # Expandable details — 5-day windows
    if weather:
        try:
            windows = _analyze_5day_windows(weather)
            if windows:
                with st.sidebar.expander("5-Day Windows"):
                    for w in windows:
                        detail = f"{w['wind']:.0f}kts"
                        if w['rain'] > 0:
                            detail += f", {w['rain']:.1f}mm"
                        st.caption(f"{_ICON[w['status']]} {w['label']} ({detail})")
        except Exception as e:
            print(f"Go/NoGo 5-day error: {e}")
