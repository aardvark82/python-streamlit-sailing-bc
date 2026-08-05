import io
import re

import streamlit as st
import pandas as pd
import numpy as np
import pytz
import plotly.graph_objects as go
from datetime import datetime, timedelta
from bs4 import BeautifulSoup

from utils import cached_fetch_url, cached_fetch_url_live, cached_fetch_url_buoy
from fetch_weather import fetch_from_open_weather, get_wind_direction
from wind_utils import direction_arrow, direction_degrees
from fetch_forecast import (
    fetch_beautifulsoup_marine_forecast_for_url,
    openAIFetchForecastForURL,
    clean_wind_speed,
)
from fetch_tides import (
    beautifulSoupFetchTidesForURL,
    fetch_tide_extremes_selenium,
    fetch_iwls_tide_extremes_pt_atkinson,
    process_tide_data,
    parse_tide_datetime,
    extract_meters,
)

# --- Thresholds ---
WIND_GO = 10        # knots — ideal
WIND_CAUTION = 15   # knots — manageable but exposed in a 14ft RIB
WAVE_GO = 0.51      # meters (51 cm)
WAVE_CAUTION = 0.75 # meters
PRECIP_GO = 0.5     # mm — light drizzle OK
PRECIP_CAUTION = 2.0
TIDE_NOGO = 2.0     # meters — below this, can't launch at Horseshoe Bay (learned the hard way at 80cm)
TIDE_CAUTION = 2.5  # meters — marginal; doable but tight

_SEVERITY = {'nogo': 0, 'caution': 1, 'go': 2}   # sort order for Current Conditions

VANCOUVER_LAT = 49.32
VANCOUVER_LON = -123.16
URL_HOWE_SOUND = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=06400'
URL_SOUTH_NANAIMO = 'https://weather.gc.ca/marine/forecast_e.html?mapID=02&siteID=14305'


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

# Short card titles for the Current Conditions metric cards (keyed by factor id)
_CARD_TITLES = {
    'tide': 'Tide',
    'howe_current': 'Howe Sound Forecast',
    'warnings': 'Marine Warnings',
}

# Fixed display order for the Current Conditions cards. Row 1 = tide /
# wind-vs-tide / temperature; row 2 = the two marine forecasts + rain;
# the "Next" forecast period trails at the end. Keys not listed sort last.
_CARD_ORDER = [
    'tide', 'rain', 'temp',
    'howe_current', 'south_nanaimo', 'wind_vs_tide',
    'howe_next', 'south_nanaimo_next', 'wind_vs_tide_3h',
]


def _fmt_forecast_range(r):
    """Compact wind string for a parsed marine-forecast row, e.g. '↘ 10-15kts'.
    Leads with a downwind arrow when the direction is known."""
    spd, gust, d = r.get('wind_speed'), r.get('max_wind_speed'), r.get('direction')
    arrow = direction_arrow(d)
    dtxt = f"{arrow} " if arrow else ""
    if spd is not None and gust is not None:
        return f"{dtxt}{spd:.0f}-{gust:.0f}kts"
    if gust is not None:
        return f"{dtxt}{gust:.0f}kts"
    return f"{dtxt}n/a"


def _forecast_wind_help(r, area):
    """Tooltip for a forecast card: area + cardinal direction when known."""
    d = r.get('direction')
    return f"EC marine forecast — {area}" + (f" · wind from {d}" if d else "")


def _fcst_status(gust):
    """Forecast-wind status for the coloured cards: green < 10, orange 10-20,
    red 20+ kts (thresholds asked for on the forecast cards)."""
    return _status(gust, 10, 20) if gust is not None else 'go'


def _fcst_icon(gust):
    """Status icon (✅/⚠️/🔴) for a forecast card, keyed on the gust."""
    return _ICON[_fcst_status(gust)]
_COLOR_MAP = {'go': '#2ecc71', 'caution': '#f39c12', 'nogo': '#e74c3c'}
_NUMERIC = {'go': 1, 'caution': 0.5, 'nogo': 0}


def _fetch_buoy_wind_wave(buoy_id='46304'):
    """Lightweight scrape of Environment Canada offshore buoy for wind & wave."""
    url = (
        'https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html'
        f'?mapID=02&siteID=14305&stationID={buoy_id}'
    )
    try:
        # Buoys publish hourly — 15-min cache (vs the old 3-min live cache)
        # cuts default-page scrape frequency ~5×.
        res = cached_fetch_url_buoy(url)
        soup = BeautifulSoup(res.content, 'html.parser')
        table = soup.find('table', class_='table')
        if not table or not table.tbody:
            return None, None
        rows = table.tbody.find_all('tr')

        wind_text = rows[0].find_all('td')[0].text.strip()
        winds = re.findall(r'\d+', wind_text)
        max_wind = max(int(w) for w in winds) if winds else None
        # Leading token is the cardinal direction (e.g. "NW 15") when present.
        parts = wind_text.split()
        direction = parts[0] if parts and not parts[0][0].isdigit() else None

        wave_height = None
        if buoy_id in ('46146', '46304') and len(rows) > 1:
            wave_text = rows[1].find_all('td')[0].text.strip()
            wave_nums = re.findall(r'[-+]?\d*\.\d+|\d+', wave_text)
            wave_height = float(wave_nums[0]) if wave_nums else None

        return max_wind, wave_height, direction
    except Exception as e:
        print(f"Go/NoGo buoy fetch error: {e}")
        return None, None, None


def _get_tide_data():
    """Fetch tide extremes and build interpolation arrays.
    Returns (extremes_df, x_timestamps, y_heights) or (None, None, None).
    extremes_df has columns: datetime, Height, type (high/low).

    Tries the Selenium-based CSV source first (same pipeline as the Tides page),
    falls back to the BeautifulSoup scraper if Selenium is unavailable."""
    data = None

    # Preferred: DFO IWLS REST API — the SAME source the Tides page uses
    # (reliable on Streamlit Cloud, where Selenium can't run). This is why
    # the sidebar Tides chart worked but Go/No-Go said 'tide unavailable'.
    try:
        extremes_list = fetch_iwls_tide_extremes_pt_atkinson()
        if extremes_list:
            data = {'data': [
                {'height': e['Height'], 'time': e['Time (PDT)& Date'], 'type': e['type']}
                for e in extremes_list
            ]}
    except Exception as e:
        print(f"Go/NoGo IWLS tide source failed: {e}")

    # Fallback: Selenium-fetched CSV (only works locally with Chrome)
    if not data or not data.get('data'):
        try:
            extremes_list = fetch_tide_extremes_selenium()
            if extremes_list:
                data = {'data': [
                    {'height': e['Height'], 'time': e['Time (PDT)& Date'], 'type': e['type']}
                    for e in extremes_list
                ]}
        except Exception as e:
            print(f"Go/NoGo Selenium tide source failed: {e}")

    # Last resort: BeautifulSoup scraper
    if not data or not data.get('data'):
        try:
            data = beautifulSoupFetchTidesForURL("https://www.tides.gc.ca/en/stations/07795")
        except Exception as e:
            print(f"Go/NoGo BeautifulSoup tide source failed: {e}")
            return None, None, None

    if not data or not data.get('data'):
        return None, None, None

    try:
        tide_df = process_tide_data(data)
        tide_df = tide_df.rename(columns={'Time (PDT)& Date': 'datetime'})
        tide_df['datetime'] = tide_df['datetime'].apply(parse_tide_datetime)
        tide_df['Height'] = tide_df['Height'].astype(str).apply(extract_meters)
        tide_df = tide_df.dropna(subset=['Height', 'datetime'])

        if len(tide_df) < 2:
            return None, None, None

        x_ts = tide_df['datetime'].apply(lambda dt: dt.timestamp()).values
        y_h = tide_df['Height'].values
        return tide_df, x_ts, y_h
    except Exception as e:
        print(f"Go/NoGo tide data error: {e}")
        return None, None, None


def _fmt_tide_time(dt):
    """Format tide time in 24-hour format, e.g. '08:00', '13:45'."""
    return f"{dt.hour:02d}:{dt.minute:02d}"


def _nearest_tides_in_window(tide_df, window_start, window_end):
    """Find low and high tide times within a time window.
    Returns dict with 'low' and 'high' keys, each (time_str, height) or None."""
    if tide_df is None or tide_df.empty:
        return {'low': None, 'high': None}

    mask = (tide_df['datetime'] >= window_start) & (tide_df['datetime'] <= window_end)
    window_tides = tide_df[mask]

    result = {'low': None, 'high': None}
    if window_tides.empty:
        return result

    low_row = window_tides.loc[window_tides['Height'].idxmin()]
    high_row = window_tides.loc[window_tides['Height'].idxmax()]

    # Only label as low/high if there's meaningful difference
    if low_row.name != high_row.name:
        result['low'] = (_fmt_tide_time(low_row['datetime']), low_row['Height'])
        result['high'] = (_fmt_tide_time(high_row['datetime']), high_row['Height'])
    else:
        # Single tide point in window
        row = low_row
        label = 'low' if row['Height'] < 2.5 else 'high'
        result[label] = (_fmt_tide_time(row['datetime']), row['Height'])

    return result


def _tide_at(x_ts, y_h, target_dt):
    """Interpolate tide height at a specific datetime. Returns float or None."""
    if x_ts is None or y_h is None:
        return None
    ts = target_dt.timestamp()
    if ts < x_ts[0] or ts > x_ts[-1]:
        return None  # outside data range
    return float(np.interp(ts, x_ts, y_h))


def _get_current_tide_height():
    """Estimate current tide height from BeautifulSoup extremes."""
    _, x_ts, y_h = _get_tide_data()
    if x_ts is None:
        return None, None

    vancouver_tz = pytz.timezone('America/Vancouver')
    now = datetime.now(vancouver_tz)
    current_h = _tide_at(x_ts, y_h, now)
    if current_h is None:
        return None, None

    # Determine rising/falling by checking height 30 min from now
    future_h = _tide_at(x_ts, y_h, now + timedelta(minutes=30))
    direction = ""
    if future_h is not None:
        direction = "Rising" if future_h > current_h else "Falling"

    return current_h, direction


def _flood_at(x_ts, y_h, target_dt):
    """True if the tide is flooding (rising, setting INTO Howe Sound) at
    target_dt, False if ebbing, None if it can't be determined."""
    h0 = _tide_at(x_ts, y_h, target_dt)
    h1 = _tide_at(x_ts, y_h, target_dt + timedelta(minutes=30))
    if h0 is None or h1 is None:
        return None
    return h1 > h0


def _near_slack(x_ts, target_dt, within_min=60):
    """True if target_dt is within `within_min` of a tide extreme (slack water).
    x_ts holds the extreme (high/low) timestamps."""
    if x_ts is None or len(x_ts) == 0:
        return False
    ts = target_dt.timestamp()
    return min(abs(ts - t) for t in x_ts) <= within_min * 60


def _classify_wind_tide(is_flood, wind_deg, wind_kts, near_slack):
    """Five-state wind-vs-tide readout for Howe Sound (N–S axis: flood sets
    ~N/into the sound, ebb sets ~S/out). Returns (label, status).

      Calm       — within 1 h of slack and wind < 5 kts
      Aligned N  — flood + southerly wind (both driving north)
      Aligned S  — ebb + northerly wind (both driving south)
      Light chop — wind opposes the tide, <= 10 kts
      Heavy chop — wind opposes the tide, 11+ kts
    """
    if is_flood is None or wind_deg is None or wind_kts is None:
        return None, None
    if near_slack and wind_kts < 5:
        return "Calm", 'go'
    from_south = 90 <= wind_deg <= 270      # wind FROM the southerly half → blows ~N
    if is_flood and from_south:
        return "Aligned N", 'go'
    if (not is_flood) and (not from_south):
        return "Aligned S", 'go'
    # wind opposes the tidal stream → chop
    if wind_kts <= 10:
        return "Light chop", 'go'
    return "Heavy chop", 'caution'


def _gather_current_factors():
    """Gather all current condition factors. Returns (factors dict, weather_data)."""
    factors = {}
    weather = None
    tide_dir_now = None     # "Rising" / "Falling"
    wind_deg_now = None     # degrees the wind is coming FROM (OpenWeather, W Van)
    pam_deg_now = None      # degrees the wind comes FROM at Pam Rocks (entrance)
    pam_kts_now = None      # wind speed (kts) at Pam Rocks

    # 0. Tide — collected first so it displays at the top (mission-critical for launch)
    try:
        tide_h, tide_dir = _get_current_tide_height()
        tide_dir_now = tide_dir
        if tide_h is not None:
            arrow = ""
            if tide_dir == "Rising":
                arrow = " ↑"
            elif tide_dir == "Falling":
                arrow = " ↓"
            dir_text = f" {tide_dir}" if tide_dir else ""

            # Warning badge takes priority over the "2m minimum" reminder:
            # - Below 2m and falling = will only get worse, can't launch
            # - Below 1m and rising = currently way too shallow, even though improving
            if tide_h < TIDE_NOGO and tide_dir == "Falling":
                tide_badge = {'text': '⚠️ Low & falling', 'color': 'red'}
            elif tide_h < 1.0 and tide_dir == "Rising":
                tide_badge = {'text': '⚠️ Very low', 'color': 'red'}
            else:
                tide_badge = {'text': '2m minimum', 'color': 'gray'}

            factors['tide'] = {
                'status': _status(tide_h, TIDE_NOGO, TIDE_CAUTION, higher_is_worse=False),
                'label': f"Tide: {tide_h:.2f}m{arrow}{dir_text}",
                'value': tide_h,
                'page': 'Tides',
                'badge': tide_badge,
            }
        else:
            factors['tide'] = {
                'status': 'caution',
                'label': "Tide: data unavailable",
                'page': 'Tides',
                'badge': {'text': '2m minimum', 'color': 'gray'},
            }
    except Exception as e:
        print(f"Go/NoGo tide error: {e}")
        factors['tide'] = {
            'status': 'caution',
            'label': f"Tide: error ({e})",
            'page': 'Tides',
        }

    # 1. Current weather (wind + precipitation)
    try:
        api_key = st.secrets["openweather_api_key"]
        weather = fetch_from_open_weather(VANCOUVER_LAT, VANCOUVER_LON, api_key)
        if weather:
            # Kept for the Wind vs Tide fallback when Pam Rocks is unavailable;
            # the West-Van OpenWeather wind is no longer shown as its own card.
            wind_deg_now = weather.wind_direction_now

            # Temperature — context only; informational card (no verdict impact)
            factors['temp'] = {
                'status': 'go',
                'informational': True,
                'card_title': '🌡️ Temperature',
                'label': f"{weather.temperature:.0f}°C",
            }
            # Rain over the next 6 hours (OpenWeather is 3-hourly → first 2 slots)
            rain_6h = sum(
                item.get('rain', {}).get('3h', 0)
                for item in (weather.hourly_forecast or [])[:2]
            )
            # Rain colour: green at 0 mm, orange up to 10 mm, red above 10 mm.
            if rain_6h < 0.05:
                rain_status = 'go'
            elif rain_6h <= 10:
                rain_status = 'caution'
            else:
                rain_status = 'nogo'
            factors['rain'] = {
                'status': rain_status,
                'informational': True,
                'card_title': f"{_ICON[rain_status]} Rain (6 hours)",
                'label': f"{rain_6h:.1f}mm",
                'help': "Next 6 hours — OpenWeather (West Vancouver)",
            }
    except Exception as e:
        print(f"Go/NoGo weather error: {e}")

    # 2. Marine forecast — Howe Sound warnings. Kept OUT of the card grid
    #    (hide_card) but still drives the verdict + red-reason pills.
    try:
        forecast = fetch_beautifulsoup_marine_forecast_for_url(URL_HOWE_SOUND, "Howe Sound")
        if forecast and not forecast.get('error'):
            if forecast.get('strong_wind_warning'):
                factors['warnings'] = {'status': 'nogo', 'label': 'Strong Wind Warning!', 'page': 'Marine_Forecast', 'hide_card': True}
            elif forecast.get('wind_warning'):
                factors['warnings'] = {'status': 'caution', 'label': 'Wind Warning', 'page': 'Marine_Forecast', 'hide_card': True}
            else:
                factors['warnings'] = {'status': 'go', 'label': 'No Warnings', 'page': 'Marine_Forecast', 'hide_card': True}
    except Exception as e:
        print(f"Go/NoGo forecast error: {e}")

    # 2b. Howe Sound marine forecast wind — current period (drives the verdict)
    #     + next period (informational). Both carry wind direction when parsed.
    try:
        rows = _get_marine_forecast_rows(URL_HOWE_SOUND)
        if rows:
            r = rows[0]
            # The "current conditions" row often omits direction (e.g. "light
            # winds") — borrow the next period's direction so the arrow shows.
            if not r.get('direction') and len(rows) > 1:
                r = {**r, 'direction': rows[1].get('direction')}
            gust = r.get('max_wind_speed')
            period = r.get('time') or 'now'
            period_disp = period[:1].upper() + period[1:]
            factors['howe_current'] = {
                'status': _status(gust, WIND_GO, WIND_CAUTION) if gust is not None else 'caution',
                'card_title': f"Howe Sound Forecast ({period_disp})",
                'label': f"Howe Sound: {_fmt_forecast_range(r)}",
                'help': _forecast_wind_help(r, "Howe Sound (morning & afternoon)"),
                'page': 'Marine_Forecast',
            }
            if len(rows) > 1:
                r2 = rows[1]
                gust2 = r2.get('max_wind_speed')
                factors['howe_next'] = {
                    'status': _fcst_status(gust2),
                    'informational': True,
                    'card_title': f"{_fcst_icon(gust2)} Next · Howe Sound ({r2.get('time', '')})",
                    'label': _fmt_forecast_range(r2),
                    'help': _forecast_wind_help(r2, "Howe Sound"),
                    'page': 'Marine_Forecast',
                }
    except Exception as e:
        print(f"Go/NoGo Howe Sound forecast error: {e}")

    # 2c. Strait of Georgia, south of Nanaimo — informational marine forecast
    #     (broader area context; shown but not driving the verdict).
    try:
        srows = _get_marine_forecast_rows(URL_SOUTH_NANAIMO)
        if srows:
            sr = srows[0]
            factors['south_nanaimo'] = {
                'status': _fcst_status(sr.get('max_wind_speed')),
                'informational': True,
                'card_title': f"{_fcst_icon(sr.get('max_wind_speed'))} S. of Nanaimo ({sr.get('time', 'now')})",
                'label': _fmt_forecast_range(sr),
                'help': _forecast_wind_help(sr, "Strait of Georgia, south of Nanaimo"),
                'page': 'Marine_Forecast',
            }
            if len(srows) > 1:
                sr2 = srows[1]
                factors['south_nanaimo_next'] = {
                    'status': _fcst_status(sr2.get('max_wind_speed')),
                    'informational': True,
                    'card_title': f"{_fcst_icon(sr2.get('max_wind_speed'))} S. of Nanaimo ({sr2.get('time', 'next')})",
                    'label': _fmt_forecast_range(sr2),
                    'help': _forecast_wind_help(sr2, "Strait of Georgia, south of Nanaimo"),
                    'page': 'Marine_Forecast',
                }
    except Exception as e:
        print(f"Go/NoGo South of Nanaimo forecast error: {e}")

    # 3. Pam Rocks buoy (WAS) — observed entrance wind. No card (removed from
    #    the grid); kept as a hidden verdict factor and feeds Wind vs Tide.
    try:
        pam_wind, _, pam_dir = _fetch_buoy_wind_wave('WAS')
        pam_kts_now = pam_wind
        pam_deg_now = direction_degrees(pam_dir)
        if pam_wind is not None:
            dtxt = f"{pam_dir} " if pam_dir else ""
            factors['pam_wind'] = {
                'status': _status(pam_wind, WIND_GO, WIND_CAUTION),
                'label': f"Pam Rocks: {dtxt}{pam_wind}kts",
                'value': pam_wind,
                'page': 'Marine_Forecast',
                'hide_card': True,
            }
    except Exception as e:
        print(f"Go/NoGo Pam Rocks error: {e}")

    # 4. English Bay buoy (46304) — observed wind + waves. No cards (waves moved
    #    to the map); kept as hidden verdict factors so big wind/seas still count.
    try:
        buoy_wind, buoy_wave, bay_dir = _fetch_buoy_wind_wave('46304')
        if buoy_wind is not None:
            dtxt = f"{bay_dir} " if bay_dir else ""
            factors['buoy_wind'] = {
                'status': _status(buoy_wind, WIND_GO, WIND_CAUTION),
                'label': f"English Bay: {dtxt}{buoy_wind}kts",
                'value': buoy_wind,
                'page': 'English_Bay',
                'hide_card': True,
            }
        if buoy_wave is not None:
            wave_cm = buoy_wave * 100
            factors['waves'] = {
                'status': _status(buoy_wave, WAVE_GO, WAVE_CAUTION),
                'label': f"Waves: {wave_cm:.0f}cm",
                'value': buoy_wave,
                'page': 'English_Bay',
                'hide_card': True,
            }
    except Exception as e:
        print(f"Go/NoGo buoy error: {e}")

    # Wind vs Tide — 5-state readout for NOW (Pam Rocks wind vs the current
    # flood/ebb + slack) plus a 3-HOUR FORECAST (OpenWeather 3h wind vs the
    # interpolated tide in 3 h). See _classify_wind_tide for the states.
    try:
        _tdf, x_ts, y_h = _get_tide_data()
        van_now = datetime.now(pytz.timezone('America/Vancouver'))

        def _wvt_card(is_flood, w_deg, w_kts, near_slack, title, informational, help_txt):
            label, status = _classify_wind_tide(is_flood, w_deg, w_kts, near_slack)
            if label is None:
                label, status = "data unavailable", 'go'
                arrow = ''
            else:
                arrow = direction_arrow(get_wind_direction(w_deg)) if w_deg is not None else ''
            value = f"{arrow} {label}".strip()
            card = {'status': status, 'label': value, 'help': help_txt, 'page': 'Tides'}
            if informational:
                card['informational'] = True
                card['card_title'] = f"{_ICON[status]} {title}"
            else:
                card['card_title'] = title
            return card

        # NOW — Pam Rocks wind (fallback to the local OpenWeather reading).
        is_flood_now = (tide_dir_now == "Rising") if tide_dir_now else None
        deg_now = pam_deg_now if pam_deg_now is not None else wind_deg_now
        kts_now = pam_kts_now if pam_kts_now is not None else \
            (weather.wind_speed_now * 1.94384 if weather else None)
        src_now = "Pam Rocks" if pam_deg_now is not None else "local"
        factors['wind_vs_tide'] = _wvt_card(
            is_flood_now, deg_now, kts_now, _near_slack(x_ts, van_now),
            "Wind vs Tide (Now)", False,
            f"{src_now} wind vs current Howe Sound tide (flood/ebb + slack)",
        )

        # 3-HOUR FORECAST — OpenWeather 3 h wind vs the interpolated tide in 3 h.
        if weather is not None:
            t3 = van_now + timedelta(hours=3)
            factors['wind_vs_tide_3h'] = _wvt_card(
                _flood_at(x_ts, y_h, t3),
                weather.wind_direction_3h, weather.wind_speed_3h * 1.94384,
                _near_slack(x_ts, t3),
                "Wind vs Tide (3h)", True,
                "OpenWeather 3 h wind vs the interpolated tide in 3 hours",
            )
    except Exception as e:
        print(f"Go/NoGo wind-vs-tide error: {e}")

    return factors, weather


_HOURS = list(range(8, 20))   # 08:00 … 19:00 → hourly boxes (4 per old 4h block)


def _tide_dot_color(height):
    """Tide-level dot: green > 2.5 m, orange 1.5–2.5 m, red < 1.5 m."""
    if height is None:
        return None
    if height > 2.5:
        return '#2ecc71'   # green
    if height >= 1.5:
        return '#f39c12'   # orange
    return '#e74c3c'       # red


def _analyze_5day_windows(weather_data):
    """Hourly boating windows (08:00–19:00) for the next 6 days. Box status
    from wind+rain (OpenWeather is 3-hourly, so nearby hours may match);
    a per-hour tide dot from the interpolated tide height."""
    if not weather_data or not weather_data.hourly_forecast:
        return []

    vancouver_tz = pytz.timezone('America/Vancouver')
    now = datetime.now(vancouver_tz)
    today = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Tide extremes + interpolation arrays (interp gives per-hour height)
    tide_df, x_ts, y_h = _get_tide_data()

    windows = []
    for day_offset in range(0, 6):
        day = today + timedelta(days=day_offset)
        for hour in _HOURS:
            target = day.replace(hour=hour)
            if target < now - timedelta(hours=1):
                continue

            # Nearest forecast points (3-hourly source → within 2h)
            items = [
                item for item in weather_data.hourly_forecast
                if abs((datetime.fromtimestamp(item['dt']).astimezone(vancouver_tz) - target).total_seconds()) <= 7200
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

            tide_h = _tide_at(x_ts, y_h, target) if x_ts is not None else None

            windows.append({
                'day': day.strftime('%a %b %d'),
                'period': f"{hour:02d}:00",
                'datetime': target,
                'status': status,
                'wind': max_wind,
                'rain': total_rain,
                'tide_h': tide_h,
                'tide_dot': _tide_dot_color(tide_h),
            })

    return windows


def _get_overall(factors):
    """Compute overall status from factors dict."""
    if not factors:
        return 'caution', 'N/A'
    statuses = [f['status'] for f in factors.values() if not f.get('informational')]
    if 'nogo' in statuses:
        return 'nogo', 'NO-GO'
    if 'caution' in statuses:
        return 'caution', 'CAUTION'
    return 'go', 'GO'


def _red_reason_pills_html(factors):
    """Red (no-go) reasons as inline pill tags. Warnings/caution are
    ignored — only the factors that actually drive a NO-GO are listed."""
    reds = [f['label'] for f in factors.values()
            if f.get('status') == 'nogo' and not f.get('informational')]
    if not reds:
        return None
    return " ".join(
        '<span style="background:#e74c3c;color:#fff;padding:3px 10px;'
        'border-radius:12px;font-size:0.82rem;font-weight:600;'
        'margin:2px 6px 2px 0;display:inline-block;">🔴 ' + str(r) + '</span>'
        for r in reds
    )


# ──────────────────────────────────────────────
# Sidebar: compact badge only
# ──────────────────────────────────────────────

def display_gonogo_sidebar():
    """Compact Go/No-Go badge in the sidebar."""
    st.sidebar.markdown("---")

    factors, _ = _gather_current_factors()
    overall, overall_label = _get_overall(factors)

    st.sidebar.badge(overall_label, color=_BADGE[overall])

    # Red (no-go) reasons as tags directly under the verdict
    pills = _red_reason_pills_html(factors)
    if pills:
        st.sidebar.markdown(pills, unsafe_allow_html=True)

    # One-line summary of worst factors (caution + nogo)
    bad = [f['label'] for f in factors.values()
           if f['status'] != 'go' and not f.get('informational')]
    if bad:
        st.sidebar.caption(", ".join(bad))
    else:
        st.sidebar.caption("All clear")


# ──────────────────────────────────────────────
# Full page: detailed view with chart
# ──────────────────────────────────────────────

def _get_marine_forecast_rows(url):
    """Get the first 2 rows of the GPT-parsed marine forecast for a siteID URL.
    Returns list of dicts with 'time', 'direction', 'wind_speed',
    'max_wind_speed' or an empty list."""
    try:
        csv_text = openAIFetchForecastForURL(url=url)
        if not csv_text:
            return []
        csv_clean = csv_text.replace('```csv', '').replace('```', '')
        df = pd.read_csv(io.StringIO(csv_clean), sep=',', on_bad_lines='skip')
        df = df.dropna(how='all').reset_index(drop=True)
        df.columns = df.columns.str.strip().str.lower()
        rows = []
        for _, row in df.head(2).iterrows():
            r = {}
            r['time'] = str(row.get('time', ''))
            # Wind direction column varies in the GPT output ('wind direction',
            # 'direction', 'wind_direction'); take the first that's present.
            r['direction'] = None
            for dcol in ('wind direction', 'wind_direction', 'direction'):
                if dcol in df.columns and pd.notna(row.get(dcol)):
                    r['direction'] = str(row[dcol]).strip() or None
                    break
            if 'wind_speed' in df.columns:
                r['wind_speed'] = clean_wind_speed(row['wind_speed'])
            elif 'wind speed' in df.columns:
                r['wind_speed'] = clean_wind_speed(row['wind speed'])
            else:
                r['wind_speed'] = None
            if 'max_wind_speed' in df.columns:
                r['max_wind_speed'] = clean_wind_speed(row['max_wind_speed'])
            elif 'max wind speed' in df.columns:
                r['max_wind_speed'] = clean_wind_speed(row['max wind speed'])
            else:
                r['max_wind_speed'] = None
            rows.append(r)
        return rows
    except Exception:
        return []


def _get_howe_sound_forecast_rows():
    """Howe Sound marine-forecast rows (kept for callers like fetch_alex)."""
    return _get_marine_forecast_rows(URL_HOWE_SOUND)


def display_gonogo_page(container=None, page_links=None):
    """Full Go/No-Go page with heatmap chart and current conditions."""
    draw = container or st
    page_links = page_links or {}

    draw.subheader("Go / No-Go — Boating Conditions")
    draw.caption("Horseshoe Bay launch | Howe Sound / Pt Atkinson / English Bay")

    factors, weather = _gather_current_factors()
    overall, overall_label = _get_overall(factors)

    # Overall verdict
    draw.badge(overall_label, color=_BADGE[overall])

    # Red reasons (no-go factors only) as tags, right above the conditions grid
    pills = _red_reason_pills_html(factors)
    if pills:
        draw.markdown(pills, unsafe_allow_html=True)

    draw.markdown("---")

    # Current conditions — one unified metric-card grid. Decision factors sort
    # by severity (no-go / caution first); informational context cards
    # (temperature, next forecast period) sort to the end. Hidden factors
    # (marine warnings) drive the verdict/pills but aren't shown as cards.
    draw.markdown("**Current Conditions**")
    cards = [(k, f) for k, f in factors.items() if not f.get('hide_card')]
    cards.sort(key=lambda kv: _CARD_ORDER.index(kv[0])
               if kv[0] in _CARD_ORDER else len(_CARD_ORDER))
    n_cols = 3
    for i in range(0, len(cards), n_cols):
        cols = draw.columns(n_cols)
        for col, (key, f) in zip(cols, cards[i:i + n_cols]):
            label = f['label']
            # Card value = text after the first colon, with any trailing
            # " — detail" clause trimmed for a compact metric value.
            value = label.split(':', 1)[1].strip() if ':' in label else label
            if ' — ' in value:
                value = value.split(' — ', 1)[0].strip()

            base_title = f.get('card_title') or _CARD_TITLES.get(key, key.replace('_', ' ').title())
            if f.get('informational'):
                # Context card — its own emoji title, no pass/fail status icon.
                title = base_title
            else:
                title = f"{_ICON[f['status']]} {base_title}"

            col.metric(title, value, border=True, help=f.get('help'))

            badge = f.get('badge')
            if badge:
                col.badge(badge['text'], color=badge['color'])
            page_func = page_links.get(f.get('page')) if f.get('page') else None
            if page_func:
                col.page_link(page_func, label="Details →")

    draw.markdown(
        f"*Thresholds: Wind GO < {WIND_GO}kts, CAUTION < {WIND_CAUTION}kts  |  "
        f"Waves GO < {int(WAVE_GO * 100)}cm  |  "
        f"Tide NO-GO < {TIDE_NOGO:.1f}m (Horseshoe Bay minimum)*"
    )

    # ── Marine station map (live wind & waves) ──
    draw.markdown("---")
    draw.markdown("**Marine Stations — Live Wind & Waves**")
    try:
        from fetch_alex import build_marine_station_map
        draw.plotly_chart(build_marine_station_map(), width='stretch')
        draw.caption(
            "Source: Environment Canada buoys (Pam Rocks, Halibut Bank, Pt Atkinson), "
            "Jericho station, Howe Sound marine forecast"
        )
    except Exception as e:
        print(f"Go/NoGo station map error: {e}")
        draw.caption(f"Station map unavailable: {e}")

    # 5-day heatmap chart
    if weather:
        windows = _analyze_5day_windows(weather)
        if windows:
            draw.markdown("---")
            draw.markdown("**Weekly Outlook**")
            _draw_weekly_chart(draw, windows)

            with draw.expander("Details"):
                for w in windows:
                    detail = f"{w['wind']:.0f}kts"
                    if w['rain'] > 0:
                        detail += f", {w['rain']:.1f}mm rain"
                    if w.get('tide_h') is not None:
                        detail += f", tide {w['tide_h']:.1f}m"
                    draw.caption(f"{_ICON[w['status']]} {w['day']} {w['period']} — {detail}")


def _draw_weekly_chart(draw, windows):
    """Heatmap: days x HOURLY slots (08:00–19:00). Cell colour = wind/rain
    status; a tide-level dot (green/orange/red) sits in each cell."""
    days = []
    seen = set()
    for w in windows:
        if w['day'] not in seen:
            days.append(w['day'])
            seen.add(w['day'])

    periods = [f"{h:02d}:00" for h in _HOURS]   # 12 hourly rows

    # index windows for quick lookup
    grid = {(w['day'], w['period']): w for w in windows}

    z, text = [], []
    for period in periods:
        row_z, row_text = [], []
        for day in days:
            m = grid.get((day, period))
            if m:
                row_z.append(_NUMERIC[m['status']])
                th = m.get('tide_h')
                parts = [f"{m['wind']:.0f} kts", f"tide {th:.1f} m" if th is not None else "tide —"]
                if m['rain'] > 0:
                    parts.append(f"{m['rain']:.1f} mm")
                row_text.append("<br>".join(parts))
            else:
                row_z.append(None)
                row_text.append("")
        z.append(row_z)
        text.append(row_text)

    colorscale = [
        [0, '#e74c3c'], [0.25, '#e74c3c'],
        [0.25, '#f39c12'], [0.75, '#f39c12'],
        [0.75, '#2ecc71'], [1, '#2ecc71'],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=days, y=periods, text=text,
        colorscale=colorscale, zmin=0, zmax=1, showscale=False,
        hovertemplate="<b>%{x} %{y}</b><br>%{text}<extra></extra>",
        xgap=2, ygap=2,
    ))
    fig.update_layout(
        height=560,
        margin=dict(l=55, r=20, t=30, b=10),
        yaxis=dict(autorange='reversed'),
        xaxis=dict(side='top'),
        plot_bgcolor='white',
    )

    # Per-cell: tide dot on top, wind number below.
    for period in periods:
        for day in days:
            m = grid.get((day, period))
            if not m:
                continue
            dot = m.get('tide_dot')
            if dot:
                fig.add_annotation(x=day, y=period, text='●', showarrow=False,
                                   font=dict(color=dot, size=15), yshift=11)
            fig.add_annotation(x=day, y=period, text=f"<b>{m['wind']:.0f}</b>",
                               showarrow=False, font=dict(color='white', size=11), yshift=-7)

    fig.update_traces(texttemplate=None)
    draw.plotly_chart(fig, width='stretch')
    draw.caption("Cell colour = wind/rain rule · tide dot: 🟢 > 2.5 m · 🟠 1.5–2.5 m · 🔴 < 1.5 m")


# ──────────────────────────────────────────────
# Kiosk / screensaver mode — dark, large fonts
# ──────────────────────────────────────────────

_KIOSK_CSS = """
<style>
    /* Hide Streamlit chrome for kiosk mode */
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="stHeader"] { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    footer { display: none !important; }
    .block-container {
        padding-top: 1rem !important;
        /* Generous bottom padding so the Home button is reachable on
           mobile — without this Safari/Chrome bottom URL bars can hide it. */
        padding-bottom: 8rem !important;
        max-width: 100% !important;
    }

    /* Dark background */
    .stApp { background-color: #0a0a0a !important; }
    .stApp * { color: #e0e0e0 !important; }

    /* Compact verdict banner */
    .kiosk-verdict {
        font-size: 2.2rem;
        font-weight: 900;
        text-align: center;
        padding: 0.25rem 0.5rem;
        border-radius: 12px;
        margin-bottom: 0.3rem;
    }
    .kiosk-go { background: #2ecc71; color: #000 !important; }
    .kiosk-caution { background: #f39c12; color: #000 !important; }
    .kiosk-nogo { background: #e74c3c; color: #fff !important; }

    .kiosk-factor {
        font-size: 1.3rem;
        padding: 0.15rem 0;
    }
    .kiosk-time {
        font-size: 1rem;
        color: #888 !important;
        text-align: center;
    }

    /* Kiosk metric cards */
    .kiosk-metrics {
        display: flex;
        justify-content: space-around;
        flex-wrap: wrap;
        gap: 0.3rem;
        margin: 0.4rem 0;
    }
    .kiosk-metric {
        text-align: center;
        flex: 1;
        min-width: 120px;
    }
    .kiosk-metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #fff !important;
    }
    .kiosk-metric-label {
        font-size: 0.85rem;
        color: #888 !important;
    }

    /* Big 'Home / Exit kiosk' button at the bottom */
    .kiosk-home-wrap {
        margin: 3rem 0 5rem 0;  /* extra bottom margin so user can scroll past */
        text-align: center;
    }
    .kiosk-home-wrap [data-testid="stPageLink"] {
        display: block;
        width: 100%;
    }
    .kiosk-home-wrap [data-testid="stPageLink"] a,
    .kiosk-home-wrap [data-testid="stPageLink"] button {
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        width: 100% !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #fff !important;
        background: #1f3a5f !important;
        border: 2px solid #4a90e2 !important;
        border-radius: 14px !important;
        padding: 1.4rem 1rem !important;
        text-decoration: none !important;
        box-shadow: 0 2px 10px rgba(74, 144, 226, 0.3);
    }
    .kiosk-home-wrap [data-testid="stPageLink"] a:hover,
    .kiosk-home-wrap [data-testid="stPageLink"] button:hover {
        background: #2c5282 !important;
        border-color: #6cb3ff !important;
    }
    .kiosk-home-wrap [data-testid="stPageLink"] * {
        color: #fff !important;
    }
</style>
"""


def display_kiosk_page(home_page=None):
    """Full-screen dark kiosk mode for TV / Nvidia Shield screensaver.

    `home_page` is an optional st.Page reference for the main Go/No-Go page.
    When provided we render a big 'Home' button at the bottom so mobile users
    can exit kiosk mode (the sidebar is hidden by the kiosk CSS)."""
    st.markdown(_KIOSK_CSS, unsafe_allow_html=True)

    factors, weather = _gather_current_factors()
    overall, overall_label = _get_overall(factors)

    vancouver_tz = pytz.timezone('America/Vancouver')
    now = datetime.now(vancouver_tz)

    # Large verdict
    css_class = f"kiosk-{overall}"
    st.markdown(
        f'<div class="kiosk-verdict {css_class}">{overall_label}</div>',
        unsafe_allow_html=True,
    )

    # Current time
    st.markdown(
        f'<div class="kiosk-time">{now.strftime("%A %B %d, %H:%M")}</div>',
        unsafe_allow_html=True,
    )

    # ── Snapshot metrics row (dark-styled) ──
    _draw_kiosk_snapshot(weather)

    # Current conditions — large text (problems sorted to the top). Hidden
    # factors (marine warnings) drive the verdict but aren't listed here.
    for f in sorted(factors.values(), key=lambda f: _SEVERITY.get(f['status'], 1)):
        if f.get('hide_card'):
            continue
        if f.get('informational'):
            title = f.get('card_title', '')
            text = f"{title}: {f['label']}" if title else f['label']
        else:
            text = f"{_ICON[f['status']]} {f['label']}"
        st.markdown(
            f'<div class="kiosk-factor">{text}</div>',
            unsafe_allow_html=True,
        )

    # Weekly heatmap — dark themed
    if weather:
        windows = _analyze_5day_windows(weather)
        if windows:
            _draw_kiosk_chart(windows)

    # ── Big Home button at the bottom — exit kiosk mode on mobile ──
    if home_page is not None:
        st.markdown('<div class="kiosk-home-wrap">', unsafe_allow_html=True)
        st.page_link(home_page, label="🏠  Home / Exit kiosk", icon=None)
        st.markdown('</div>', unsafe_allow_html=True)


def _draw_kiosk_snapshot(weather):
    """Render snapshot metrics as dark-styled HTML for kiosk mode."""
    metrics = []

    # Temperature
    if weather:
        metrics.append(("🌡️ Temp", f"{weather.temperature:.0f}°C"))
        metrics.append(("🌧️ Rain 3h", f"{weather.next_3_hours_precipitation:.1f}mm"))
    else:
        metrics.append(("🌡️ Temp", "N/A"))
        metrics.append(("🌧️ Rain 3h", "N/A"))

    # Tide
    try:
        tide_h, tide_dir = _get_current_tide_height()
        if tide_h is not None:
            suffix = f" {tide_dir}" if tide_dir else ""
            metrics.append(("🌊 Tide", f"{tide_h:.1f}m{suffix}"))
        else:
            metrics.append(("🌊 Tide", "N/A"))
    except Exception:
        metrics.append(("🌊 Tide", "N/A"))

    # Pam Rocks wind
    try:
        pam_wind, _ = _fetch_buoy_wind_wave('WAS')
        if pam_wind is not None:
            metrics.append(("💨 Pam Rocks", f"{pam_wind}kts"))
        else:
            metrics.append(("💨 Pam Rocks", "N/A"))
    except Exception:
        metrics.append(("💨 Pam Rocks", "N/A"))

    # Howe Sound forecast
    try:
        rows = _get_howe_sound_forecast_rows()
        if rows:
            r = rows[0]
            speed = f"{r['wind_speed']:.0f}" if r.get('wind_speed') is not None else "?"
            gust = f"{r['max_wind_speed']:.0f}" if r.get('max_wind_speed') is not None else "?"
            metrics.append((f"💨 Howe ({r['time']})", f"{speed}-{gust}kts"))
            if len(rows) > 1:
                r2 = rows[1]
                s2 = f"{r2['wind_speed']:.0f}" if r2.get('wind_speed') is not None else "?"
                g2 = f"{r2['max_wind_speed']:.0f}" if r2.get('max_wind_speed') is not None else "?"
                metrics.append((f"💨 Next ({r2['time']})", f"{s2}-{g2}kts"))
    except Exception:
        pass

    cards = ""
    for label, value in metrics:
        cards += (
            f'<div class="kiosk-metric">'
            f'<div class="kiosk-metric-value">{value}</div>'
            f'<div class="kiosk-metric-label">{label}</div>'
            f'</div>'
        )

    st.markdown(f'<div class="kiosk-metrics">{cards}</div>', unsafe_allow_html=True)


def _draw_kiosk_chart(windows):
    """Dark-themed heatmap for kiosk display."""
    days = []
    seen = set()
    for w in windows:
        if w['day'] not in seen:
            days.append(w['day'])
            seen.add(w['day'])

    periods = ['08:00', '12:00', '16:00']

    z = []
    for period in periods:
        row_z = []
        for day in days:
            match = next((w for w in windows if w['day'] == day and w['period'] == period), None)
            row_z.append(_NUMERIC[match['status']] if match else None)
        z.append(row_z)

    colorscale = [
        [0, '#e74c3c'],
        [0.25, '#e74c3c'],
        [0.25, '#f39c12'],
        [0.75, '#f39c12'],
        [0.75, '#2ecc71'],
        [1, '#2ecc71'],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=days,
        y=periods,
        colorscale=colorscale,
        zmin=0, zmax=1,
        showscale=False,
        xgap=4, ygap=4,
        hoverinfo='skip',
    ))

    fig.update_layout(
        height=300,
        margin=dict(l=70, r=20, t=10, b=10),
        yaxis=dict(autorange='reversed'),
        xaxis=dict(side='top'),
        plot_bgcolor='#0a0a0a',
        paper_bgcolor='#0a0a0a',
        font=dict(color='#e0e0e0', size=16),
    )

    # Annotations — wind + tide times, large
    for i, period in enumerate(periods):
        for j, day in enumerate(days):
            match = next((w for w in windows if w['day'] == day and w['period'] == period), None)
            if match:
                label = f"<b>{match['wind']:.0f}</b>kts"
                th = match.get('tide_h')
                if th is not None:
                    dot = match.get('tide_dot') or '#e0e0e0'
                    label += f'<br><span style="color:{dot}">●</span> {th:.1f}m'
                fig.add_annotation(
                    x=day, y=period,
                    text=label,
                    showarrow=False,
                    font=dict(color='white', size=16),
                )

    fig.update_traces(texttemplate=None)
    st.plotly_chart(fig, width='stretch')
