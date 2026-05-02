import requests
import streamlit as st
import pytz
import plotly.graph_objects as go
from dataclasses import dataclass
from datetime import datetime, timedelta
from timeago import format as timeago_format

from wind_utils import create_arrow_html


@dataclass
class WeatherData:
    hourly_forecast: list
    temperature: float
    cloud_condition: str
    outside_humidity: int
    next_3_hours_precipitation: float
    next_24_hours_precipitation: float
    timestamp: datetime
    wind_speed_now: float
    wind_direction_now: int
    wind_speed_3h: float
    wind_direction_3h: int
    weather_icon: str
    sunrise: datetime
    sunset: datetime


@st.cache_data(ttl=300)
def fetch_from_open_weather(lat: float, lon: float, api_key: str) -> WeatherData:
    """Fetch weather data from OpenWeatherMap API"""
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    hourly_forecast_url = "https://api.openweathermap.org/data/2.5/forecast"

    try:
        current_params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric"
        }

        current_response = requests.get(base_url, params=current_params)
        current_response.raise_for_status()
        current_data = current_response.json()

        vancouver_tz = pytz.timezone('America/Vancouver')
        sunrise = datetime.fromtimestamp(current_data['sys']['sunrise'], tz=pytz.UTC).astimezone(vancouver_tz)
        sunset = datetime.fromtimestamp(current_data['sys']['sunset'], tz=pytz.UTC).astimezone(vancouver_tz)

        hourly_params = {
            "lat": lat,
            "lon": lon,
            "appid": api_key,
            "units": "metric",
            "cnt": 72
        }

        hourly_response = requests.get(hourly_forecast_url, params=hourly_params)
        hourly_response.raise_for_status()
        hourly_data = hourly_response.json()

        next_3_hours_precip = hourly_data['list'][0].get('rain', {}).get('3h', 0) if 'list' in hourly_data else 0
        next_24_hours_precip = sum(
            item.get('rain', {}).get('3h', 0)
            for item in hourly_data.get('list', [])[:8]
        )

        return WeatherData(
            sunrise=sunrise,
            sunset=sunset,
            temperature=current_data['main']['temp'],
            cloud_condition=current_data['weather'][0]['description'],
            outside_humidity=current_data['main']['humidity'],
            next_3_hours_precipitation=next_3_hours_precip,
            next_24_hours_precipitation=next_24_hours_precip,
            timestamp=datetime.fromtimestamp(current_data['dt']),
            wind_speed_now=current_data['wind']['speed'],
            wind_direction_now=current_data['wind']['deg'],
            wind_speed_3h=hourly_data['list'][0]['wind']['speed'],
            wind_direction_3h=hourly_data['list'][0]['wind']['deg'],
            weather_icon=current_data["weather"][0]["icon"],
            hourly_forecast=hourly_data['list']
        )

    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None


def get_wind_direction(degrees: int) -> str:
    """Convert wind direction from degrees to cardinal direction"""
    directions = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW']
    index = round(degrees / (360. / len(directions))) % len(directions)
    return directions[index]


def display_weather_info(container=None, lat=None, long=None, title=''):
    if container is None:
        container = st

    api_key = st.secrets["openweather_api_key"]
    weather_data = fetch_from_open_weather(lat, long, api_key)

    vancouver_tz = pytz.timezone('America/Vancouver')
    weather_timestamp_van = weather_data.timestamp.astimezone(vancouver_tz)

    # Harmonised staleness badge — consistent with all other pages.
    from utils import display_last_updated_badge
    if title:
        container.subheader(title)
    source_url = (
        f"https://openweathermap.org/weathermap?lat={lat}&lon={long}&zoom=10"
        if lat is not None and long is not None
        else "https://openweathermap.org/"
    )
    raw_summary = ''
    try:
        wind_kts = weather_data.wind_speed_now * 1.94384
        wind_dir_short = get_wind_direction(weather_data.wind_direction_now)
        raw_summary = (
            f"{wind_dir_short} {wind_kts:.0f}kts · "
            f"{weather_data.temperature:.0f}°C · "
            f"{weather_data.next_3_hours_precipitation:.1f}mm/3h"
        )
    except Exception:
        pass
    display_last_updated_badge(
        container, weather_timestamp_van, label="Issued",
        source_url=source_url, source_label='openweathermap.org',
        extra_text=raw_summary or None,
    )

    if weather_data:
        col1, col2, col3 = container.columns(3)

        wind_speed_now_kts = weather_data.wind_speed_now * 1.94384
        wind_speed_3h_kts = weather_data.wind_speed_3h * 1.94384

        with col1:
            sunrise_time = weather_data.sunrise.astimezone(vancouver_tz).strftime('%H:%M')
            st.metric("🌅 Sunrise", sunrise_time)
            icon_url = f"http://openweathermap.org/img/wn/{weather_data.weather_icon}@2x.png"
            st.image(icon_url, width=64)
            wind_dir_now = get_wind_direction(weather_data.wind_direction_now)
            st.metric("💨 Wind Now", f"{wind_dir_now} {wind_speed_now_kts:.1f}kts",
                      delta=round(wind_speed_3h_kts), delta_color="inverse")

        with col2:
            sunset_time = weather_data.sunset.astimezone(vancouver_tz).strftime('%H:%M')
            st.metric("🪐 Sunset", sunset_time)
            st.metric("🌧️ 3h Precipitation", f"{weather_data.next_3_hours_precipitation:.1f}mm",
                      delta=weather_data.next_24_hours_precipitation, delta_color="inverse")
            wind_dir_3h = get_wind_direction(weather_data.wind_direction_3h)
            st.metric("💨 Wind in 3h", f"{wind_dir_3h} {wind_speed_3h_kts:.1f}kts")

        with col3:
            st.metric("🌡️ Temperature", f"{weather_data.temperature:.1f}°C")
            st.metric("🌧️ 24h Precipitation", f"{weather_data.next_24_hours_precipitation:.1f}mm")
            st.metric("💧️ Humidity", f"{weather_data.outside_humidity}%")

        col1.markdown(create_arrow_html(wind_dir_now, wind_speed_now_kts), unsafe_allow_html=True)
        col2.markdown(create_arrow_html(wind_dir_now, wind_speed_3h_kts), unsafe_allow_html=True)

        display_precipitation_forecast(weather_data, container)


def display_precipitation_forecast(weather_data, container):
    timestamps = []
    precip_chances = []

    vancouver_tz = pytz.timezone('America/Vancouver')

    for item in weather_data.hourly_forecast:
        dt = datetime.fromtimestamp(item['dt']).astimezone(vancouver_tz)
        timestamps.append(dt)
        rain = item.get('rain', 0)
        risk_of_rain = -.1
        if rain:
            risk_of_rain = rain.get('3h', 0)
        precip_chances.append(risk_of_rain * 10)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=timestamps,
        y=precip_chances,
        name='Precipitation Chance',
        marker_color='rgba(58, 134, 255, 0.6)'
    ))

    fig.update_layout(
        title='5-Day Precipitation Forecast',
        xaxis_title='Day',
        yaxis_title='Precipitation Chance (%)',
        yaxis=dict(range=[0, 30]),
        plot_bgcolor='white',
        hovermode='x unified',
        xaxis=dict(
            dtick='D1',
            tickformat='%A',
            tickangle=45,
        )
    )

    for day in range(6):
        base_time = datetime.now(vancouver_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = base_time + timedelta(days=day)

        fig.add_vline(x=day_start, line_dash="dash", line_color="green", opacity=0.8)
        fig.add_annotation(
            x=day_start, y=100,
            text=day_start.strftime('%A'),
            showarrow=False, yshift=10,
            font=dict(color="green", size=12, weight="bold")
        )

    add_wind_forecast_to_plotly_chart(weather_data, fig)
    container.plotly_chart(fig, width='stretch')


def add_wind_forecast_to_plotly_chart(weather_data, fig):
    timestamps = []
    wind_speeds = []
    wind_gusts = []

    vancouver_tz = pytz.timezone('America/Vancouver')

    for item in weather_data.hourly_forecast:
        dt = datetime.fromtimestamp(item['dt']).astimezone(vancouver_tz)
        timestamps.append(dt)
        wind_speed = item['wind'].get('speed', 0) * 1.94384
        wind_gust = item['wind'].get('gust', wind_speed) * 1.94384
        wind_speeds.append(wind_speed)
        wind_gusts.append(wind_gust)

    fig.add_trace(go.Scatter(
        x=timestamps, y=wind_speeds,
        name='Wind Speed',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=timestamps, y=wind_gusts,
        name='Wind Gust',
        line=dict(color='red', width=2, dash='dash')
    ))

    fig.update_layout(
        title='5-Day Wind Forecast',
        xaxis_title='Time',
        yaxis_title='Wind Speed (knots)',
        plot_bgcolor='white',
        hovermode='x unified',
        xaxis=dict(
            dtick=8 * 3600 * 1000,
            tickformat='%H:00',
        )
    )

    for day in range(6):
        base_time = datetime.now(vancouver_tz).replace(hour=0, minute=0, second=0, microsecond=0)
        day_start = base_time + timedelta(days=day)

        fig.add_vline(x=day_start, line_dash="dash", line_color="gray", opacity=0.5)
        fig.add_annotation(
            x=day_start,
            y=max(max(wind_speeds), max(wind_gusts)),
            text=day_start.strftime('%a'),
            showarrow=False, yshift=10, xanchor='left', xshift=5,
            font=dict(color="green", size=12, weight="bold")
        )

    current_time = datetime.now(vancouver_tz)
    current_time_ts = current_time.timestamp() * 1000

    fig.add_vline(
        x=current_time_ts,
        line_width=2, line_dash="dash", line_color="red",
        annotation_text="Now", annotation_position="top right"
    )


def display_clear_skies_html(container, title="Clear Skies"):
    if title:
        container.subheader(title)
    container.caption("From https://www.cleardarksky.com/c/Vancouverkey.html")
    container.markdown(
        '<a href=https://www.cleardarksky.com/c/Vancouverkey.html>'
        '<img src="https://www.cleardarksky.com/c/Vancouvercsk.gif?c=986410"></a>',
        unsafe_allow_html=True
    )
