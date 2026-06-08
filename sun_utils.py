"""Day/night background shading for time-series Plotly charts.

Dependency-free sunrise/sunset (NOAA / Almanac sunrise equation) so we
can shade a chart's daylight hours light sky-blue and night hours darker
navy without pulling in astral or hitting an external API.

Shared by st.py (wave chart) and fetch_tides.py (tide chart).
"""
from __future__ import annotations

import math
from datetime import date as _date, datetime, timedelta

import pytz
import pandas as pd
import plotly.graph_objects as go

VAN_TZ = pytz.timezone('America/Vancouver')

DAY_FILL = 'rgba(135, 206, 250, 0.28)'    # light sky blue
NIGHT_FILL = 'rgba(28, 52, 94, 0.33)'     # darker navy
DAY_LEGEND = 'rgba(135, 206, 250, 0.85)'
NIGHT_LEGEND = 'rgba(28, 52, 94, 0.85)'


def _sunrise_sunset_ut(year, month, day, lat, lon):
    """NOAA / Almanac sunrise equation. Returns (sunrise_ut, sunset_ut) as
    fractional hours UTC for the given date, or None where the sun never
    rises/sets (not a concern at Vancouver's latitude)."""
    to_rad = math.pi / 180.0
    N = _date(year, month, day).timetuple().tm_yday

    def calc(is_sunrise):
        lng_hour = lon / 15.0
        t = N + ((6 if is_sunrise else 18) - lng_hour) / 24.0
        M = (0.9856 * t) - 3.289
        L = (M + 1.916 * math.sin(M * to_rad) + 0.020 * math.sin(2 * M * to_rad) + 282.634) % 360
        RA = (math.atan(0.91764 * math.tan(L * to_rad)) / to_rad) % 360
        RA += (math.floor(L / 90) * 90) - (math.floor(RA / 90) * 90)
        RA /= 15.0
        sin_dec = 0.39782 * math.sin(L * to_rad)
        cos_dec = math.cos(math.asin(sin_dec))
        zenith = 90.833  # official sunrise/sunset including refraction
        cos_h = (math.cos(zenith * to_rad) - sin_dec * math.sin(lat * to_rad)) / \
                (cos_dec * math.cos(lat * to_rad))
        if cos_h > 1 or cos_h < -1:
            return None
        H = (360 - math.acos(cos_h) / to_rad) if is_sunrise else (math.acos(cos_h) / to_rad)
        H /= 15.0
        T = H + RA - (0.06571 * t) - 6.622
        return (T - lng_hour) % 24

    return calc(True), calc(False)


def _sun_event_local(d, ut, tz):
    """Convert a fractional-hour UTC sun event on date d to a tz-aware local
    datetime, snapping it back onto local date d (corrects the UTC date
    drift that otherwise puts Vancouver sunset on the wrong calendar day)."""
    if ut is None:
        return None
    base = datetime(d.year, d.month, d.day, tzinfo=pytz.UTC)
    dt_utc = base + timedelta(hours=ut)
    local = dt_utc.astimezone(tz)
    drift = (local.date() - d).days
    if drift:
        local = (dt_utc - timedelta(days=drift)).astimezone(tz)
    return local


# Light grey night, no day fill — used on the wind charts.
NIGHT_GREY = 'rgba(120, 120, 120, 0.18)'
NIGHT_GREY_LEGEND = 'rgba(120, 120, 120, 0.75)'


def add_day_night_shading(fig, start, end, lat=49.28, lon=-123.12, add_legend=True,
                          day_fill=DAY_FILL, night_fill=NIGHT_FILL,
                          day_legend=DAY_LEGEND, night_legend=NIGHT_LEGEND):
    """Shade fig's background by sun position across [start, end]. Bands sit
    BELOW the data. `start`/`end` should be tz-aware datetimes.

    Pass day_fill=None to draw only night bands (e.g. light-grey nights on
    the wind charts). Colors are overridable for per-chart styling."""
    tz = VAN_TZ
    start = start.astimezone(tz)
    end = end.astimezone(tz)

    # Daylight intervals per calendar day in the window (±1 day slack)
    day_intervals = []
    d = start.date() - timedelta(days=1)
    last = end.date() + timedelta(days=1)
    while d <= last:
        sr_ut, ss_ut = _sunrise_sunset_ut(d.year, d.month, d.day, lat, lon)
        sr = _sun_event_local(d, sr_ut, tz)
        ss = _sun_event_local(d, ss_ut, tz)
        if sr and ss and sr < ss:
            a, b = max(sr, start), min(ss, end)
            if a < b:
                day_intervals.append((a, b))
        d += timedelta(days=1)
    day_intervals.sort()

    # Night = complement of daylight within the window
    night_intervals = []
    cursor = start
    for a, b in day_intervals:
        if cursor < a:
            night_intervals.append((cursor, a))
        cursor = max(cursor, b)
    if cursor < end:
        night_intervals.append((cursor, end))

    if day_fill:
        for a, b in day_intervals:
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=pd.Timestamp(a), x1=pd.Timestamp(b), y0=0, y1=1,
                          fillcolor=day_fill, line_width=0, layer='below')
    if night_fill:
        for a, b in night_intervals:
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=pd.Timestamp(a), x1=pd.Timestamp(b), y0=0, y1=1,
                          fillcolor=night_fill, line_width=0, layer='below')

    if add_legend:
        # Use a real in-window datetime as x (with y=None so nothing draws).
        # x=[None] would leave the x-axis type ambiguous and, combined with a
        # numeric add_vline elsewhere, can flip the whole axis to linear.
        x_anchor = pd.Timestamp(start)
        if day_fill:
            fig.add_trace(go.Scatter(x=[x_anchor], y=[None], mode='markers', name='Day',
                                     marker=dict(size=12, color=day_legend, symbol='square')))
        if night_fill:
            fig.add_trace(go.Scatter(x=[x_anchor], y=[None], mode='markers', name='Night',
                                     marker=dict(size=12, color=night_legend, symbol='square')))


_NIGHT_WORDS = ('evening', 'night', 'tonight', 'overnight')


def is_night_period(label) -> bool:
    """Classify a marine-forecast period label as night. Used to shade the
    *categorical* forecast chart (which plots 'now', 'this morning',
    'Tuesday evening', … — not real clock times). 'now' falls back to the
    current Vancouver hour (night if before 6am or after 9pm)."""
    s = str(label or '').strip().lower()
    if not s or s == 'now':
        h = datetime.now(VAN_TZ).hour
        return h < 6 or h >= 21
    return any(w in s for w in _NIGHT_WORDS)


def add_categorical_night_shading(fig, time_labels,
                                  night_fill=NIGHT_GREY, night_legend=NIGHT_GREY_LEGEND,
                                  add_legend=True):
    """Shade night periods on a categorical-x chart by column index. Each
    label maps to category index i; night labels get a grey band over
    [i-0.5, i+0.5]. Returns the count of night bands drawn."""
    labels = list(time_labels)
    n = 0
    for i, label in enumerate(labels):
        if is_night_period(label):
            fig.add_shape(type='rect', xref='x', yref='paper',
                          x0=i - 0.5, x1=i + 0.5, y0=0, y1=1,
                          fillcolor=night_fill, line_width=0, layer='below')
            n += 1
    if add_legend and n:
        fig.add_trace(go.Scatter(x=[labels[0]], y=[None], mode='markers', name='Night',
                                 marker=dict(size=12, color=night_legend, symbol='square')))
    return n
