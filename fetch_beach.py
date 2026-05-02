import requests
import streamlit as st
import pytz
import pandas as pd
import pdfplumber
import io
import re
from datetime import datetime, timedelta
from timeago import format as timeago_format
import plotly.graph_objects as go


@st.cache_data(ttl=3600)  # Cap at 1 fetch / hour — VCH is rate-sensitive
def fetch_water_quality_for_url(_draw, url, title):
    try:
        # VCH blocks the default python-requests User-Agent with 403.
        # Send a browser-like UA + Accept headers to match what browsers do.
        headers = {
            'User-Agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                'AppleWebKit/537.36 (KHTML, like Gecko) '
                'Chrome/120.0.0.0 Safari/537.36'
            ),
            'Accept': 'application/pdf,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()

        ecoli_sample1 = None
        ecoli_sample2 = None
        time_measurement = None

        with pdfplumber.open(io.BytesIO(response.content)) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                date_match = re.search(r'\b(\d{1,2}/\d{1,2}/\d{4}\s+\d{1,2}:\d{2}\s+[AP]M)\b', text)
                if date_match and not time_measurement:
                    time_str = date_match.group(1)
                    tz = pytz.timezone("America/Vancouver")
                    naive_dt = datetime.strptime(time_str, "%m/%d/%Y %I:%M %p")
                    time_measurement = tz.localize(naive_dt)

                tables = page.extract_tables()
                for table in tables:
                    df = pd.DataFrame(table[1:], columns=table[0])
                    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True).str.strip()
                    if "Sample Name" in df.columns and "Ecoli MPN/100mLs" in df.columns:
                        match1 = df[df["Sample Name"].str.strip() == "BWV-04-01"]
                        if not match1.empty:
                            ecoli_sample1 = match1.iloc[0]["Ecoli MPN/100mLs"]
                        match2 = df[df["Sample Name"].str.strip() == "BWV-04-02"]
                        if not match2.empty:
                            ecoli_sample2 = match2.iloc[0]["Ecoli MPN/100mLs"]

        return ecoli_sample1, ecoli_sample2, time_measurement

    except Exception as e:
        print(f"Beach water quality fetch failed: {e}")
        return None, None, None


def _ecoli_color(value_str):
    """Return badge color based on E.coli MPN level."""
    try:
        val = int(value_str.replace("<", ""))
    except (ValueError, TypeError):
        return "gray"
    if val > 400:
        return "red"
    if val > 200:
        return "orange"
    return "green"


def _ecoli_status(value_str):
    """go / caution / nogo based on the latest E.coli reading."""
    try:
        val = int(str(value_str).replace("<", "").strip())
    except (ValueError, TypeError):
        return 'caution'
    if val > 400:
        return 'nogo'
    if val > 200:
        return 'caution'
    return 'go'


def _tide_status_for_beach(height, direction):
    """Sandy Cove rules:
       - GO when falling AND tide > 3m
       - GO when rising  AND tide > 2m
       Otherwise NO-GO (beach is fully underwater at extreme high tide)."""
    if height is None:
        return 'caution'
    if direction == 'falling' and height > 3.0:
        return 'go'
    if direction == 'rising' and height > 2.0:
        return 'go'
    return 'nogo'


def _worst_status(*statuses):
    order = {'go': 0, 'caution': 1, 'nogo': 2}
    return max(statuses, key=lambda s: order.get(s, 0))


def _build_beach_windows(ecoli_status_value, hours=24, step_hours=2):
    """Build a list of 2-hour beach-condition slots across the next `hours` hours.
    Returns list of dicts: {time_dt, day, time_label, tide_height, tide_dir, status}."""
    # Imported lazily to avoid circular imports between fetch_beach <-> fetch_gonogo
    from fetch_gonogo import _get_tide_data, _tide_at

    _, x_ts, y_h = _get_tide_data()
    vancouver_tz = pytz.timezone('America/Vancouver')
    now = datetime.now(vancouver_tz)

    # Anchor to the next even-numbered hour so labels read 8AM / 10AM / ... cleanly
    start = now.replace(minute=0, second=0, microsecond=0)
    if start.hour % step_hours:
        start = start + timedelta(hours=step_hours - (start.hour % step_hours))

    windows = []
    for i in range(hours // step_hours):
        slot = start + timedelta(hours=i * step_hours)
        height = _tide_at(x_ts, y_h, slot) if x_ts is not None else None
        direction = None
        if height is not None and x_ts is not None:
            future = _tide_at(x_ts, y_h, slot + timedelta(minutes=30))
            if future is not None:
                direction = 'rising' if future > height else 'falling'

        tide_st = _tide_status_for_beach(height, direction)
        status = _worst_status(ecoli_status_value, tide_st)

        # Day label uses today / tomorrow / weekday for readability
        if slot.date() == now.date():
            day_label = 'Today'
        elif slot.date() == (now + timedelta(days=1)).date():
            day_label = 'Tomorrow'
        else:
            day_label = slot.strftime('%a')

        hour12 = slot.hour % 12 or 12
        ampm = 'AM' if slot.hour < 12 else 'PM'

        windows.append({
            'time_dt': slot,
            'day': day_label,
            'time_label': f'{hour12}{ampm}',
            'tide_height': height,
            'tide_dir': direction,
            'status': status,
        })
    return windows


_BEACH_NUMERIC = {'go': 1, 'caution': 0.5, 'nogo': 0}


def display_beach_gonogo_table(draw, ecoli_status_value):
    """Render a heatmap-style Go/No-Go grid for beach swimming over the next 24h
    in 2-hour increments. Days on the y-axis, time slots on the x-axis."""
    windows = _build_beach_windows(ecoli_status_value, hours=24, step_hours=2)
    if not windows:
        draw.warning("Could not build beach Go/No-Go table (tide data unavailable).")
        return

    # Group by day, preserve order
    days = []
    seen = set()
    for w in windows:
        if w['day'] not in seen:
            days.append(w['day'])
            seen.add(w['day'])
    times = []
    seen_t = set()
    for w in windows:
        if w['time_label'] not in seen_t:
            times.append(w['time_label'])
            seen_t.add(w['time_label'])

    z = []
    for day in days:
        row = []
        for time_label in times:
            match = next((w for w in windows
                          if w['day'] == day and w['time_label'] == time_label), None)
            row.append(_BEACH_NUMERIC[match['status']] if match else None)
        z.append(row)

    colorscale = [
        [0.00, '#e74c3c'], [0.25, '#e74c3c'],
        [0.25, '#f39c12'], [0.75, '#f39c12'],
        [0.75, '#2ecc71'], [1.00, '#2ecc71'],
    ]

    fig = go.Figure(data=go.Heatmap(
        z=z, x=times, y=days,
        colorscale=colorscale,
        zmin=0, zmax=1,
        showscale=False,
        xgap=3, ygap=3,
        hoverinfo='skip',
    ))

    # Annotate each cell with tide height + arrow
    for w in windows:
        if w['tide_height'] is None:
            label = "—"
        else:
            arrow = ' ↑' if w['tide_dir'] == 'rising' else (' ↓' if w['tide_dir'] == 'falling' else '')
            label = f"<b>{w['tide_height']:.1f}m</b>{arrow}"
        fig.add_annotation(
            x=w['time_label'], y=w['day'],
            text=label, showarrow=False,
            font=dict(color='white', size=12),
        )

    fig.update_layout(
        height=170 + 50 * len(days),
        margin=dict(l=80, r=20, t=10, b=20),
        yaxis=dict(autorange='reversed'),
        xaxis=dict(side='top'),
        plot_bgcolor='white',
    )
    fig.update_traces(texttemplate=None)

    draw.markdown("**Beach Go/No-Go — next 24h (2-hour increments)**")
    draw.plotly_chart(fig, width='stretch')
    draw.caption(
        "Rules: water quality must be < 200 MPN, AND the beach must be exposed — "
        "tide > 3m if falling, or > 2m if rising. At high tide Sandy Cove is "
        "fully underwater."
    )


def display_beach_quality_for_sandy_cove(draw=None, title=''):
    if draw is None:
        draw = st

    draw.subheader(title)

    ecoli_sample1, ecoli_sample2, time_measurement = fetch_water_quality_for_url(
        _draw=draw,
        url='https://www.vch.ca/en/Documents/VCH-beach-route3.pdf',
        title=title
    )
    if not ecoli_sample1:
        st.warning('Beach water quality data unavailable — the VCH PDF may be slow or temporarily down. Will retry on next refresh.')
        return

    # Prominent measurement date — harmonised colored staleness badge
    from utils import display_last_updated_badge
    display_last_updated_badge(draw, time_measurement, label="Sampled")
    draw.caption(time_measurement.strftime('%A, %B %d %Y at %I:%M %p'))

    # Legend
    draw.markdown(
        "**E.coli MPN/100mL:**  "
        ":green[< 200 Safe]  &nbsp;  "
        ":orange[200-400 Caution]  &nbsp;  "
        ":red[> 400 Unsafe]"
    )

    if not ecoli_sample1:
        ecoli_sample1 = 'not found'
    if not ecoli_sample2:
        ecoli_sample2 = 'not found'

    col1, col2 = draw.columns(2)

    col1.metric("Station 1 (BWV-04-01)", ecoli_sample1)
    col1.badge(ecoli_sample1, color=_ecoli_color(ecoli_sample1))

    col2.metric("Station 2 (BWV-04-02)", ecoli_sample2)
    col2.badge(ecoli_sample2, color=_ecoli_color(ecoli_sample2))

    # Worst-case water-quality status drives the table cells across the next 24h.
    water_status = _worst_status(_ecoli_status(ecoli_sample1), _ecoli_status(ecoli_sample2))

    draw.markdown("---")
    try:
        display_beach_gonogo_table(draw, water_status)
    except Exception as e:
        draw.warning(f"Beach Go/No-Go table unavailable: {e}")
