import requests
import streamlit as st
import pytz
import pandas as pd
import pdfplumber
import io
import re
from datetime import datetime
from timeago import format as timeago_format


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

    # Prominent measurement date
    relative_date = timeago_format(time_measurement, datetime.now(pytz.timezone('America/Vancouver')))
    draw.subheader(f"Sampled {relative_date}")
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
