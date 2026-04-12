import requests
import streamlit as st
import pytz
import pandas as pd
import pdfplumber
import io
import re
from datetime import datetime
from timeago import format as timeago_format


@st.cache_data(ttl=18800)
def fetch_water_quality_for_url(_draw, url, title):
    try:
        response = requests.get(url)
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
        return None, None, None


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
        st.info('Could not load URL')
        return

    relative_date = timeago_format(time_measurement, datetime.now(pytz.timezone('America/Vancouver')))
    draw.subheader(f"Issued {relative_date}")

    if not ecoli_sample1:
        ecoli_sample1 = 'not found'
    if not ecoli_sample2:
        ecoli_sample2 = 'not found'

    col1, col2, col3, col4 = draw.columns(4)
    col1.text('Sandy cove station 1 BWV-04-01')
    col2.badge(ecoli_sample1, color='green')

    ecoli_sample1 = ecoli_sample1.replace("<", '')
    ecoli_sample2 = ecoli_sample2.replace("<", '')

    if int(ecoli_sample1) > 200:
        col2.badge(ecoli_sample1, color='orange')
    if int(ecoli_sample1) > 400:
        col2.badge(ecoli_sample1, color='red')

    col3.text('Sandy cove station 2 BWV-04-02')
    col4.badge(ecoli_sample2, color='green')
    if int(ecoli_sample2) > 200:
        col4.badge(ecoli_sample2, color='orange')
    if int(ecoli_sample2) > 400:
        col4.badge(ecoli_sample2, color='red')
