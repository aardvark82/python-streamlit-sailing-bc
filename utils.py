import requests
import streamlit as st
from datetime import datetime
import pytz
from timeago import format as timeago_format
from dateutil import parser
from dateutil.tz import gettz


@st.cache_data(ttl=1800)
def cached_fetch_url(url):
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return response


@st.cache_data(ttl=180)
def cached_fetch_url_live(url):
    """Short-TTL fetch for live data like buoy observations (3-minute cache)."""
    response = requests.get(url, timeout=25)
    response.raise_for_status()
    return response


def prettydate(d):
    now_vancouver = datetime.now(pytz.timezone('America/Vancouver'))
    return timeago_format(d, now_vancouver)


def displayStreamlitDateTime(datetime_input, container=None):
    """Accepts a string or datetime object, tries its best at recognizing/parsing it, and displays it in Streamlit format."""
    draw = container or st
    if isinstance(datetime_input, str):
        tzinfos = {
            "PDT": gettz("America/Vancouver"),
            "PST": gettz("America/Vancouver"),
        }
        datetime_van = parser.parse(datetime_input, tzinfos=tzinfos)
    else:
        datetime_van = datetime_input.replace(tzinfo=gettz('America/Vancouver'))

    draw.title(prettydate(datetime_van))
    draw.text(datetime_van)
