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


def displayStreamlitDateTime(datetime_input, container=None, label="Last updated"):
    """Render a prominent color-coded staleness badge for a datetime/string.
    Replaces the older title+text layout so every page shows freshness in
    the same harmonised format. The optional `label` lets callers say
    'Issued', 'Sampled', etc."""
    draw = container or st
    if isinstance(datetime_input, str):
        tzinfos = {
            "PDT": gettz("America/Vancouver"),
            "PST": gettz("America/Vancouver"),
        }
        datetime_van = parser.parse(datetime_input, tzinfos=tzinfos)
    else:
        datetime_van = datetime_input.replace(tzinfo=gettz('America/Vancouver'))

    display_last_updated_badge(draw, datetime_van, label=label)


def _relative_time_phrase(secs):
    """Convert a non-negative second delta into a friendly relative phrase."""
    if secs < 0:
        return 'just now'
    if secs < 60:
        return f'{secs}s ago'
    if secs < 3600:
        return f'{secs // 60}min ago'
    if secs < 86400:
        h = secs // 3600
        return f'{h} hour ago' if h == 1 else f'{h} hours ago'
    d = secs // 86400
    return f'{d} day ago' if d == 1 else f'{d} days ago'


def display_last_updated_badge(container, last_seen, label="Last updated", now=None):
    """Render a prominent, color-coded staleness banner at the top of a page.

    Color buckets (Vancouver-local relative age):
      < 15 min     → green   (fresh)
      < 1 hour     → orange  (recent)
      < 6 hours    → red     (stale)
      otherwise    → gray    (very stale / unknown)

    `last_seen` accepts datetime (aware or naive UTC), unix timestamp, or None.
    """
    draw = container or st
    van_tz = pytz.timezone('America/Vancouver')
    now = now or datetime.now(van_tz)

    target = None
    if last_seen is not None:
        try:
            if isinstance(last_seen, (int, float)):
                target = datetime.fromtimestamp(int(last_seen), tz=pytz.UTC).astimezone(van_tz)
            elif isinstance(last_seen, datetime):
                if last_seen.tzinfo is None:
                    target = van_tz.localize(last_seen)
                else:
                    target = last_seen.astimezone(van_tz)
        except Exception:
            target = None

    if target is None:
        text, color, bg = '—', '#374151', '#e5e7eb'
    else:
        secs = int((now - target).total_seconds())
        text = _relative_time_phrase(secs)
        if secs < 15 * 60:
            color, bg = '#0a6e3a', '#d4f3df'   # green
        elif secs < 60 * 60:
            color, bg = '#8a5a00', '#fff4cf'   # orange
        elif secs < 6 * 3600:
            color, bg = '#9c2027', '#fde2e3'   # red
        else:
            color, bg = '#374151', '#e5e7eb'   # gray

    draw.markdown(
        f'<div style="background:{bg};color:{color};'
        f'padding:0.6rem 1rem;border-radius:10px;'
        f'font-size:1.15rem;font-weight:700;display:inline-block;'
        f'margin-bottom:0.6rem;">'
        f'{label}: {text}</div>',
        unsafe_allow_html=True,
    )
