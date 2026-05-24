"""Buoy data fetcher — parsing logic mirrors st.py::refreshBuoy exactly
so the values written to Cloudflare KV are interchangeable between
the Streamlit app and this helper-app.

Sources: Environment Canada marine current-conditions HTML pages.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import requests
from bs4 import BeautifulSoup

# Station registry — id, human name, whether the station reports waves,
# optional source override (defaults to weather.gc.ca buoy).
# Order drives the UI: first entry = default location for Log/Graph,
# first card on Trends/Reconcile. Pam Rocks is the primary signal for
# the user's sailing-decision workflow.
BUOYS = [
    {"id": "WAS",     "name": "Pam Rocks",     "waves": False},
    {"id": "46146",   "name": "Halibut Bank",  "waves": True},
    {"id": "46304",   "name": "English Bay",   "waves": True},
    {"id": "WSB",     "name": "Point Atkinson", "waves": False},
    {"id": "JERICHO", "name": "Jericho Wind",  "waves": False, "source": "jericho"},
]
BUOY_BY_ID = {b["id"]: b for b in BUOYS}

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


@dataclass
class BuoyReading:
    buoy_id: str
    name: str
    issued_time: str           # raw "issuedTime" string from the page
    wind_text: str             # raw, e.g. "W 17 gust 20"
    direction: Optional[str]   # compass label e.g. "W"
    wind_speed: int            # highest knots seen in the text
    wave_height_m: Optional[float]
    air_temp: Optional[str]
    water_temp: Optional[str]
    wave_period_s: Optional[str]


def _parse_wind(wind_text: str) -> tuple[Optional[str], int]:
    """Mirror of refreshBuoy::parse_wind_data — direction + highest int seen."""
    if not isinstance(wind_text, str) or not wind_text.strip():
        return None, 0
    parts = wind_text.strip().split()
    if not parts:
        return None, 0
    direction = parts[0]
    nums = [int(n) for n in re.findall(r"\d+", wind_text)]
    return direction, (max(nums) if nums else 0)


def _parse_wave_m(text: str) -> Optional[float]:
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text or "")
    if not matches:
        return None
    try:
        return float(matches[0])
    except ValueError:
        return None


def _fetch_jericho() -> BuoyReading:
    """Davis weather station CSV from JSCA. Parsing identical to main app's
    parseJerichoWindHistory so values are interchangeable. Wind speed and
    high-speed columns are treated as the highest reading in the last row
    (same convention as buoy parsing — _parse_wind picks max of all ints)."""
    url = "https://jsca.bc.ca/main/downld02.txt"
    r = requests.get(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/plain, text/csv, */*",
        "Accept-Language": "en-US,en;q=0.9",
    }, timeout=25)
    r.raise_for_status()

    csv_raw = r.content.decode("utf-8")
    csv_fixed = "\n".join(csv_raw.splitlines()[3:])
    column_names = [
        "Date", "Time", "Temp Out", "Temp Hi", "Temp Low", "Hum Out",
        "Dew Pt.", "Wind Speed", "Wind Dir", "Wind Run", "Wind Hi Speed",
        "Wind Hi Dir", "Wind Chill", "Heat Index", "THW Index", "Bar",
        "Rain", "Rain Rate", "Heat D-D", "Cool D-D", "In Temp", "In Hum",
        "In Dew", "In Heat", "In EMC", "In Air Density", "Wind Samp",
        "Wind TX", "IS Recept.", "Arc Int",
    ]
    df = pd.read_csv(io.StringIO(csv_fixed), header=None, names=column_names,
                     sep=r"\s+", engine="python", on_bad_lines="skip")
    last = df.iloc[-1]
    try:
        wind = float(last["Wind Speed"])
        hi = float(last["Wind Hi Speed"])
        wind_speed = int(round(max(wind, hi)))
    except (ValueError, TypeError):
        wind_speed = 0
    direction = str(last["Wind Dir"]).strip() or None
    wind_text = f"{direction or '?'} {last['Wind Speed']} hi {last['Wind Hi Speed']}"
    return BuoyReading(
        buoy_id="JERICHO",
        name="Jericho Wind",
        issued_time=f"{last['Date']} {last['Time']}",
        wind_text=wind_text,
        direction=direction,
        wind_speed=wind_speed,
        wave_height_m=None,
        air_temp=str(last["Temp Out"]) + "°C" if "Temp Out" in last else None,
        water_temp=None,
        wave_period_s=None,
    )


def fetch_buoy(buoy_id: str) -> BuoyReading:
    """Dispatch by source — gc.ca buoys by default, jericho for JSCA CSV."""
    meta = BUOY_BY_ID[buoy_id]
    if meta.get("source") == "jericho":
        return _fetch_jericho()

    url = (
        "https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html"
        f"?mapID=02&siteID=14305&stationID={buoy_id}"
    )
    resp = requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html"}, timeout=20)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.content, "html.parser")
    table = soup.find("table", class_="table")
    if table is None:
        raise RuntimeError(f"No conditions table found for buoy {buoy_id}")
    issued_span = soup.find("span", class_="issuedTime")
    issued_time = issued_span.string.strip() if issued_span and issued_span.string else ""

    rows = table.tbody.find_all("tr")
    wind_text = rows[0].find_all("td")[0].text.strip()

    wave_m = air_temp = water_temp = wave_period = None
    if meta["waves"] and len(rows) >= 3:
        wave_raw = rows[1].find_all("td")[0].text.strip()
        wave_m = _parse_wave_m(wave_raw)
        air_temp = rows[1].find_all("td")[1].text.strip() + "°C"
        wave_period = rows[2].find_all("td")[0].text.strip() + "s"
        water_temp = rows[2].find_all("td")[1].text.strip() + "°C"

    direction, wind_speed = _parse_wind(wind_text)

    return BuoyReading(
        buoy_id=buoy_id,
        name=meta["name"],
        issued_time=issued_time,
        wind_text=wind_text,
        direction=direction,
        wind_speed=wind_speed,
        wave_height_m=wave_m,
        air_temp=air_temp,
        water_temp=water_temp,
        wave_period_s=wave_period,
    )
