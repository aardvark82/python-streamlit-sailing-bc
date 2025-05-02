#  > pip install -r requirements.txt
#  > python -m streamlit run st.py
#  http://localhost:8501/

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import pytz

from datetime import datetime

def prettydate(d):
    d = d.replace(tzinfo=pytz.timezone('America/Vancouver'))

    now_vancouver = datetime.datetime.now(pytz.timezone('America/Vancouver'))  # UTC+7
    diff = now_vancouver - d

    s = diff.seconds
    if diff.days > 7 or diff.days < 0:
        return d.strftime('%d %b %y')
    elif diff.days == 1:
        return '1 day ago'
    elif diff.days > 1:
        return '{} days ago'.format(diff.days)
    elif s <= 1:
        return 'just now'
    elif s < 60:
        return '{} seconds ago'.format(s)
    elif s < 120:
        return '1 minute ago'
    elif s < 3600:
        return '{} minutes ago'.format(int(s/60))
    elif s < 7200:
        return '1 hour ago'
    else:
        return '{} hours ago'.format(int(s/3600))


def displayStreamlitDateTime(datetime):
    """ accepts a string or datetime object, tries its best at recognizing/parsing it, and displays it in Streamlit format."""

    if isinstance(datetime,str):
        datetime = datetime.replace('T', ' ')
        datetime = datetime.replace('Z', '')
        from dateutil import parser
        from dateutil.tz import gettz
        tzinfos = {"PDT": gettz("America/Los_Angeles")}
        datetime = parser.parse(datetime, tzinfos=tzinfos)

    datetime_van = datetime.replace(tzinfo=pytz.timezone('America/Vancouver'))

    st.title(prettydate(datetime_van))
    st.header(datetime_van)

# Selector

def headerbox():
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.button('Halibut Bank', on_click=refresh_all, args=('46146','Halibut Bank'))
    col2.button('Point Atkinson', on_click=refresh_all, args=('WSB','Point Atkinson'))
    col3.button('Pam Rocks', on_click=refresh_all, args=('WAS','Pam Rocks'))
    col4.button('Jericho Beach', on_click=parseJerichoWindHistory,)
    displayPointAtkinsonTides()

def displayWindWarningIfNeeded(wind_speed):
    warning_wind = (wind_speed>9)
    if (warning_wind):
        st.badge("Wind warning", color="orange")


def fetch_point_atkinson_tides():
    # Point Atkinson station ID is 7795
    url = "https://www.tide-forecast.com/locations/Point-Atkinson-British-Columbia/tides/latest"

    try:
        # Using pandas to read the HTML table from the website
        tables = pd.read_html(url)
        # Usually the tide table is the first table on the page
        tide_df = tables[0]
        return tide_df
    except Exception as e:
        st.error(f"Error fetching tide data: {str(e)}")
        return None

def displayPointAtkinsonTides():
    # Set up the Streamlit page
    st.subheader("Tide Information, Vancouver, BC, Point Atkinson")

    # Fetch the tide data
    tide_data = fetch_point_atkinson_tides()

    if tide_data is not None:
        # Convert time to Vancouver timezone
        vancouver_tz = pytz.timezone('America/Vancouver')
        current_time = datetime.now(vancouver_tz)  # This will work with the new import
        st.write(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M %Z')}")

        # Display the tide table
        st.dataframe(tide_data)

        # Get the next tide event
        try:
            next_tide = tide_data.iloc[0]
            st.metric(
                label="Next Tide",
                value=f"{next_tide['Height']}",
                delta=f"{next_tide['Type']} at {next_tide['Time']}"
            )
        except:
            st.warning("Could not determine next tide event")

        # Create a line chart of tide heights
        try:
            st.line_chart(tide_data['Height'])
        except:
            st.warning("Could not create tide height chart")

    else:
        st.error("Unable to fetch tide data. Please try again later.")


def parseJerichoWindHistory():
    headerbox()
    # https://jsca.bc.ca/main/downld02.txt
    url = "https://jsca.bc.ca/main/downld02.txt"
    res = requests.get(url)
    # stupid csv file as 2 first rows as column headers with columns 0,1,13,14 first line missing, fix this
    csv_raw = res.content.decode('utf-8')
    lines = csv_raw.splitlines()
    csv_fixed = '\n'.join(lines[3:]) # drop the 1st 3 rows that is '--------------' as it messes with separator
   # csv_fixed = '\n'.join(lines[:1] + lines[2:])  # drop the 3rd (now 2nd) row that is '--------------' as it messes with separator

   #
    #st.write(csv_fixed)
    #print(csv_fixed)
    import io

    df = pd.read_csv(io.StringIO(csv_fixed) ,header=None, delim_whitespace=True)

    print(f"Number of columns: {len(df.columns)}")
    df.columns = ['Date',
                  'Time',
                  'Temp Out',
                  'Temp Hi',
                  'Temp Low',
                  'Hum Out',
                  'Dew Pt.',
                  'Wind Speed',
                  'Wind Dir',
                  'Wind Run',
                  'Wind Hi Speed',
                  'Wind Hi Dir',
                  'Wind Chill',
                  'Heat Index',
                  'THW Index',
                  'Bar',
                  'Rain',
                  'Rain Rate',
                  'Heat D-D',
                  'Cool D-D',
                  'In Temp',
                  'In Hum',
                  'In Dew',
                  'In Heat',
                  'In EMC',
                  'In Air Density',
                  'Wind Samp',
                  'Wind TX',
                  'IS Recept.',
                  'Arc Int',
                  ]  # adjust the number of columns as needed

# combine 1st and 2nd column date and time, drop first 2 columns
    # Combine date and time columns into a single datetime column
    df['datetime'] = pd.to_datetime(df.iloc[:, 0] + ' ' + df.iloc[:, 1], utc=False)

    # If you want to drop the original date and time columns
    df = df.drop(df.columns[[0, 1]], axis=1)
    # If you want to move the datetime column to the front
    cols = df.columns.tolist()
    df = df[['datetime'] + cols[:-1]]  # excluding the last column since it's datetime

#display time
    datetime_last_measurement = df.iloc[-1, 0]  # -1 for last row, 1 for second column (0-based index)
    displayStreamlitDateTime(datetime_last_measurement)

    # display last values
    temp_out = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)
    wind_speed = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)
    temp_out = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)

    col1, col2, col3, col4, col5 = st.columns(5)
    displayWindWarningIfNeeded(df.iloc[-1, 9])
    col1.metric(label="Wind Speed",     value=df.iloc[-1, 6])
    col2.metric(label="Wind High",      value=df.iloc[-1, 9])
    col3.metric(label="Bar",            value=df.iloc[-1, 14])
    col4.metric(label="Rain",           value=df.iloc[-1, 15])
    col5.metric(label="Temperature",    value=df.iloc[-1, 1])

    #display graph of last 6 hours (12 entries)
    #display graph of last 6 hours (12 entries)

    st.line_chart(
    data=df.tail(12).set_index(df.columns[0])[
        [df.columns[6], df.columns[9], ]
    ]
    )

    st.dataframe(df.tail(12))



def headerbox_menu_deprecated():
    with st.popover("Select a Buoy"):
        c = st.container(border = True)

        buoy = c.selectbox('Choose a Buoy', ['46146', 'WSB', 'WAS'])
        c.write('46146 = Halibut Bank')
        c.write('WSB = Point Atkinson')
        c.write('WAS = Pam Rocks')
        title = 'N/A'

        if buoy == '46146':
            title = 'Halibut Bank'
        if buoy == 'WSB':
            title = 'Point Atkinson'
        if buoy == 'WAS':
            title = 'Pam Rocks'

    refresh_all(buoy = buoy, title = title)

def refresh_all(buoy = '46146', title = 'Halibut Bank - 46146'):
    headerbox()
    url = f'https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html?mapID=02&siteID=14305&stationID={buoy}'
    res = requests.get(url)

    soup = BeautifulSoup(res.content, 'html.parser')

    tables = soup.find_all('table')
    table = soup.find('table', class_='table')
    # parsing
    time = soup.find('span', class_='issuedTime').string

    # --- debug ---
    print('TABLE:')
    #(table.tbody.find_all('tr'))
    for row in table.tbody.find_all('tr'):
        columns = row.find_all('td')
        print('COLUMNS LENGTH:', len(columns))


    # --- debug end ---

    rows = table.tbody.find_all('tr')
    print('TABLE LENGTH:', len(rows))

    data_wind = data_pressure = data_wave_height = data_airtemp = data_waveperiod = data_watertemp = 'N/A'
    if buoy == '46146':
        data_wind = rows[0].find_all('td')[0].text.strip()
        data_pressure = rows[0].find_all('td')[1].text.strip() + 'kPa'

        data_wave_height = rows[1].find_all('td')[0].text.strip() + 'm'
        data_airtemp = rows[1].find_all('td')[1].text.strip() + '°C'

        data_waveperiod = rows[2].find_all('td')[0].text.strip() + 's'
        data_watertemp = rows[2].find_all('td')[1].text.strip() + '°C'

    if buoy == 'WSB': # Point Atkinson
        data_wind = rows[0].find_all('td')[0].text.strip()
    if buoy == 'WAS': # Pam Rocks
        data_wind = rows[0].find_all('td')[0].text.strip()

    # data_wave_height = row.parent.find_all('td')[1]  # last cell in the row
        #data_airtemp = row.parent.find_all('td')[3]  # last cell in the row
        #data_waveperiod = row.parent.find_all('td')[2]  # last cell in the row
        #data_watertemp = row.parent.find_all('td')[5]  # last cell in the row

    st.header('Weather Data for '+ title + ' - ' + buoy)

    displayStreamlitDateTime(time)

    import re
    winds = re.findall(r'\d+', data_wind)
    highest_wind = 0
    if winds:
        highest_wind = int(winds[0])

    displayWindWarningIfNeeded(highest_wind)

    waves = re.findall(r'\d+', data_wave_height)
    highest_wave = 0
    if waves:
        highest_wave = float(waves[0])
    warning_wave = (highest_wave>=1)

    if (warning_wave):
        st.badge("Wave warning", color="orange")


    col1, col2, col3 = st.columns(3)
    col1.metric("Wind", data_wind )
    col2.metric("Wave Height", data_wave_height)
    col3.metric("Air Temp", data_airtemp)
    col1.metric("Water Temp", data_watertemp)
    col2.metric("Wave Period", data_waveperiod)
    col3.metric("Pressure", data_pressure)

    st.write(url)
    # st.code(soup) # debug HTML


#Halibut Bank - 46146
    if buoy == '46146':
        data = pd.DataFrame({
            'latitude': [49.34],
            'longitude': [-123.72]
        })
    if buoy == 'WSB':
        data = pd.DataFrame({
            'latitude': [49.330],
            'longitude': [-123.2646]
        })
    if buoy == 'WAS':
        data = pd.DataFrame({
            'latitude': [49.49],
            'longitude': [-123.3]
        })
    ## Create a map with the data
    st.map(data, zoom=10)

headerbox()
