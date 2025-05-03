#  > pip install -r requirements.txt
#  > python -m streamlit run st.py
# http://localhost:8501/
# http://python-app-sailing-bc-nckqtfynerhhf26ujtt5u6


import streamlit as st
import requests
from bs4 import BeautifulSoup
import pytz
import numpy as np
import pandas as pd

from datetime import datetime
import pytz

def prettydate(d):

    now_vancouver = datetime.now(pytz.timezone('America/Vancouver'))
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

from dateutil import parser
from dateutil.tz import gettz

def displayStreamlitDateTime(datetime, container=None):
    """ accepts a string or datetime object, tries its best at recognizing/parsing it, and displays it in Streamlit format."""
    draw = container
    if isinstance(datetime,str):
        tzinfos = {"PDT": gettz("America/Vancouver")    ,
                   "PST": gettz("America/Vancouver"),
        }
        print("Parsing time ", datetime)
        datetime_van = parser.parse(datetime, tzinfos=tzinfos)
        datetime_van = datetime_van

    else:
        datetime_van = datetime.replace(tzinfo=gettz('America/Vancouver'))

    draw.title(prettydate(datetime_van))
    draw.text(datetime_van)

# Selector

def headerbox():
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Tides", "Jericho Beach", "Halibut Bank", "Point Atkinson", "Pam Rocks"])
    displayPointAtkinsonTides(container=tab1)
    parseJerichoWindHistory(container=tab2)
    refreshBuoy('46146','Halibut Bank', container=tab3)
    refreshBuoy('WSB', 'Point Atkinson', container=tab4)
    refreshBuoy('WAS', 'Pam Rocks', container=tab5)
#    col1, col2, col3, col4, col5 = st.columns(5)
    #col1.button('Halibut Bank',     on_click=refresh_buoys, args=('46146','Halibut Bank'))
    #col2.button('Point Atkinson',   on_click=refresh_buoys, args=('WSB','Point Atkinson'))
    #col3.button('Pam Rocks',        on_click=refresh_buoys, args=('WAS','Pam Rocks'))
    #col4.button('Jericho Beach',    on_click=parseJerichoWindHistory,)
    #col4.button('Tides',            on_click=displayPointAtkinsonTides, )
    #displayPointAtkinsonTides()

def displayWindWarningIfNeeded(wind_speed, container=None):
    """ above 9 knots """
    if container:
        draw = container
    else:
        draw = st
    warning_wind = (wind_speed>9)
    if warning_wind:
        draw.badge("Wind warning", color="orange")


def fetchTidesPointAtkinson(container=None):
    # Point Atkinson station ID is 7795
    url = "https://www.tide-forecast.com/locations/Point-Atkinson-British-Columbia/tides/latest"
    if container:
        draw = container
    else:
        draw = st

    try:
        # Using pandas to read the HTML table from the website
        tables = pd.read_html(url)
        # Usually the tide table is the first table on the page
        tide_df = tables[0]
        return tide_df
    except Exception as e:
        draw.error(f"Error fetching tide data: {str(e)}")
        return None


def create_natural_tide_chart(tide_df, container=None):
    if container:
        draw = container
    else:
        draw = st

#### Time cleaning

    #Ah, I see the issue. The date string format from tide-forecast.com is in a non-standard format. Let's fix the datetime parsing:
    # Convert time strings to datetime objects with proper parsing
    def parse_tide_datetime(time_str):
        try:
            if isinstance(time_str, str):  # Check if it's a string
                # Example format: "4:20 AM(Fri 02 May)"
                if '(' in time_str:
                    time_part, date_part = time_str.split('(')
                    date_part = date_part.rstrip(')')
                    # Combine in the correct order for parsing
                    datetime_str = f"{time_part.strip()} {date_part}"
                    # Add current year since it's missing from the input
                    current_year = datetime.now().year
                    datetime_str = f"{datetime_str} {current_year}"

                    # Parse with the exact format matching the input
                    vancouver_tz = pytz.timezone('America/Vancouver')
                    dt = pd.to_datetime(datetime_str, format="%I:%M %p %a %d %b %Y")
                    if dt.tz is None:
                        dt =  dt.tz_localize(vancouver_tz)
                    return dt
                else:
                    # Handle other formats if needed
                    vancouver_tz = pytz.timezone('America/Vancouver')
                    dt = pd.to_datetime(time_str)
                    if dt.tz is None:
                        return dt.tz_localize(vancouver_tz)
                    return dt
        except (ValueError, AttributeError) as e:
            print(f"Error parsing datetime for value: {time_str}")
            return pd.NaT  # Return NaT (Not a Time) for any parsing errors
        return pd.NaT  # Return NaT for non-string inputs


    # Cleanup columns
    print("----------------------------------------------------------------------------")
    print("ALL TIDES")
    print("----------------------------------------------------------------------------")
    print(tide_df)

    print("Columns in tide_df:", tide_df.columns)
    # The datetime is already in the Time column, so we'll use that directly
    # First convert to datetime
    tide_df = tide_df.rename(columns={'Time (PDT)& Date': 'datetime'})

    tide_df['datetime'] = tide_df['datetime'].apply(parse_tide_datetime)
    print("----------------------------------------------------------------------------")
    print("ALL TIDES CLEAN TIME")
    print("----------------------------------------------------------------------------")
    print(tide_df)

#### Height cleaning
    # Clean the height data - remove any 'm' or other units if present
    # Debug: Print the data types and check for any non-numeric values
    print("Height column data:", tide_df['Height'])

    def extract_meters(height_str):
        try:
            # Extract the number before 'm'
            meters = float(height_str.split('m')[0].strip())
            return meters
        except (ValueError, AttributeError) as e:
            print(f"Error parsing height from value: {height_str}")
            return None

    tide_df['Height'] = tide_df['Height'].astype(str).apply(extract_meters)

    if tide_df['Height'].isnull().any():
        # Optionally, report or handle NAs here
        # For now, let's forward-fill them (or use .dropna())
        tide_df['Height'] = tide_df['Height'].fillna(method='ffill')
    print("----------------------------------------------------------------------------")
    print("ALL TIDES CLEAN HEIGHT")
    print("----------------------------------------------------------------------------")
    print(tide_df)

### SMOOTH INTERPOLATION

    # Create smooth interpolation for better visualization
    # Resample to 15-minute intervals
    min_time = tide_df['datetime'].min()
    max_time = tide_df['datetime'].max()

    if pd.isna(min_time) or pd.isna(max_time):
        draw.error("Invalid time range in tide data")
        return

    # Create timezone-aware date range
    full_index = pd.date_range(
        start=min_time,
        end=max_time,
        freq='15min'
    )

    # Make the index timezone aware if it isn't already
    vancouver_tz = pytz.timezone('America/Vancouver')
    if full_index.tz is None:
        full_index = full_index.tz_localize(vancouver_tz)

    # Create interpolated series
    tide_interpolated = pd.DataFrame(index=full_index)

    # Convert to timestamps for interpolation
    x_timestamps = tide_df['datetime'].astype(np.int64) // 10 ** 9
    x_new_timestamps = full_index.astype(np.int64) // 10 ** 9

    # Perform interpolation
    tide_interpolated['Height'] = np.interp(
        x=x_new_timestamps,
        xp=x_timestamps,
        fp=tide_df['Height'].values
    )
    print("----------------------------------------------------------------------------")
    print("tide_interpolated")
    print("----------------------------------------------------------------------------")
    print(tide_interpolated)

    # Create the visualization
    draw.subheader("🌊 Point Atkinson Tide Chart")

    # Use Plotly for better interactivity
    import plotly.graph_objects as go

    fig = go.Figure()

    # Add the smooth tide line
    fig.add_trace(go.Scatter(
        x=tide_interpolated.index,
        y=tide_interpolated['Height'],
        name='Tide Level',
        line=dict(color='#2E86C1', width=3),
        fill='tozeroy',  # Fill to zero
        fillcolor='rgba(46, 134, 193, 0.2)'  # Light blue fill
    ))

    # Add actual data points
    fig.add_trace(go.Scatter(
        x=tide_df['datetime'],
        y=tide_df['Height'],
        mode='markers',
        name='Measured Points',
        marker=dict(
            size=8,
            color='#1A5276',
            symbol='circle'
        )
    ))

    # Customize the layout
    fig.update_layout(
        title={
            'text': 'Tide Levels at Point Atkinson',
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        xaxis_title="Time",
        yaxis_title="Height (meters)",
        hovermode='x unified',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(128,128,128,0.2)',
            zeroline=True,
            zerolinewidth=2,
            zerolinecolor='rgba(128,128,128,0.5)'
        )
    )

    # Add current time marker
    vancouver_tz = pytz.timezone('America/Vancouver')
    current_time = datetime.now(vancouver_tz)
    current_time_ts = current_time.timestamp() * 1000  # multiply by 1000 for milliseconds

    fig.add_vline(
        x=current_time_ts,
        line_width=2,
        line_dash="dash",
        line_color="red",
        annotation_text="Current Time",
        annotation_position="top right"
    )



    # Show the plot in Streamlit
    draw.plotly_chart(fig, use_container_width=True)

    # Add tide statistics
    col1, col2, col3 = draw.columns(3)

    current_height = np.interp(
        current_time.timestamp(),
        tide_interpolated.index.astype(np.int64) // 10 ** 9,
        tide_interpolated['Height']
    )

    col1.metric(
        "Current Tide Level",
        f"{current_height:.2f}m",
    )

    # Check if we have the tide data columns before trying to display next tide info
    if 'datetime' in tide_df.columns:
        try:
            next_tide = tide_df[tide_df['datetime'] > current_time].iloc[0]
            time_diff = (next_tide['datetime'] - current_time)

            # Use Height only since we don't have Type information
            col2.metric(
                "Next Tide",
                f"{next_tide['Height']}m",
                f"in {time_diff.total_seconds() // 3600:.0f}h {(time_diff.total_seconds() // 60 % 60):.0f}m"
            )
        except (IndexError, KeyError):
            col2.metric(
                "Next Tide",
                "No data available",
                ""
            )
    else:
        col2.metric(
            "Next Tide",
            "No data available",
            ""
        )

    if 'Height' in tide_df.columns:
        daily_range = tide_df['Height'].max() - tide_df['Height'].min()
        col3.metric(
            "Daily Tide Range",
            f"{daily_range:.2f}m"
        )
    else:
        col3.metric(
            "Daily Tide Range",
            "No data available"
        )


# Modify your displayPointAtkinsonTides function to use the new visualization
def displayPointAtkinsonTides(container=None):
    if container:
        draw = container
    else:
        draw = st

    # Fetch the tide data
    tide_data = fetchTidesPointAtkinson()

    if tide_data is not None:
        # Create the natural tide chart
        create_natural_tide_chart(tide_data, container)
    else:
        draw.error("Unable to fetch tide data. Please try again later.")

def displayPointAtkinsonTidesOldUgly(container=None):
    if container:
        draw = container
    else:
        draw = st

    # Set up the Streamlit page
    draw.subheader("Tide Information, Vancouver, BC, Point Atkinson")

    # Fetch the tide data
    tide_data = fetchTidesPointAtkinson()

    if tide_data is not None:
        # Convert time to Vancouver timezone
        vancouver_tz = pytz.timezone('America/Vancouver')
        current_time = datetime.now(vancouver_tz)  # This will work with the new import

        displayStreamlitDateTime(current_time, container=draw)
        #draw.write(f"Current time: {current_time.strftime('%Y-%m-%d %H:%M %Z')}")

        # Display the tide table
        draw.dataframe(tide_data)

        # Get the next tide event
        try:
            next_tide = tide_data.iloc[0]
            draw.metric(
                label="Next Tide",
                value=f"{next_tide['Height']}",
                delta=f"{next_tide['Type']} at {next_tide['Time']}"
            )
        except:
            draw.warning("Could not determine next tide event")

        # Create a line chart of tide heights
        try:
            draw.line_chart(tide_data['Height'])
        except:
            draw.warning("Could not create tide height chart")

    else:
        draw.error("Unable to fetch tide data. Please try again later.")


def parseJerichoWindHistory(container = None):

    container.subheader("Jericho Beach Wind History")

    if container:
        draw = container
    else:
        draw = st
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
    displayStreamlitDateTime(datetime_last_measurement, draw)

    # display last values
    temp_out = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)
    wind_speed = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)
    temp_out = df.iloc[-1, 1]  # -1 for last row, 1 for second column (0-based index)

    col1, col2, col3, col4, col5 = draw.columns(5)
    displayWindWarningIfNeeded(df.iloc[-1, 9])
    col1.metric(label="Wind Speed",     value=df.iloc[-1, 6])
    col2.metric(label="Wind High",      value=df.iloc[-1, 9])
    col3.metric(label="Bar",            value=df.iloc[-1, 14])
    col4.metric(label="Rain",           value=df.iloc[-1, 15])
    col5.metric(label="Temperature",    value=df.iloc[-1, 1])

    #display graph of last 6 hours (12 entries)
    #display graph of last 6 hours (12 entries)

    draw.line_chart(
    data=df.tail(12).set_index(df.columns[0])[
        [df.columns[6], df.columns[9], ]
    ]
    )

    draw.dataframe(df.tail(12))



def headerboxMenuDeprecated():
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

    refreshBuoy(buoy = buoy, title = title)

def refreshBuoy(buoy = '46146', title = 'Halibut Bank - 46146', container = None):
    if container:
        draw = container
    else:
        draw = st
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
        # print('COLUMNS LENGTH:', len(columns))


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

    draw.subheader('Weather Data for '+ title + ' - ' + buoy)

    displayStreamlitDateTime(time, draw)

    import re
    winds = re.findall(r'\d+', data_wind)
    highest_wind = 0
    if winds:
        highest_wind = int(winds[0])

    displayWindWarningIfNeeded(highest_wind, container=draw)

    waves = re.findall(r'\d+', data_wave_height)
    highest_wave = 0
    if waves:
        highest_wave = float(waves[0])
    warning_wave = (highest_wave>=1)

    if (warning_wave):
        draw.badge("Wave warning", color="orange")


    col1, col2, col3 = draw.columns(3)
    col1.metric("Wind", data_wind )
    col2.metric("Wave Height", data_wave_height)
    col3.metric("Air Temp", data_airtemp)
    col1.metric("Water Temp", data_watertemp)
    col2.metric("Wave Period", data_waveperiod)
    col3.metric("Pressure", data_pressure)

    draw.write(url)
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
    draw.map(data, zoom=10)

headerbox()