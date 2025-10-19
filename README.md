# python-streamlit-sailing-bc
Real-time Tides, Wind and Wave Height for sailors out of vancouver, BC

# started May 2025 by aardvark82 in West Vancouver, BC

# GitHub repo at
https://github.com/aardvark82/python-streamlit-sailing-bc

# public app available from GitHub at
https://python-app-sailing-bc-nckqtfynerhhf26ujtt5u6.streamlit.app/

# data sources (without permission)

## Tides data from 

## Live Buoy data from government of Canada at Halibut Bank (parsed with BeautifulSoup4)
https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html?mapID=02&siteID=14305&stationID=46146

## live weather API from openweathermap.org
https://home.openweathermap.org/
- 1000 calls/day for free

## live wind data from land stations WAS and WSB (parsed with BeautifulSoup4)
https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html?mapID=02&siteID=14305&stationID=WAS
https://www.weather.gc.ca/marine/weatherConditions-currentConditions_e.html?mapID=02&siteID=14305&stationID=WSB

## Wind data in English Bay (parsed with pandas read_csv)
The excellent Jericho wind sailing page 
https://jsca.bc.ca/services/weather/
publishes historical data in a csv at
https://jsca.bc.ca/main/downld02.txt

