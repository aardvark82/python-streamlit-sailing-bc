API_KEY_STORMGLASS_IO = '4b108f2a-27f4-11f0-88e2-0242ac130003-4b109010-27f4-11f0-88e2-0242ac130003'

MAKE_LIVE_REQUESTS = False

def fetchTidesPointAtkinson(container=None):
    """Fetch tide data for Point Atkinson from Stormglass API"""
    import requests
    import pandas as pd
    from datetime import datetime, timedelta
    import pytz
    import json

    try:
        # Point Atkinson coordinates
        lat = 49.3370
        lon = -123.2630

        if MAKE_LIVE_REQUESTS:
            # Stormglass API configuration
            base_url = "https://api.stormglass.io/v2/tide/extremes/point"
            api_key = "4b108f2a-27f4-11f0-88e2-0242ac130003-4b109010-27f4-11f0-88e2-0242ac130003"
            
            vancouver_tz = pytz.timezone('America/Vancouver')
            now = datetime.now(vancouver_tz)
            
            # Get 4 days before and 4 days after current time
            start_date = now - timedelta(days=1)
            end_date = now + timedelta(days=1)

            params = {
                'lat': lat,
                'lng': lon,
                'start': start_date.strftime('%Y-%m-%dT%H:%M:%S+00:00'),
                'end': end_date.strftime('%Y-%m-%dT%H:%M:%S+00:00')
            }

            headers = {
                'Authorization': api_key
            }

            response = requests.get(base_url, params=params, headers=headers, timeout=10)
            
            if response.status_code != 200:
                error_msg = f"Failed to fetch tide data. Status code: {response.status_code}"
                if container:
                    container.error(error_msg)
                return None

            data = response.json()
            print (data)

        else:
            container.warning("Using stub data for tide data")
            # Use the stub data when not making live requests
            data1 = { # 2 days of data, 4 points per day
                "data": [
                    {"height": 1.6281896122083954, "time": "2025-04-29T02:59:00+00:00", "type": "high"},
                    {"height": 0.09792586003904245, "time": "2025-04-29T08:16:00+00:00", "type": "low"},
                    {"height": 1.374891079134516, "time": "2025-04-29T13:10:00+00:00", "type": "high"},
                    {"height": -2.7679098753738876, "time": "2025-04-29T20:27:00+00:00", "type": "low"},
                    {"height": 1.716952818959448, "time": "2025-04-30T03:55:00+00:00", "type": "high"},
                    {"height": 0.32124120920566857, "time": "2025-04-30T09:10:00+00:00", "type": "low"},
                    {"height": 1.2659443022297774, "time": "2025-04-30T13:45:00+00:00", "type": "high"},
                    {"height": -2.755521186344923, "time": "2025-04-30T21:11:00+00:00", "type": "low"}
                ]
            }
            # 2 days of data, 4 points per day
            data = {'data': [{'height': 1.6432596903918761, 'time': '2025-05-02T05:51:00+00:00', 'type': 'high'},
                             {'height': 0.5114702000679019, 'time': '2025-05-02T11:16:00+00:00', 'type': 'low'},
                             {'height': 0.8861352640591091, 'time': '2025-05-02T15:04:00+00:00', 'type': 'high'},
                             {'height': -2.339105387349193, 'time': '2025-05-02T22:49:00+00:00', 'type': 'low'},
                             {'height': 1.551146455775455, 'time': '2025-05-03T06:52:00+00:00', 'type': 'high'},
                             {'height': 0.4471043214497481, 'time': '2025-05-03T12:40:00+00:00', 'type': 'low'},
                             {'height': 0.624905587275962, 'time': '2025-05-03T15:55:00+00:00', 'type': 'high'},
                             {'height': -2.0247559154532104, 'time': '2025-05-03T23:44:00+00:00', 'type': 'low'}
                             ]
                    }
        # 'meta': {'cost': 1, 'dailyQuota': 10, 'datum': 'MSL', 'end': '2025-05-04 01:00', 'lat': 49.337, 'lng': -123.263, 'offset': 0, 'requestCount': 6, 'start': '2025-05-02 01:00', 'station': {'distance': 1, 'lat': 49.34, 'lng': -123.25, 'name': 'station', 'source': 'ticon3('}}'


        # Convert predictions to pandas DataFrame
        predictions = []
        for prediction in data['data'][:8]:  # Take only first 8 points
            dt = pd.to_datetime(prediction['time'])
            dt = dt.tz_convert('America/Vancouver')
            predictions.append({
                'Time (PDT)& Date': dt,
                'Height': float(prediction['height'])
            })

        # Create DataFrame and sort by time
        tide_df = pd.DataFrame(predictions)
        tide_df = tide_df.sort_values('Time (PDT)& Date', ignore_index=True)
        print(tide_df)  # Add this line at the end of fetchTidesPointAtkinson() to see the output
        return tide_df

    except Exception as e:
        error_msg = f"Error fetching tide data: {str(e)}"
        if container:
            container.error(error_msg)
        print(error_msg)
        return None