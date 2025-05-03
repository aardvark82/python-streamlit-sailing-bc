API_KEY_STORMGLASS_IO = '4b108f2a-27f4-11f0-88e2-0242ac130003-4b109010-27f4-11f0-88e2-0242ac130003'

MAKE_LIVE_REQUESTS = True

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
        else:
            # Use the stub data when not making live requests
            data = {
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