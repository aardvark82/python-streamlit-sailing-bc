def create_arrow_html(direction, wind_speed=''):
    if direction:
        direction_degrees = {
            'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
            'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
            'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
            'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
        }

        try:
            degree = direction_degrees.get(direction.upper(), 1000)
        except Exception:
            degree = 1000

        if degree == 1000:
            return '<div style="text-align: center;"><div style="width: 20px; height: 20px;background-color: #808080; border-radius: 50%; display: inline-block;"></div></div>'

        if isinstance(wind_speed, (int, float)):
            wind_speed_value = int(wind_speed)
        else:
            wind_speed = str(wind_speed).split()[0]
            wind_speed_value = int(float(wind_speed)) if wind_speed.replace('.', '').strip().isdigit() else 0

        arrow_count = max(0, int(wind_speed_value / 5))

        if arrow_count == 0:
            return '<div style="text-align: center;"><div style="width: 20px; height: 20px; background-color: #1f77b4; border-radius: 50%; display: inline-block;"></div></div>'

        arrow_html = '<div style="text-align: center; white-space: nowrap;">'
        for _ in range(arrow_count):
            arrow_html += f'<div style="width: 0; height: 0; border-left: 10px solid transparent; border-right: 10px solid transparent; border-bottom: 30px solid #1f77b4; display: inline-block; transform: rotate({180+degree}deg); margin: 0 5px;"></div>'
        arrow_html += '</div>'
        return arrow_html
