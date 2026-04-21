"""Wind direction arrow rendering for the sailing dashboard.

Produces a clean SVG arrow pointing in the downwind direction, colored by
wind speed. Accepts the 16-point compass abbreviations (N/NNE/.../NNW) plus
common long-form names ("northwesterly", "southeast", "variable") and is
tolerant of whitespace, case, and trailing punctuation.
"""

import re


_DIRECTION_DEGREES = {
    'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
    'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
    'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
    'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5,
}

# Long-form / alternative spellings → canonical 16-point abbreviation
_LONG_FORM = {
    'NORTH': 'N', 'NORTHERLY': 'N',
    'NORTHEAST': 'NE', 'NORTHEASTERLY': 'NE', 'NORTH EAST': 'NE',
    'EAST': 'E', 'EASTERLY': 'E',
    'SOUTHEAST': 'SE', 'SOUTHEASTERLY': 'SE', 'SOUTH EAST': 'SE',
    'SOUTH': 'S', 'SOUTHERLY': 'S',
    'SOUTHWEST': 'SW', 'SOUTHWESTERLY': 'SW', 'SOUTH WEST': 'SW',
    'WEST': 'W', 'WESTERLY': 'W',
    'NORTHWEST': 'NW', 'NORTHWESTERLY': 'NW', 'NORTH WEST': 'NW',
    'NORTH NORTHEAST': 'NNE', 'NORTH NORTHEASTERLY': 'NNE',
    'EAST NORTHEAST': 'ENE', 'EAST NORTHEASTERLY': 'ENE',
    'EAST SOUTHEAST': 'ESE', 'EAST SOUTHEASTERLY': 'ESE',
    'SOUTH SOUTHEAST': 'SSE', 'SOUTH SOUTHEASTERLY': 'SSE',
    'SOUTH SOUTHWEST': 'SSW', 'SOUTH SOUTHWESTERLY': 'SSW',
    'WEST SOUTHWEST': 'WSW', 'WEST SOUTHWESTERLY': 'WSW',
    'WEST NORTHWEST': 'WNW', 'WEST NORTHWESTERLY': 'WNW',
    'NORTH NORTHWEST': 'NNW', 'NORTH NORTHWESTERLY': 'NNW',
    # No-space compound forms sometimes emitted by GPT
    'NORTHNORTHEAST': 'NNE', 'EASTNORTHEAST': 'ENE',
    'EASTSOUTHEAST': 'ESE', 'SOUTHSOUTHEAST': 'SSE',
    'SOUTHSOUTHWEST': 'SSW', 'WESTSOUTHWEST': 'WSW',
    'WESTNORTHWEST': 'WNW', 'NORTHNORTHWEST': 'NNW',
}


def _normalize_direction(direction):
    """Return a canonical 16-point abbreviation, or None if not recognized."""
    if direction is None:
        return None
    s = str(direction).strip().upper()
    # Strip trailing punctuation (e.g. "WSW," or "SW.")
    s = re.sub(r'[^A-Z ]+$', '', s).strip()
    if not s:
        return None
    if s in _DIRECTION_DEGREES:
        return s
    if s in _LONG_FORM:
        return _LONG_FORM[s]
    # Try compressing whitespace (e.g. "WEST  SOUTHWEST" → "WEST SOUTHWEST")
    compact = re.sub(r'\s+', ' ', s)
    if compact in _LONG_FORM:
        return _LONG_FORM[compact]
    return None


def _extract_wind_speed_kts(wind_speed):
    """Pull an integer knot value out of any reasonable input."""
    if wind_speed is None:
        return 0
    if isinstance(wind_speed, (int, float)):
        try:
            return int(wind_speed)
        except (ValueError, OverflowError):
            return 0
    s = str(wind_speed).strip()
    if not s:
        return 0
    nums = re.findall(r'\d+', s)
    if not nums:
        return 0
    # Use the largest number — handles both "15" and "10-15" and "15 gust 25"
    return max(int(n) for n in nums)


def _color_for_speed(kts):
    """Return an arrow fill color based on knots."""
    if kts < 5:
        return '#4a90e2'   # calm        <5 — blue
    if kts < 10:
        return '#27ae60'   # light       5-10 — green
    if kts < 20:
        return '#f39c12'   # moderate    10-20 — orange
    if kts < 30:
        return '#e74c3c'   # strong      20-30 — red
    return '#111111'       # dangerous   30+ — black


def create_arrow_html(direction, wind_speed=''):
    """Render an SVG arrow HTML fragment.

    - Arrow points in the DOWNWIND direction (i.e. where the wind is blowing to),
      rotated from the compass bearing the wind is coming from + 180°.
    - Fill color reflects wind speed bucket.
    - Falls back to a neutral grey dot if direction can't be parsed
      (e.g. 'V' for variable, or unrecognized text).
    """
    canonical = _normalize_direction(direction)
    kts = _extract_wind_speed_kts(wind_speed)

    if canonical is None:
        return (
            '<div style="text-align:center;">'
            '<div title="Variable / unknown direction" '
            'style="width:22px;height:22px;background:#9aa5b1;'
            'border-radius:50%;display:inline-block;'
            'box-shadow:0 1px 3px rgba(0,0,0,0.2);"></div>'
            '</div>'
        )

    degree = _DIRECTION_DEGREES[canonical]
    # 0° in the data = wind FROM north. Arrow should point downwind (south),
    # so rotate by 180 + degree. SVG 0° is up, clockwise positive.
    rotation = (180 + degree) % 360
    color = _color_for_speed(kts)

    # Beautiful arrow: tapered shaft + clean triangular head. ViewBox is centered
    # so the rotation pivots around the arrow's geometric center.
    svg = f'''
    <svg width="56" height="56" viewBox="-28 -28 56 56"
         xmlns="http://www.w3.org/2000/svg"
         style="display:block;margin:0 auto;">
      <g transform="rotate({rotation})">
        <!-- Soft drop shadow -->
        <defs>
          <filter id="arrShadow" x="-50%" y="-50%" width="200%" height="200%">
            <feGaussianBlur in="SourceAlpha" stdDeviation="1"/>
            <feOffset dx="0" dy="1" result="offsetblur"/>
            <feComponentTransfer>
              <feFuncA type="linear" slope="0.35"/>
            </feComponentTransfer>
            <feMerge>
              <feMergeNode/>
              <feMergeNode in="SourceGraphic"/>
            </feMerge>
          </filter>
        </defs>
        <!-- Arrow shaft + head as a single path -->
        <path d="M 0 -22
                 L 9 -6
                 L 3.5 -6
                 L 3.5 22
                 L -3.5 22
                 L -3.5 -6
                 L -9 -6 Z"
              fill="{color}"
              stroke="#2c3e50"
              stroke-width="1"
              stroke-linejoin="round"
              filter="url(#arrShadow)"/>
      </g>
    </svg>
    '''

    label = f'<div style="text-align:center;font-size:11px;color:#6b7280;margin-top:-4px;">{canonical} · {kts}kts</div>'
    return f'<div style="text-align:center;">{svg}{label}</div>'
