# Whale-watching fleet — Vancouver area
### LLM brief: paste this into Claude / Cursor / Copilot to build a client

You're being asked to fetch the **current position, name, last-update time,
heading and speed** for a curated list of whale-watching vessels operating
out of Vancouver, BC. Use this document as the full spec — every fact you
need (data sources, fleet membership, field semantics, MMSI overrides) is
below.

---

## Goal

Produce, for each boat in the fleet, a record like:

```json
{
  "name": "Salish Sea Freedom",
  "operator": "Prince of Whales",
  "mmsi": 316042213,
  "latitude": 49.286,
  "longitude": -123.118,
  "speed_kts": 18.4,
  "heading_deg": 285,
  "heading_compass": "WNW",
  "last_update_utc": "2026-05-29T01:17:16Z",
  "age_minutes": 3,
  "stale": false
}
```

`stale = true` when `age_minutes > 30` (AIS broadcasts every 2–10 s
underway; > 30 min idle means moored, docked, or out of receiver range).

---

## Data sources (pick one)

### Option A — VesselAPI Ship Tracking (REST, polling)

Simplest, no WebSocket. Per-vessel HTTP GET.

**Auth:** OAuth-style API key passed as header `x-api-key: <key>`.
Free tier ~250 calls/day; the helper-app rotates between a primary
and backup key on HTTP 429 (rate-limited).

**Endpoint:**
```
GET https://api.vesselapi.com/vessel/{mmsi}/position?filter.idType=mmsi
```

**Response envelope** (note the camelCase wrapper):
```json
{
  "vesselPosition": {
    "mmsi": 316042213,
    "latitude": 49.286,
    "longitude": -123.118,
    "speed": 18.4,                  // knots
    "course": 285,                  // degrees true
    "heading": 287,                 // separate column; usually equal-ish
    "timestamp": "2026-05-29T01:17:16Z",
    "shipName": "SALISH SEA FREEDOM"
  }
}
```

**To find the MMSI from a name** (only if not already known):
```
GET https://api.vesselapi.com/vessel/search?filter.name=<name>
```
Returns candidates ranked by AIS hit-rate. Pick the one whose
`shipName` matches and cache the MMSI permanently — name→MMSI is
stable per vessel.

**Caveats:**
- Same `mmsi` may return `404` when the vessel is out of receiver
  range or AIS is off. Treat that as "no current position".
- Free-tier rate limits — back off + fall through to a secondary key.

---

### Option B — AISStream.io (WebSocket, push)

Better for live tracking of many vessels at once. Free tier.
**WebSocket endpoint:** `wss://stream.aisstream.io/v0/stream`

**Subscribe message (send first):**
```json
{
  "APIKey": "<your-aisstream-key>",
  "BoundingBoxes": [[[48.6, -124.4], [49.9, -122.7]]],
  "FilterMessageTypes": ["PositionReport", "ShipStaticData"]
}
```

The bounding box above covers Strait of Georgia + Howe Sound + Burrard Inlet.

**Messages of interest:**
- `PositionReport` — has `MetaData.MMSI`, `Message.PositionReport.Latitude/Longitude`,
  `Sog` (speed over ground, knots), `Cog` (course over ground, degrees),
  `TrueHeading`, `MetaData.time_utc`.
- `ShipStaticData` — has `MetaData.MMSI`, `Message.ShipStaticData.Name`.
  Broadcast roughly every 6 minutes, so listen ≥ 20 s to catch some
  names on first contact. Once you've seen a name for an MMSI, cache
  it forever (vessels keep the same name + MMSI).

**Strategy:** open a connection, listen for ~20–30 s, accumulate the
latest record per MMSI, close. Cross-reference MMSIs to your fleet's
known MMSI list **first** (don't depend on `ShipName` matching — names
can be mis-cased, abbreviated, or absent).

---

## Curated fleet

Match by **MMSI** when available, otherwise by case-insensitive
normalized name (`re.sub(r'[^A-Z0-9]', '', name.upper())`).

| Name | Operator | MMSI |
|---|---|---|
| Aurora I | Wild Whales Vancouver | — |
| Aurora II | Wild Whales Vancouver | — |
| Eagle Eyes | Wild Whales Vancouver | — |
| Jing Yu | Wild Whales Vancouver | — |
| Explorathor II | Vancouver Whale Watch | — |
| Express | Vancouver Whale Watch | — |
| Strider | Vancouver Whale Watch | — |
| Lightship | Vancouver Whale Watch | — |
| Salish Sea Dream | Prince of Whales | — |
| Salish Sea Freedom | Prince of Whales | **316042213** |
| Salish Sea Eclipse | Prince of Whales | — |
| Ocean Magic | Prince of Whales | — |
| Ocean Magic II | Prince of Whales | — |
| **Countess** | **Other (Vancouver)** | **316004455** |

Smaller zodiacs (Aurora I/II, Strider, Lightship) often don't transmit
AIS — they'll be absent from your results, which is expected. Filter
out boats not seen in the last 24 h from the active list.

---

## Recommended pipeline (works for both sources)

1. **Resolve MMSI** per fleet entry, in priority order:
   a. Hard-coded MMSI on the record (above table)
   b. Persistent cache from previous runs (write to disk: `mmsi_cache.json`)
   c. `vessel/search?filter.name=...` (REST) or wait for `ShipStaticData` (WS)

2. **Fetch positions**:
   - REST: parallelize per-vessel GETs (10-thread pool is plenty).
   - WS: single connection, accumulate-then-flush after `listen_seconds`.

3. **Normalize each record** to the schema at top of this doc:
   - `heading_deg` → `heading_compass`: map degrees to one of
     N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW, NNW
     using 22.5° buckets.
   - `last_update_utc`: always UTC ISO 8601 with trailing `Z`.
   - `age_minutes`: `(now_utc − last_update_utc).total_seconds() / 60`.

4. **Stale filter**: keep all but flag `stale=true` when
   `age_minutes > 30` so the UI can show them dim/gray.

---

## Compass helper (drop-in)

```python
_COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
            "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]

def deg_to_compass(deg: float | None) -> str | None:
    if deg is None:
        return None
    return _COMPASS[round((deg % 360) / 22.5) % 16]
```

---

## Test cases

- **Countess (MMSI 316004455)** — Vancouver-area; should resolve via REST
  `/vessel/316004455/position`. If `vesselPosition` is null/404, the
  boat is dockside or AIS off.
- **Salish Sea Freedom (316042213)** — usually has live AIS in season
  (Apr–Oct), often near Steveston or Plumper Sound.
- **Strider** — typically NO AIS (small zodiac). Expect absence.

---

## Don't

- Don't poll faster than once per **60 s** per vessel via VesselAPI
  (rate limits) — cache results.
- Don't trust `ShipName` for matching when an MMSI is known.
- Don't infer position when there's no fresh record; surface `stale`
  honestly.
- Don't expose API keys client-side; keep them server-side and proxy.

---

## Existing implementations in this repo (for reference)

- `fetch_whales.py` — AISStream WebSocket implementation
  (`_collect_ais`, `WHALE_FLEET`). 20-second listen window, persistent
  MMSI→name learning across sessions.
- `fetch_whales2.py` — VesselAPI REST implementation
  (`fetch_fleet_positions`, `_resolve_mmsi`, `_fetch_vessel_position`).
  24-hour MMSI cache + manual override editor.

Read those files if you want the exact prompts, retry/backoff logic,
and cache semantics already battle-tested for this fleet.
