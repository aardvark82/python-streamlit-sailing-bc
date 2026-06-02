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

## Curated fleet (17 vessels)

Match by **MMSI** when available (faster + immune to name typos),
otherwise by case-insensitive normalized name
(`re.sub(r'[^A-Z0-9]', '', name.upper())`).

### Wild Whales Vancouver — blue `#1f77b4`

| Name | MMSI | Region |
|---|---|---|
| Aurora I | **316040487** | Vancouver |
| Aurora II | **316040366** | Vancouver |
| Eagle Eyes | **316034894** | Vancouver |
| Jing Yu | **316032442** | Vancouver |

### Vancouver Whale Watch — green `#2ca02c`

| Name | MMSI | Region |
|---|---|---|
| Explorathor II | **316008046** | Vancouver |
| Explorathor Express | **316008045** | Vancouver |
| Express | — | Vancouver |
| Strider | **316035167** | Vancouver |
| Lightship | **316014609** | Vancouver |

### Prince of Whales — orange `#ff7f0e`

| Name | MMSI | Region |
|---|---|---|
| Salish Sea Dream | **316032858** | Vancouver |
| Salish Sea Freedom | **316042213** | Vancouver |
| Salish Sea Eclipse | **316039686** | Victoria |
| Salish Sea Glory | **316059231** | Vancouver |
| Ocean Magic | **316006789** | Telegraph Cove |
| Ocean Magic II | **316008331** | Telegraph Cove |

### Other / private — purple `#9467bd`

Sister vessels (consecutive MMSIs 316004454–456).

| Name | MMSI | Region |
|---|---|---|
| The Duchess | **316004454** | Vancouver |
| Countess | **316004455** | Vancouver |
| Lady Di | **316004456** | Vancouver |

### Hard-coded MMSIs at a glance (16 of 17 known)

```
316004454  The Duchess
316004455  Countess
316004456  Lady Di
316006789  Ocean Magic
316008045  Explorathor Express
316008046  Explorathor II
316008331  Ocean Magic II
316014609  Lightship
316032442  Jing Yu
316032858  Salish Sea Dream
316034894  Eagle Eyes
316035167  Strider
316039686  Salish Sea Eclipse
316040366  Aurora II
316040487  Aurora I
316042213  Salish Sea Freedom
316059231  Salish Sea Glory
```

Only **Express** (Vancouver Whale Watch) is still without an MMSI —
add it via `/vessel/search?filter.name=Express` once observed.

Filter boats not seen in the last 24 h from the "active" list — even
with valid MMSIs, vessels in port often have AIS off and will appear
to be "missing" until they leave the dock.

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

- **Countess (316004455) / Lady Di (316004456) / The Duchess (316004454)**
  — three sister vessels with consecutive MMSIs; if you can resolve one
  you can usually resolve all three. Often moored at the same dock.
- **Salish Sea Freedom (316042213)** — usually has live AIS in season
  (Apr–Oct), often near Steveston or Plumper Sound.
- **Salish Sea Glory (316059231)** — newest Prince of Whales hull;
  Vancouver-based.
- **Explorathor Express (316008045)** — Vancouver Whale Watch flagship
  with reliable AIS in season.
- **Strider / Lightship / Aurora I-II** — typically NO AIS (small
  zodiacs). Expect absence; that's not a bug.

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
