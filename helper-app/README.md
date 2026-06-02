# Sailing-BC Helper

A small headless companion to the main Streamlit dashboard. Runs reliably
(plain Flask + APScheduler in one process), captures buoy readings every
hour even when nobody is visiting the UI, and writes to the **same
Cloudflare KV namespace** so the main app sees the values too. Local
**SQLite** (`/data/readings.sqlite`) is the dashboard's read source so
the UI is instant regardless of CF state.

## What it does

- **Hourly cron** (`APScheduler` in-process) fetches wind + waves for:
  - Pam Rocks      (`WAS`)     — Howe Sound proxy, wind only
  - Halibut Bank   (`46146`)   — Strait of Georgia, wind + waves
  - English Bay    (`46304`)   — Vancouver, wind + waves
  - Point Atkinson (`WSB`)     — wind only
  - Jericho Wind   (`JERICHO`) — JSCA Davis weather station CSV
- Parsing logic mirrors `st.py::refreshBuoy` and `parseJerichoWindHistory`
  exactly, so values are interchangeable with the main app's data.
- **Dual writes**: SQLite (always, fast local source-of-truth) + Cloudflare
  KV (best-effort; if KV fails, the row is marked `kv_synced=0` and
  Reconcile can push it later).
- Flask web UI on **port 5111**:
  - **Graph** (default) — 3-day wind/wave series, normalized 0–35 kt
  - **Log** — last 12 readings per location
  - **Trends** — 14-day morning / afternoon / evening averages, calmest
    window highlighted
  - **Marine Forecast** — weather.gc.ca scrape, parsed by OpenAI
    (gpt-5-mini) into a period table + a near-term (3–6 h) Go/No-Go
    verdict
  - **Wave Model** — empirical `wave_m ≈ a·wind² + b` fit on stored
    history, with per-direction breakdown
  - **Alexa** — endpoint preview + skill setup instructions
  - **API** — public `/api/v1/*` documentation with copyable LLM prompt
  - **Reconcile** — diff SQLite ↔ CF KV, sync either direction
  - **Settings** — OpenAI key, CF credential test, KV rate-limit panel
    with projections, OpenAI usage log, 3-month cleanup

## Run with Docker

```bash
cp .env.example .env
# edit .env with your CF credentials (case-insensitive, quotes optional)
docker compose up -d --build
```

Browse to `http://<host>:5111`.

To rebuild after a code change cleanly:

```bash
docker compose down --rmi local && docker compose up -d --build
```

## Run locally without Docker

```bash
python -m venv .venv
. .venv/bin/activate          # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

export CLOUDFLARE_ACCOUNT_ID=...
export CLOUDFLARE_NAMESPACE_ID=...
export CLOUDFLARE_API_TOKEN=...
export HELPER_DATA_DIR=./data

python -m backend.app
```

## Configuration

All env vars are looked up **case-insensitively** (so `cloudflare_api_token`
or `CLOUDFLARE_API_TOKEN` both work) and quoted values from `.env` get
stripped automatically.

| Env var | Default | Purpose |
|---|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | — | Same as main app `secrets.toml` |
| `CLOUDFLARE_NAMESPACE_ID` | — | Same KV namespace as main app |
| `CLOUDFLARE_API_TOKEN` | — | Token with KV read+write |
| `FETCH_INTERVAL_MIN` | `60` | Minutes between fetch cycles |
| `OPENAI_API_KEY` | (opt) | Aliases: `OPENAI_KEY`, `OpenAI_key`. Overrides UI-stored key |
| `PORT` | `5111` | Web UI port |
| `HELPER_DATA_DIR` | `/data` | Where `settings.json`, `readings.sqlite`, etc. live |

## Public API — `/api/v1/*`

Stable read-only JSON surface for third-party clients (iOS chart plotter,
LLM agents, custom dashboards). **CORS open, no auth.** Backed by local
SQLite — sub-millisecond, zero CF KV reads.

### Endpoints

| Method | Path | Response |
|---|---|---|
| GET | `/api/v1/locations` | Registry `[{id, name, waves}]` |
| GET | `/api/v1/current` | All stations, keyed by id |
| GET | `/api/v1/current/{id}` | One station (404 unknown / 503 no data) |

### Response shape

```json
{
  "id": "WAS",
  "name": "Pam Rocks",
  "wind_kts": 11,
  "wind_direction": "W",
  "wave_height_m": null,
  "wave_height_cm": null,
  "waves_supported": false,
  "updated_at_utc": "2026-05-29T01:17:16Z",
  "updated_at_local": "2026-05-28T18:17:16-07:00",
  "age_minutes": 12,
  "stale": false
}
```

`stale = true` when `age_minutes > 90` (cron miss or upstream outage).

### Quick examples

```bash
# All locations
curl http://<host>:5111/api/v1/current

# Just Pam Rocks
curl http://<host>:5111/api/v1/current/WAS

# Registry
curl http://<host>:5111/api/v1/locations
```

### iOS / Swift

```swift
struct CurrentReading: Decodable {
  let id: String
  let name: String
  let wind_kts: Int?
  let wind_direction: String?
  let wave_height_m: Double?
  let wave_height_cm: Int?
  let updated_at_utc: String   // ISO 8601 Z
  let age_minutes: Int
  let stale: Bool
}

let url = URL(string: "http://<host>:5111/api/v1/current/WAS")!
URLSession.shared.dataTask(with: url) { data, _, _ in
  guard let data = data,
        let reading = try? JSONDecoder().decode(CurrentReading.self, from: data)
  else { return }
  print(reading.wind_kts ?? -1, reading.wind_direction ?? "?", reading.updated_at_utc)
}.resume()
```

### Paste into Claude / Cursor / Copilot

Use this prompt to brief a coding assistant on the API cold (also available
in the UI's **API** tab with a copy button):

```
You can fetch current marine weather conditions for buoys around Vancouver, BC.

BASE URL: http://<host>:5111

ENDPOINTS (all GET, no auth, CORS-enabled, return JSON):
  /api/v1/locations            → list of station registries [{id, name, waves}]
  /api/v1/current              → current readings for all stations
  /api/v1/current/{id}         → current reading for one station

CURRENT-READING SHAPE (per station):
  id              string  station id (e.g. "WAS")
  name            string  human name (e.g. "Pam Rocks")
  wind_kts        int     sustained wind speed, knots
  wind_direction  string  compass abbrev (N, NE, E, …, NW) or null
  wave_height_m   float   significant wave height in meters, or null
  wave_height_cm  int     same in centimeters, or null
  waves_supported bool    whether the station reports waves at all
  updated_at_utc  string  ISO 8601 UTC, e.g. "2026-05-29T01:17:16Z"
  updated_at_local string ISO 8601 with Vancouver offset
  age_minutes     int     minutes since the reading was captured
  stale           bool    true if age_minutes > 90

STATIONS:
  WAS      Pam Rocks       (Howe Sound, wind only)
  46146    Halibut Bank    (Strait of Georgia, wind + waves)
  46304    English Bay     (Vancouver, wind + waves)
  WSB      Point Atkinson  (wind only)
  JERICHO  Jericho Wind    (JSCA Davis station, wind only)
```

## Other endpoints (internal — used by the UI)

```
GET  /api/locations                     buoy registry
GET  /api/log?location=&limit=12        recent readings
GET  /api/series?location=&days=3       time-series for charting
GET  /api/trends?location=all&days=14   morning/afternoon/evening summary
GET  /api/forecast?region=howe_sound    parsed marine forecast
GET  /api/forecast/gonogo?region=...    forecast-driven verdict
GET  /api/wave_model                    empirical wave-height fit
GET  /api/reconcile                     SQLite ↔ KV diff
POST /api/reconcile/sync_kv_to_db       backfill SQLite from KV
POST /api/reconcile/sync_db_to_kv       push local rows to KV
POST /api/reconcile/sync_all            both directions, all buoys
POST /api/cleanup/old        body {months: 3}  delete old data
GET  /api/usage                         CF KV usage with projections
GET  /api/openai_log                    OpenAI call ledger
POST /api/openai_log/reset              clear the ledger
GET  /api/settings                      settings (with key masked)
POST /api/settings                      update settings
GET  /api/db/stats                      SQLite row counts per buoy
GET  /api/alexa                         {speech} preview
POST /api/alexa                         full Alexa response envelope
POST /api/fetch_now                     trigger one fetch cycle
GET  /api/health                        liveness + last-cycle status
GET  /api/version                       app version
```

## Alexa integration

Endpoint `POST /api/alexa` speaks the briefing and renders an APL visual
on Echo Show devices. Requires:

- HTTPS endpoint with a trusted cert (Let's Encrypt via Coolify works on
  sslip.io domains)
- Custom skill registered at `developer.amazon.com/alexa/console/ask`
  with invocation name `sailing conditions`
- APL interface enabled (Build → Interfaces)
- **Build the model for the locale your Echo is set to** (e.g. en-CA if
  your device is Canadian; can build multiple locales)

Full step-by-step lives in the **Alexa** tab of the UI.

Cost-control: marine forecast is fetched **on demand only** (not on a
schedule), with an hourly cache that rolls over at HH:01 (forecasts
re-issue hourly). On an Alexa cache miss the skill says *"Refreshing
forecast. Ask again in a few seconds."* and kicks off a background
OpenAI fetch so the next invocation is warm.

## Notes

- **Single gunicorn worker** with `--threads 8` — APScheduler is
  in-process so multi-worker would duplicate fetches; threads let
  concurrent requests not block on one slow endpoint.
- A fetch cycle runs immediately on boot so you don't wait an hour
  for the first reading.
- 30-minute slot bucketing matches the main app exactly — a fetch at
  e.g. `14:07` overwrites the same key the Streamlit page would have
  written at `14:00`.
- CF KV reads are parallelized (50 threads, shared `requests.Session`
  with `pool_maxsize=50`) and locally cached 60 s on top of SQLite.
