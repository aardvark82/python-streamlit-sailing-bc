# Sailing-BC Helper

A small headless companion to the main Streamlit dashboard. Runs reliably
(plain Flask + APScheduler in one process), captures buoy readings every
hour even when nobody is visiting the UI, and writes to the **same
Cloudflare KV namespace** so the main app sees the values too.

## What it does

- **Hourly cron** (`APScheduler` in-process) fetches wind + waves for:
  - Halibut Bank (`46146`) — with waves
  - English Bay   (`46304`) — with waves
  - Point Atkinson (`WSB`)  — wind only
  - Pam Rocks      (`WAS`)  — wind only
- Parsing logic mirrors `st.py::refreshBuoy` exactly (same HTML scrape,
  same wind/wave extraction regex), so values are interchangeable.
- Writes to Cloudflare KV with keys `{buoy}_wind_{iso}`,
  `{buoy}_direction_{iso}`, `{buoy}_wave_{iso}` — identical schema to
  the main app's `record_buoy_data_history`.
- Flask web UI on **port 5111**:
  - **Log** — last 12 readings per location
  - **Graph** — 3-day wind/wave series (Plotly)
  - **Trends** — 14-day morning / afternoon / evening averages per
    location, with the calmest window highlighted
  - **Settings** — set the OpenAI API key (stored in `/data/settings.json`)

## Run with Docker

```bash
cp .env.example .env
# edit .env with your CF credentials
docker compose up -d --build
```

Browse to `http://<host>:5111`.

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

| Env var | Default | Purpose |
|---|---|---|
| `CLOUDFLARE_ACCOUNT_ID` | — | Same as main app `secrets.toml` |
| `CLOUDFLARE_NAMESPACE_ID` | — | Same KV namespace as main app |
| `CLOUDFLARE_API_TOKEN` | — | Token with KV read+write |
| `FETCH_INTERVAL_MIN` | `60` | Minutes between fetch cycles |
| `OPENAI_API_KEY` | (opt) | Overrides UI-stored key if set |
| `PORT` | `5111` | Web UI port |
| `HELPER_DATA_DIR` | `/data` | Where `settings.json` lives |

## Endpoints

- `GET /api/locations`
- `GET /api/log?location=46146&limit=12`
- `GET /api/series?location=46146&days=3`
- `GET /api/trends?location=all&days=14`
- `GET /api/settings`
- `POST /api/settings`  body `{"openai_api_key": "sk-..."}`
- `POST /api/fetch_now`
- `GET /api/health`

## Notes

- Only **one gunicorn worker** is used on purpose — the scheduler is
  in-process, multiple workers would duplicate fetches.
- A fetch cycle also runs immediately on boot so you don't wait an hour
  for the first reading.
- The 30-minute slot bucketing matches the main app exactly, so a fetch
  at e.g. `14:07` overwrites the same key the Streamlit page would have
  written at `14:00`.
