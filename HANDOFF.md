# Sailing-BC dashboard — session handoff

A Streamlit dashboard for sailing/boating decisions in Vancouver/Howe Sound BC.
Repo: https://github.com/aardvark82/python-streamlit-sailing-bc
Working dir: `C:\Users\acker\Dev\20250430 python-streamlit-sailing-bc`
Run: `python -m streamlit run st.py` → http://localhost:8501
Deploy: Streamlit Cloud, auto-deploys on push to `main`. Python 3.13.
VERSION file is auto-bumped on each meaningful commit (currently 126).

## File map (active modules)

| File | Purpose |
|---|---|
| `st.py` | Entry point. Defines all `st.Page` objects, the sidebar nav, buoy page logic (`refreshBuoy`, `parseJerichoWindHistory`, `plot_merged_wind_chart`, `_fetch_buoy_wind_history_df`, `record_buoy_data_history`, `drawMapWithBuoy`). |
| `fetch_gonogo.py` | Go/No-Go logic (decision matrix, current factors, kiosk mode page, weekly outlook heatmap, snapshot metrics). |
| `fetch_forecast.py` | Marine forecast HTML scrape + GPT-4o CSV parse (`openAIFetchForecastForURL`). The only active OpenAI call. Time-bucketed cache key. |
| `fetch_tides.py` | Pt Atkinson tides. **Primary path: DFO IWLS REST API** (`fetch_iwls_tide_extremes_pt_atkinson`). Selenium-CSV path remains as fallback. `find_local_extrema` uses `scipy.signal.find_peaks` with 5cm prominence. |
| `fetch_weather.py` | OpenWeatherMap fetch + display (used by Dashboard, Squamish, Lions Bay). |
| `fetch_beach.py` | Sandy Cove water quality (VCH PDF) + Beach Go/No-Go heatmap. |
| `fetch_alex.py` | Live boat tracker (flespi.io) for Zodiac Pro 420 / Teltonika FMM13A. Also station overlay (Pam Rocks/Halibut/Pt Atkinson/Jericho/Howe Sound). Hosts the `display_iot_usage_page` for IoT Usage sidebar. |
| `fetch_whales.py` | Whales 1 — AIS via aisstream.io WebSocket. |
| `fetch_whales2.py` | Whales 2 — VesselAPI Ship Tracking, OAuth fallback w/ vesselapi2_key. |
| `wind_utils.py` | SVG wind arrow + `_color_for_speed` (3 buckets: <10 green, <20 orange, ≥20 red) + `wind_arrow_glyph` (1/2/3 unicode arrows pointing downwind). |
| `utils.py` | `cached_fetch_url` (30min), `cached_fetch_url_live` (3min), `displayStreamlitDateTime`, **`display_last_updated_badge`** (the green/orange/red staleness pill used everywhere). |

## Sidebar layout (`pages` dict in st.py)

- **Conditions**: Go / No-Go (default), Dashboard
- **Live Data**: Alex Location, Pam Rocks, Whale boats, Whale boats 2, Jericho Wind, English Bay, Pt Atkinson, Halibut Bank
- **Forecast & Tides**: Marine Forecast, Tides, Beach
- **Regional Weather**: Squamish W, Lions Bay W
- **Display**: Kiosk Mode, IoT Usage

## Storage

- **Wind history**: Cloudflare KV (free tier 100k reads/day). Keys: `{buoy_id}_wind_{iso}`, `{buoy_id}_direction_{iso}`, `{buoy_id}_wave_{iso}`. Written by `record_buoy_data_history`, read by `_fetch_buoy_wind_history_df`.
- Stormglass / kvdb.io are fully removed.

## Secrets needed (`.streamlit/secrets.toml`)

`OpenAI_key`, `openweather_api_key`, `cloudflare_account_id` / `cloudflare_namespace_id` / `cloudflare_api_token`, `aisstream-io_key`, `vesselapi_key` + `vesselapi2_key`, `flespi_api_key`, `1nce_client_id` + `1nce_client_secret`. (Stormglass + kvdb removed.)

## Conventions / idioms

- 24-hour time format **everywhere** in display strings. `strptime` patterns parsing 12-hour external sources (weather.gc.ca's "Issued HH:MM AM/PM", VCH PDF) stay as `%I:%M %p`.
- Every "freshness" indicator uses `display_last_updated_badge(draw, dt, label=…, source_url=…, extra_text=…, extra_html=…)`. Color buckets: <15min green, <1h orange, <6h red, else gray.
- All map markers use `wind_utils._color_for_speed` for speed coloring (3-bucket green/orange/red).
- Wind direction arrows on maps: unicode glyph from `wind_arrow_glyph`, point downwind, count = intensity (1/2/3 for <10 / 10-20 / 20+ kts).
- `width='stretch'` on `st.dataframe` raises `'str' object cannot be interpreted as an integer` on this Streamlit version — omit width entirely for tables. (`st.plotly_chart(width='stretch')` is fine.)
- VESSELAPI: call helpers route through `_http_get` which auto-falls-back from primary→backup token on HTTP 429.
- 1NCE auth: OAuth client credentials only (`_get_1nce_access_token`). Session-state cache key includes md5(client_id|client_secret) so credential changes auto-invalidate.
- Streamlit cache invalidation pattern: include a `_cache_buster=N` parameter on `@st.cache_data` functions; bumping N drops stale entries on deploy.

## Recently fixed (last 20 commits)

- v126 — Tides Cloud bug: hardcoded IWLS station id was wrong (`5cebf1df…` → real id `5cebf1de3d0f4a073c4bb94c`). Verified against `/stations?code=07795`.
- v125 — Stormglass + kvdb deleted entirely.
- v124 — Jericho CSV: ragged-row tolerance (`engine='python'`, `on_bad_lines='skip'`, explicit `names=`).
- v123 — Tides primary path = DFO IWLS API (replaces Selenium-on-Cloud which was flaky).
- v122 — 1NCE: dropped `1nce_bearer_token` fallback; OAuth-only.
- v120 — Tides: scipy.find_peaks for extrema (catches mixed semi-diurnal pair at Pt Atkinson).
- v119 — Tides chart limited to next 48h window.
- v109 — IoT Usage: 2 KB metric cards at top (Flespi Data, 1NCE Data).
- v108 — IoT Usage moved off Alex page into its own sidebar entry under "Display".
- v92 — Marine station map markers replaced dots with directional unicode arrows (1/2/3 by intensity).

## Known fragile / watch points

- `processCSVResponseToJSONSelenium` does NOT like the full ~10k-row tides CSV. It's fine with `csv_lines[::20]` (every 20 min). Keep that subsample if Selenium fallback is ever exercised.
- `pd.read_csv` on Davis weather format (Jericho) needs explicit `names=` + python engine + skip-bad-lines. C engine aborts on the first ragged row.
- `Scattermapbox.textfont.color` is **scalar per trace**, not array — to color stations differently emit one trace per station.
- Plotly `add_vline` with a tz-aware datetime can raise `int + datetime` on some Plotly builds. Use `add_shape(type='line')` + `add_annotation` with `pd.Timestamp(dt)` instead.
- 1NCE OAuth body must be JSON with `Content-Type: application/json` (form-encoded gives 400).
- 1NCE response uses `vesselPosition` wrapper key (camelCase), not `vessel`. Reader unwraps both.
- VesselAPI position endpoint: `/vessel/{mmsi}/position?filter.idType=mmsi`. Wrapper key: `vesselPosition`.

## OpenAI usage summary (gpt-4o only)

Active surface: `openAIFetchForecastForURL` — full-HTML in, CSV table out. Time-bucketed 30-min cache. ~4 300 tokens/call ≈ $0.013/call. Cold full-page-traversal session = ~3 calls (~$0.04). Daily ceiling ~12 calls (~$0.16).

`openAIFetchTidesForURL` exists but is gated by `USE_CHAT_GPT=False` — never executes.

## How to resume in a new session

Paste this prompt verbatim:

> Read `HANDOFF.md` in the repo root for context. Working dir is `C:\Users\acker\Dev\20250430 python-streamlit-sailing-bc`. Auto-bump VERSION on important commits. Commit directly to main. Reply briefly when iterating.

Then state your next task. The new session will have full context with minimal tokens consumed.
