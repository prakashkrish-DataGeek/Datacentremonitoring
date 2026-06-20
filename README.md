# Asset Exposure Watch 🛡️

**An Agentic Event-to-Action Engine for Operations & Risk Duty Officers.**

> *"An event just happened in the world — does it touch anything I own, and what should I do in the next hour?"*

This is not a feed reader. It is a **decision engine**: every run ends in one of two states — **stand down** or **actionable exposure with a draft action**. The dashboard exists to explain that decision after the fact, not to replace it.

---

## What It Does

Three agents, one shared state, conditional routing via LangGraph:

| Agent | Role | Decision |
|-------|------|----------|
| **A — Triage** | Polls live feeds every 60s. Applies transparent thresholds. | *Does this event matter at all?* |
| **B — Exposure** | Computes haversine distance + proximity decay + criticality weight. | *Does it touch anything I own?* |
| **C — Action** | Drafts Jira/PagerDuty payloads and executive briefings. | *What do I do about it?* |

Most events route to **stand down** (a correct non-alert is a successful run). Only anomalies proceed to analysis.

---

## Live Data Sources (Verified)

| Source | URL | Key Required | Status |
|--------|-----|--------------|--------|
| **USGS Earthquakes** | `https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson` | ❌ No | ✅ Live, GeoJSON, precise coordinates |
| **Open-Meteo Air Quality** | `https://air-quality-api.open-meteo.com/v1/air-quality` | ❌ No | ✅ Live, coordinate-based AQI |
| **Alpha Vantage News** | `https://www.alphavantage.co/query` | ✅ Free key | ⚠️ Optional, rate-limited |

The app runs cleanly on the two keyless feeds. A missing Alpha Vantage key does **not** break the system.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit Dashboard

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501`. Click **"Replay Test"** to run a deterministic M5.2 earthquake replay, or **"Live Poll"** to fetch real USGS/Open-Meteo data.

### 3. Run Replay Tests

```bash
pytest test_replay.py -v
```

These assertions prove the system works:
- ✅ M5.2 near Ridgecrest flags Bakersfield & Fresno (close assets)
- ✅ Fargo (800+ km away) is correctly ignored
- ✅ Same replay event produces identical findings every time
- ✅ Raising threshold above M5.2 produces stand-down
- ✅ Impossibly high exposure floor produces stand-down-with-reason

---

## Project Structure

```
asset-exposure-watch/
├── app.py                 # Streamlit dashboard (deployable)
├── graph.py               # LangGraph topology & three agents
├── ingestion.py           # Live + replay data sources, throttling
├── schemas.py             # All Pydantic models (validated from live APIs)
├── exposure.py            # Transparent geospatial/criticality scoring
├── test_replay.py         # Deterministic assertions proving correctness
├── sample_assets.csv      # Replaceable sample register (25 assets)
├── replay_data/           # Bundled historical events for replay mode
│   └── ridgecrest_m52.json
├── requirements.txt       # Dependencies
└── README.md              # This file
```

---

## How to Plug In Your Real World

### 1. Replace the Asset Register

Upload your own CSV via the Streamlit sidebar. Required columns:

```csv
asset_id,name,latitude,longitude,type,criticality,sector
```

- `type`: `facility`, `warehouse`, `supply-node`, `market-position`
- `criticality`: `1` (low) to `5` (critical)
- `sector`: optional, used for financial sentiment filtering

### 2. Connect a Real Ticketing Endpoint

The app drafts but **never dispatches** action payloads. To connect:

1. In `graph.py`, modify `agent_action` to add a webhook dispatch step
2. Or export the draft JSON from the dashboard and POST it manually:
   ```bash
   curl -X POST https://your-jira-instance/rest/api/2/issue      -H "Content-Type: application/json"      -d @draft_jira_payload.json
   ```

### 3. Add Alpha Vantage Key (Optional)

Set in `app.py` sidebar or environment:
```bash
export ALPHA_VANTAGE_API_KEY="your_key"
```

---

## Honest Limits

We state these plainly so you know what to refine for production:

1. **Distance is straight-line (haversine), not travel time.** A real deployment would refine this per asset class — road network, rail topology, hydrology, or flight time.
2. **Financial sentiment is coarse.** Alpha Vantage news sentiment is a blunt signal. A real deployment would use sector-specific NLP models or supply-chain graph analysis.
3. **The app drafts actions but never dispatches them.** The duty officer reviews and clicks send. This is by design — no autonomous webhook firing.
4. **The sample register is illustrative.** It is spread across active seismic regions to produce demo hits. Replace it with your actual asset footprint.
5. **Air quality is point-sampled at asset coordinates.** It does not model plume dispersion or wind direction.
6. **Token spend tracking is a placeholder.** The current budget ceiling is a framework; real LLM cost tracking would need provider-specific token counting.

**The architecture is the durable part:** swap the sample register for a real one, refine the distance model per asset class, and connect the action payload to a real ticketing endpoint — the pipeline stands.

---

## Configuration

All thresholds live in one place (the sidebar or `AppConfig` in `schemas.py`):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `poll_interval_seconds` | 60 | Seconds between feed polls |
| `earthquake_min_magnitude` | 4.5 | Minimum magnitude to flag |
| `air_quality_min_aqi` | 150 | Minimum AQI to flag |
| `exposure_max_distance_km` | 500 | Max distance to consider an asset |
| `exposure_floor_score` | 0.3 | Minimum exposure score to trigger action |
| `distance_decay_factor` | 0.05 | Exponential decay rate for proximity |
| `token_budget_ceiling` | 1000 | Max token spend per run |

---

## Deployment

### Streamlit Cloud (Recommended)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Select your repo, branch, and `app.py`
4. Add a `requirements.txt` (included)

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

---

## License

MIT — built for operations teams who need decisions, not dashboards.
