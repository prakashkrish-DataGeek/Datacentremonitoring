"""
app.py — Executive Operations Dashboard for Asset Exposure Watch.
A modern control tower interface for industrial risk monitoring.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

from schemas import (
    AppConfig, Asset, RawEvent, FlaggedEvent, ExposureFinding, 
    ActionPayload, SharedState, RunOutcome, EventType
)
from graph import run_once, run_replay, build_graph
from ingestion import load_replay_event, ThrottledPoller
from exposure import haversine_distance


# ── Page Config ─────────────────────────────────────────────

st.set_page_config(
    page_title="Asset Exposure Watch | Operations Control",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS for Control Tower Aesthetic ──────────────────

st.markdown("""
<style>
    /* Root overrides */
    .stApp {
        background: #0a0e1a;
    }

    /* Header strip */
    .header-strip {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 16px 24px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }

    .header-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #e0e6ed;
        letter-spacing: 0.5px;
    }

    .header-sub {
        font-size: 0.75rem;
        color: #6b8cae;
        margin-top: 2px;
    }

    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .badge-idle { background: #1a2f1a; color: #4ade80; border: 1px solid #22c55e; }
    .badge-polling { background: #1a2a3a; color: #60a5fa; border: 1px solid #3b82f6; animation: pulse 2s infinite; }
    .badge-replaying { background: #2a1a2a; color: #c084fc; border: 1px solid #a855f7; }
    .badge-alert { background: #2a1a1a; color: #f87171; border: 1px solid #ef4444; }
    .badge-sample { background: #2a2510; color: #fbbf24; border: 1px solid #f59e0b; }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.6; }
    }

    .metric-pill {
        text-align: center;
        padding: 8px 16px;
    }

    .metric-pill .value {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e0e6ed;
    }

    .metric-pill .label {
        font-size: 0.65rem;
        color: #6b8cae;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(145deg, #111827 0%, #1a2332 100%);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 18px 20px;
        height: 100%;
    }

    .kpi-card .kpi-value {
        font-size: 2rem;
        font-weight: 800;
        color: #e0e6ed;
        line-height: 1.1;
    }

    .kpi-card .kpi-label {
        font-size: 0.7rem;
        color: #6b8cae;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 6px;
    }

    .kpi-card .kpi-delta {
        font-size: 0.75rem;
        margin-top: 8px;
        font-weight: 500;
    }

    .delta-up { color: #f87171; }
    .delta-down { color: #4ade80; }
    .delta-neutral { color: #6b8cae; }

    /* Filter Panel */
    .filter-panel {
        background: #0f172a;
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 16px;
    }

    .filter-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 14px 16px;
        margin-bottom: 10px;
    }

    .filter-card .filter-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
    }

    .filter-card .filter-title {
        font-size: 0.8rem;
        font-weight: 600;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .filter-card .filter-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #60a5fa;
        font-family: 'SF Mono', monospace;
    }

    .filter-card .filter-help {
        font-size: 0.7rem;
        color: #475569;
        margin-top: 4px;
    }

    /* Action Buttons */
    .btn-primary {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 600 !important;
        letter-spacing: 0.3px;
    }

    .btn-secondary {
        background: #1e293b !important;
        color: #94a3b8 !important;
        border: 1px solid #334155 !important;
        border-radius: 8px !important;
        padding: 10px 20px !important;
        font-weight: 500 !important;
    }

    /* Section headers */
    .section-header {
        font-size: 0.85rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin: 24px 0 12px 0;
        padding-bottom: 8px;
        border-bottom: 1px solid #1e3a5f;
    }

    /* Finding cards */
    .finding-card {
        background: linear-gradient(145deg, #111827 0%, #1a2332 100%);
        border-left: 3px solid;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin-bottom: 10px;
    }

    .finding-critical { border-left-color: #ef4444; }
    .finding-high { border-left-color: #f97316; }
    .finding-medium { border-left-color: #eab308; }
    .finding-low { border-left-color: #22c55e; }

    /* Tables */
    .stDataFrame {
        border: 1px solid #1e3a5f !important;
        border-radius: 8px !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: #0f172a;
        padding: 4px;
        border-radius: 10px;
        border: 1px solid #1e3a5f;
    }

    .stTabs [data-baseweb="tab"] {
        color: #6b8cae !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.5px;
        padding: 8px 16px !important;
        border-radius: 6px !important;
    }

    .stTabs [aria-selected="true"] {
        background: #1e3a5f !important;
        color: #e0e6ed !important;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #111827 !important;
        border: 1px solid #1e293b !important;
        border-radius: 8px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 6px; }
    ::-webkit-scrollbar-track { background: #0a0e1a; }
    ::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Sidebar styling when expanded */
    [data-testid="stSidebar"] {
        background: #0d1117;
        border-right: 1px solid #1e3a5f;
    }
</style>
""", unsafe_allow_html=True)


# ── Session State ─────────────────────────────────────────

def init_session():
    defaults = {
        "assets": [],
        "config": AppConfig(),
        "last_run": None,
        "history": [],
        "poller": None,
        "running": False,
        "engine_state": "idle",  # idle | polling | replaying
        "last_event_time": None,
        "trigger_live": False,
        "trigger_replay": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


init_session()


# ── Asset Loading ─────────────────────────────────────────

def load_assets(filepath: str) -> List[Asset]:
    """Load assets from CSV."""
    df = pd.read_csv(filepath)
    assets = []
    for _, row in df.iterrows():
        try:
            asset = Asset(
                asset_id=str(row["asset_id"]),
                name=str(row["name"]),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
                type=row["type"],
                criticality=int(row["criticality"]),
                sector=str(row["sector"]) if pd.notna(row.get("sector")) else None,
            )
            assets.append(asset)
        except Exception as e:
            st.toast(f"Skipping invalid asset: {row.get('asset_id', '?')}", icon="⚠️")
    return assets


# ── Load Assets ───────────────────────────────────────────

if not st.session_state.assets:
    sample_path = Path(__file__).parent / "sample_assets.csv"
    if sample_path.exists():
        st.session_state.assets = load_assets(str(sample_path))
    else:
        st.session_state.assets = [
            Asset(asset_id="LAX-01", name="LA Distribution", latitude=34.0522, longitude=-118.2437, 
                  type="warehouse", criticality=5, sector="logistics"),
            Asset(asset_id="SFO-02", name="SF Bay Facility", latitude=37.7749, longitude=-122.4194,
                  type="facility", criticality=4, sector="technology"),
            Asset(asset_id="BAK-11", name="Bakersfield Oil Terminal", latitude=35.3733, longitude=-119.0187,
                  type="facility", criticality=4, sector="energy"),
            Asset(asset_id="FAT-10", name="Fresno Agricultural Hub", latitude=36.7378, longitude=-119.7871,
                  type="supply-node", criticality=3, sector="agriculture"),
        ]


# ── Sidebar: Filter Panel ───────────────────────────────

with st.sidebar:
    st.markdown("<div style='font-size:1.1rem;font-weight:700;color:#e0e6ed;margin-bottom:16px;'>⚙️ Configuration</div>", unsafe_allow_html=True)

    # Asset Register
    st.markdown("<div class='section-header' style='margin-top:0;'>Asset Register</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload CSV to replace sample data", type=["csv"], key="asset_upload", label_visibility="collapsed")
    if uploaded:
        with open("/tmp/uploaded_assets.csv", "wb") as f:
            f.write(uploaded.getvalue())
        st.session_state.assets = load_assets("/tmp/uploaded_assets.csv")
        st.toast(f"Loaded {len(st.session_state.assets)} assets", icon="✅")

    if not uploaded and st.session_state.assets:
        st.markdown("<span class='status-badge badge-sample'>📋 Sample Data Active</span>", unsafe_allow_html=True)
        st.caption(f"{len(st.session_state.assets)} illustrative assets loaded")

    st.markdown("<div class='section-header'>Threshold Filters</div>", unsafe_allow_html=True)

    # Filter Cards
    st.markdown("""
        <div class="filter-card">
            <div class="filter-header">
                <span class="filter-title">🌍 Earthquake Magnitude</span>
                <span class="filter-value">4.5</span>
            </div>
            <div class="filter-help">Minimum magnitude to flag an event</div>
        </div>
    """, unsafe_allow_html=True)
    eq_thresh = st.slider("", 0.0, 10.0, st.session_state.config.earthquake_min_magnitude, 0.1, key="eq_slider", label_visibility="collapsed")

    st.markdown("""
        <div class="filter-card">
            <div class="filter-header">
                <span class="filter-title">💨 Air Quality Index</span>
                <span class="filter-value">150</span>
            </div>
            <div class="filter-help">Minimum AQI to flag an event</div>
        </div>
    """, unsafe_allow_html=True)
    aq_thresh = st.slider("", 0, 500, st.session_state.config.air_quality_min_aqi, 5, key="aq_slider", label_visibility="collapsed")

    st.markdown("""
        <div class="filter-card">
            <div class="filter-header">
                <span class="filter-title">📏 Max Distance</span>
                <span class="filter-value">500 km</span>
            </div>
            <div class="filter-help">Maximum distance to consider an asset exposed</div>
        </div>
    """, unsafe_allow_html=True)
    max_dist = st.slider("", 10, 2000, int(st.session_state.config.exposure_max_distance_km), 10, key="dist_slider", label_visibility="collapsed")

    st.markdown("""
        <div class="filter-card">
            <div class="filter-header">
                <span class="filter-title">🎯 Exposure Floor</span>
                <span class="filter-value">0.30</span>
            </div>
            <div class="filter-help">Minimum exposure score to trigger action</div>
        </div>
    """, unsafe_allow_html=True)
    floor_score = st.slider("", 0.0, 1.0, st.session_state.config.exposure_floor_score, 0.05, key="floor_slider", label_visibility="collapsed")

    # Update config
    st.session_state.config = AppConfig(
        earthquake_min_magnitude=eq_thresh,
        air_quality_min_aqi=aq_thresh,
        exposure_max_distance_km=float(max_dist),
        exposure_floor_score=floor_score,
        poll_interval_seconds=st.session_state.config.poll_interval_seconds,
        distance_decay_factor=st.session_state.config.distance_decay_factor,
        token_budget_ceiling=st.session_state.config.token_budget_ceiling,
        alpha_vantage_api_key=st.session_state.config.alpha_vantage_api_key,
    )

    # Run Controls
    st.markdown("<div class='section-header'>Run Controls</div>", unsafe_allow_html=True)

    # Status indicator
    state = st.session_state.engine_state
    if state == "polling":
        st.markdown("<span class='status-badge badge-polling'>● Polling Feeds</span>", unsafe_allow_html=True)
    elif state == "replaying":
        st.markdown("<span class='status-badge badge-replaying'>● Replaying Event</span>", unsafe_allow_html=True)
    elif state == "alert":
        st.markdown("<span class='status-badge badge-alert'>● Actionable Exposure</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-badge badge-idle'>○ Engine Idle</span>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button("🔄 Live Poll", use_container_width=True, type="primary", key="btn_live"):
            st.session_state.trigger_live = True
            st.session_state.engine_state = "polling"
    with col2:
        if st.button("▶️ Replay", use_container_width=True, key="btn_replay"):
            st.session_state.trigger_replay = True
            st.session_state.engine_state = "replaying"

    st.markdown("<div style='font-size:0.65rem;color:#475569;text-align:center;margin-top:12px;'>Asset Exposure Watch v1.0 · LangGraph · Open Data</div>", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MAIN DASHBOARD — CONTROL TOWER LAYOUT
# ═══════════════════════════════════════════════════════════

# ── Hero / Status Strip ───────────────────────────────────

header_col1, header_col2, header_col3, header_col4, header_col5 = st.columns([3, 1, 1, 1.5, 1.5])

with header_col1:
    st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-size:1.8rem;">🛡️</div>
            <div>
                <div style="font-size:1.3rem;font-weight:700;color:#e0e6ed;">Asset Exposure Watch</div>
                <div style="font-size:0.75rem;color:#6b8cae;">Agentic Event-to-Action Engine</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with header_col2:
    asset_count = len(st.session_state.assets)
    st.markdown(f"""
        <div class="metric-pill">
            <div class="value">{asset_count}</div>
            <div class="label">Assets Monitored</div>
        </div>
    """, unsafe_allow_html=True)

with header_col3:
    run_count = len(st.session_state.history)
    st.markdown(f"""
        <div class="metric-pill">
            <div class="value">{run_count}</div>
            <div class="label">Runs Completed</div>
        </div>
    """, unsafe_allow_html=True)

with header_col4:
    last_time = "—"
    if st.session_state.last_run and st.session_state.last_run.finished_at:
        last_time = st.session_state.last_run.finished_at.strftime("%H:%M:%S UTC")
    elif st.session_state.last_event_time:
        last_time = st.session_state.last_event_time.strftime("%H:%M:%S UTC")
    st.markdown(f"""
        <div class="metric-pill">
            <div class="value" style="font-size:1rem;">{last_time}</div>
            <div class="label">Last Check</div>
        </div>
    """, unsafe_allow_html=True)

with header_col5:
    if st.session_state.last_run:
        outcome = st.session_state.last_run.outcome
        if outcome and outcome.value == "actionable_exposure":
            st.markdown("<span class='status-badge badge-alert'>🚨 ALERT</span>", unsafe_allow_html=True)
        elif outcome and outcome.value == "stand_down_with_reason":
            st.markdown("<span class='status-badge badge-idle'>✓ CLEAR</span>", unsafe_allow_html=True)
        else:
            st.markdown("<span class='status-badge badge-idle'>✓ STAND DOWN</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='status-badge badge-idle'>○ IDLE</span>", unsafe_allow_html=True)

st.markdown("<hr style='border-color:#1e3a5f;margin:16px 0;'>", unsafe_allow_html=True)


# ── Handle Triggers ───────────────────────────────────────

if st.session_state.get("trigger_live"):
    st.session_state.trigger_live = False
    with st.spinner("Polling live feeds..."):
        import httpx
        from ingestion import fetch_usgs_earthquakes, fetch_openmeteo_air_quality

        async def do_live():
            async with httpx.AsyncClient() as client:
                events = []
                eq = await fetch_usgs_earthquakes(client, period="hour")
                events.extend(eq)
                aq = await fetch_openmeteo_air_quality(client, st.session_state.assets)
                events.extend(aq)

                if events:
                    result = await run_once(st.session_state.config, st.session_state.assets, events)
                else:
                    result = SharedState(
                        run_id=f"run_{uuid.uuid4().hex[:12]}",
                        config=st.session_state.config,
                        started_at=datetime.now(timezone.utc),
                        outcome=RunOutcome.STAND_DOWN,
                        outcome_reason="No events from live feeds in this cycle.",
                    )
                    result.finished_at = datetime.now(timezone.utc)
                return result

        result = asyncio.run(do_live())
        st.session_state.last_run = result
        st.session_state.history.append(result)
        st.session_state.last_event_time = datetime.now(timezone.utc)
        st.session_state.engine_state = "alert" if result.outcome and result.outcome.value == "actionable_exposure" else "idle"
        st.rerun()

if st.session_state.get("trigger_replay"):
    st.session_state.trigger_replay = False
    with st.spinner("Running Ridgecrest M5.2 replay..."):
        result = asyncio.run(run_replay(
            st.session_state.config, 
            st.session_state.assets, 
            "ridgecrest_m52.json"
        ))
        st.session_state.last_run = result
        st.session_state.history.append(result)
        st.session_state.last_event_time = datetime.now(timezone.utc)
        st.session_state.engine_state = "alert" if result.outcome and result.outcome.value == "actionable_exposure" else "idle"
        st.rerun()


# ═══════════════════════════════════════════════════════════
# ROW 1: KPI SUMMARY CARDS
# ═══════════════════════════════════════════════════════════

if st.session_state.last_run:
    run = st.session_state.last_run

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

    with kpi1:
        raw_count = len(run.raw_events)
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value">{raw_count}</div>
                <div class="kpi-label">Raw Events</div>
                <div class="kpi-delta delta-neutral">Events ingested this cycle</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi2:
        flagged_count = len(run.flagged_events)
        flagged_pct = round((flagged_count / max(raw_count, 1)) * 100)
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{'#fbbf24' if flagged_count > 0 else '#4ade80'};">{flagged_count}</div>
                <div class="kpi-label">Flagged Events</div>
                <div class="kpi-delta {'delta-up' if flagged_count > 0 else 'delta-neutral'}">{flagged_pct}% of raw</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi3:
        finding_count = len(run.exposure_findings)
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{'#f87171' if finding_count > 0 else '#4ade80'};">{finding_count}</div>
                <div class="kpi-label">Exposed Assets</div>
                <div class="kpi-delta {'delta-up' if finding_count > 0 else 'delta-neutral'}">{'Assets at risk' if finding_count > 0 else 'No exposure'}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi4:
        action_count = len(run.recommended_actions)
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{'#60a5fa' if action_count > 0 else '#6b8cae'};">{action_count}</div>
                <div class="kpi-label">Draft Actions</div>
                <div class="kpi-delta delta-neutral">{'Pending review' if action_count > 0 else 'None drafted'}</div>
            </div>
        """, unsafe_allow_html=True)

    with kpi5:
        max_score = max([f.exposure_score for f in run.exposure_findings], default=0.0)
        score_color = "#ef4444" if max_score >= 0.8 else "#f97316" if max_score >= 0.6 else "#eab308" if max_score >= 0.45 else "#4ade80"
        st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color:{score_color};">{max_score:.2f}</div>
                <div class="kpi-label">Max Exposure Score</div>
                <div class="kpi-delta delta-neutral">Highest risk asset</div>
            </div>
        """, unsafe_allow_html=True)

    # Outcome Banner
    if run.outcome and run.outcome.value == "actionable_exposure":
        st.error("🚨 **ACTIONABLE EXPOSURE DETECTED** — Review findings and draft actions immediately.")
    elif run.outcome and run.outcome.value == "stand_down_with_reason":
        st.warning("⚠️ **STAND DOWN** — Events tripped thresholds but no assets in your register are exposed.")
    else:
        st.success("✅ **STAND DOWN** — No events exceeded configured thresholds. Portfolio is clear.")
        if run.outcome_reason:
            st.caption(run.outcome_reason)
else:
    # Empty state KPIs
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    for col, label in zip([kpi1, kpi2, kpi3, kpi4, kpi5], 
                          ["Raw Events", "Flagged Events", "Exposed Assets", "Draft Actions", "Max Score"]):
        with col:
            st.markdown(f"""
                <div class="kpi-card">
                    <div class="kpi-value" style="color:#334155;">—</div>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-delta delta-neutral">Awaiting first run</div>
                </div>
            """, unsafe_allow_html=True)

    st.info("👈 Configure thresholds in the sidebar, then click **Live Poll** or **Replay Test** to begin monitoring.")


# ═══════════════════════════════════════════════════════════
# ROW 2: MAP + FINDINGS (Side by Side)
# ═══════════════════════════════════════════════════════════

st.markdown("<div class='section-header'>Operational View</div>", unsafe_allow_html=True)

map_col, findings_col = st.columns([2, 1])

with map_col:
    st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:8px;'>🗺️ Asset & Event Map</div>", unsafe_allow_html=True)

    if st.session_state.assets:
        lats = [a.latitude for a in st.session_state.assets]
        lons = [a.longitude for a in st.session_state.assets]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        m = folium.Map(
            location=[center_lat, center_lon], 
            zoom_start=4, 
            tiles="CartoDB dark_matter"
        )

        # Asset markers
        marker_cluster = MarkerCluster().add_to(m)
        for asset in st.session_state.assets:
            color = "#ef4444" if asset.criticality >= 5 else "#f97316" if asset.criticality >= 4 else "#3b82f6"
            folium.CircleMarker(
                location=[asset.latitude, asset.longitude],
                radius=5 + asset.criticality,
                popup=f"<b style='color:#e0e6ed'>{asset.name}</b><br><span style='color:#94a3b8'>ID: {asset.asset_id}</span><br><span style='color:#94a3b8'>Type: {asset.type}</span><br><span style='color:#f87171'>Criticality: {asset.criticality}/5</span>",
                tooltip=f"<span style='color:#e0e6ed'>{asset.name}</span>",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.75,
            ).add_to(marker_cluster)

        # Event markers from last run
        if st.session_state.last_run:
            for event in st.session_state.last_run.raw_events:
                if event.event_type == EventType.EARTHQUAKE and event.magnitude:
                    mag = event.magnitude
                    radius = max(4, mag * 3)
                    color = "#ef4444" if mag >= 5.0 else "#f97316" if mag >= 4.5 else "#eab308"
                    folium.CircleMarker(
                        location=[event.latitude, event.longitude],
                        radius=radius,
                        popup=f"<b style='color:#e0e6ed'>M{mag}</b><br><span style='color:#94a3b8'>{event.place or ''}</span>",
                        tooltip=f"M{mag} — {event.event_id[:20]}",
                        color=color,
                        fill=True,
                        fill_color=color,
                        fill_opacity=0.4,
                    ).add_to(m)

                    # Exposure radius ring for flagged events
                    if any(f.event_id == event.event_id for f in st.session_state.last_run.flagged_events):
                        folium.Circle(
                            location=[event.latitude, event.longitude],
                            radius=st.session_state.config.exposure_max_distance_km * 1000,
                            popup=f"Exposure radius: {st.session_state.config.exposure_max_distance_km} km",
                            color="#ef4444",
                            fill=False,
                            weight=1,
                            dash_array="5, 10",
                            opacity=0.5,
                        ).add_to(m)

        st_folium(m, width=700, height=450, returned_objects=[])
    else:
        st.warning("No assets loaded.")

with findings_col:
    st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:8px;'>⚠️ Exposure Findings</div>", unsafe_allow_html=True)

    if st.session_state.last_run and st.session_state.last_run.exposure_findings:
        for finding in st.session_state.last_run.exposure_findings:
            score = finding.exposure_score
            card_class = "finding-critical" if score >= 0.8 else "finding-high" if score >= 0.6 else "finding-medium" if score >= 0.45 else "finding-low"
            priority_badge = "🔴 P1" if score >= 0.8 else "🟠 P2" if score >= 0.6 else "🟡 P3" if score >= 0.45 else "🟢 P4"

            st.markdown(f"""
                <div class="finding-card {card_class}">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px;">
                        <span style="font-weight:700;color:#e0e6ed;font-size:0.9rem;">{finding.asset_name}</span>
                        <span style="font-size:0.75rem;font-weight:600;color:#94a3b8;">{priority_badge}</span>
                    </div>
                    <div style="font-size:0.75rem;color:#6b8cae;margin-bottom:4px;">
                        {finding.distance_km:.1f} km · Score {finding.exposure_score:.2f} · Criticality {int(finding.asset_criticality_weight * 5)}/5
                    </div>
                    <div style="font-size:0.8rem;color:#cbd5e1;">{finding.recommended_action}</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
            <div style="background:#111827;border:1px dashed #334155;border-radius:8px;padding:24px;text-align:center;">
                <div style="font-size:2rem;margin-bottom:8px;">✓</div>
                <div style="font-size:0.85rem;color:#6b8cae;font-weight:600;">No Exposure Findings</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:4px;">All assets clear of flagged events</div>
            </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# ROW 3: DRAFT ACTIONS + AUDIT TRAIL
# ═══════════════════════════════════════════════════════════

st.markdown("<div class='section-header'>Response & Audit</div>", unsafe_allow_html=True)

actions_col, audit_col = st.columns([1, 1])

with actions_col:
    st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:8px;'>🎫 Draft Actions</div>", unsafe_allow_html=True)

    if st.session_state.last_run and st.session_state.last_run.recommended_actions:
        for action in st.session_state.last_run.recommended_actions:
            priority_color = {"P1": "#ef4444", "P2": "#f97316", "P3": "#eab308", "P4": "#22c55e"}.get(action.priority, "#6b8cae")

            with st.expander(f"**{action.title}** · Priority {action.priority}"):
                st.markdown(f"<span style='color:{priority_color};font-weight:700;'>Priority: {action.priority}</span>", unsafe_allow_html=True)
                st.markdown(action.executive_briefing)

                tab_jira, tab_pd = st.tabs(["Jira Payload", "PagerDuty Payload"])
                with tab_jira:
                    st.json(action.draft_jira_payload)
                with tab_pd:
                    st.json(action.draft_pagerduty_payload)

                st.info("⚠️ Draft only — review before dispatching to webhook.")
    else:
        st.markdown("""
            <div style="background:#111827;border:1px dashed #334155;border-radius:8px;padding:24px;text-align:center;">
                <div style="font-size:2rem;margin-bottom:8px;">🎫</div>
                <div style="font-size:0.85rem;color:#6b8cae;font-weight:600;">No Draft Actions</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:4px;">Actions generate when exposure is detected</div>
            </div>
        """, unsafe_allow_html=True)

with audit_col:
    st.markdown("<div style='font-size:0.8rem;font-weight:600;color:#94a3b8;margin-bottom:8px;'>📋 Audit Trail</div>", unsafe_allow_html=True)

    if st.session_state.last_run and st.session_state.last_run.audit_log:
        audit_data = []
        for entry in st.session_state.last_run.audit_log:
            audit_data.append({
                "Time": entry.timestamp.strftime("%H:%M:%S") if entry.timestamp else "—",
                "Agent": entry.agent.upper()[:4],
                "Transition": entry.transition,
                "Decision": entry.decision,
            })

        df_audit = pd.DataFrame(audit_data)
        st.dataframe(
            df_audit,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Time": st.column_config.TextColumn("Time", width="small"),
                "Agent": st.column_config.TextColumn("Agent", width="small"),
                "Transition": st.column_config.TextColumn("Transition", width="medium"),
                "Decision": st.column_config.TextColumn("Decision", width="large"),
            }
        )

        csv = df_audit.to_csv(index=False)
        st.download_button(
            "⬇️ Export Audit CSV",
            csv,
            f"audit_{st.session_state.last_run.run_id}.csv",
            "text/csv",
            use_container_width=True,
        )
    else:
        st.markdown("""
            <div style="background:#111827;border:1px dashed #334155;border-radius:8px;padding:24px;text-align:center;">
                <div style="font-size:2rem;margin-bottom:8px;">📋</div>
                <div style="font-size:0.85rem;color:#6b8cae;font-weight:600;">No Audit Records</div>
                <div style="font-size:0.75rem;color:#475569;margin-top:4px;">Run a poll or replay to generate audit trail</div>
            </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# ROW 4: EVENT LOG (Collapsible)
# ═══════════════════════════════════════════════════════════

if st.session_state.last_run and st.session_state.last_run.raw_events:
    with st.expander("📡 Raw Event Log"):
        event_data = []
        for e in st.session_state.last_run.raw_events:
            event_data.append({
                "ID": e.event_id[:30],
                "Type": e.event_type.value,
                "Source": e.source,
                "Time": e.timestamp.strftime("%H:%M:%S UTC") if e.timestamp else "—",
                "Location": f"{e.latitude:.3f}, {e.longitude:.3f}",
                "Magnitude": e.magnitude,
                "AQI": e.aqi,
            })
        st.dataframe(pd.DataFrame(event_data), use_container_width=True, hide_index=True)


# ── Footer ────────────────────────────────────────────────

st.markdown("<hr style='border-color:#1e3a5f;margin-top:32px;'>", unsafe_allow_html=True)
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:center;padding:8px 0;">
    <div style="font-size:0.7rem;color:#475569;">
        Asset Exposure Watch v1.0 · Built with LangGraph · Data: USGS · Open-Meteo
    </div>
    <div style="font-size:0.7rem;color:#475569;">
        Sample register is illustrative — replace with your own asset footprint
    </div>
</div>
""", unsafe_allow_html=True)
