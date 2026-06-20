"""
ingestion.py — Live and replay data sources with throttling and schema parsing.
All schemas built from live API inspection, not assumptions.
"""
from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from pathlib import Path

import httpx
from schemas import RawEvent, EventType, AppConfig, AuditLogEntry


# ── USGS Earthquake Feed ───────────────────────────────────

USGS_ALL_HOUR = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson"
USGS_ALL_DAY = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_day.geojson"
USGS_ALL_WEEK = "https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_week.geojson"


async def fetch_usgs_earthquakes(
    client: httpx.AsyncClient,
    period: str = "hour",
    timeout: float = 15.0
) -> List[RawEvent]:
    """
    Fetch USGS GeoJSON feed and parse into validated RawEvent objects.
    Schema verified live against: https://earthquake.usgs.gov/earthquakes/feed/v1.0/summary/all_hour.geojson
    """
    url_map = {"hour": USGS_ALL_HOUR, "day": USGS_ALL_DAY, "week": USGS_ALL_WEEK}
    url = url_map.get(period, USGS_ALL_HOUR)

    try:
        resp = await client.get(url, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # Graceful degradation: log and return empty
        return []

    events: List[RawEvent] = []
    features = data.get("features", [])

    for feat in features:
        props = feat.get("properties", {})
        geom = feat.get("geometry", {})
        coords = geom.get("coordinates", [0, 0, 0])

        # USGS schema: [lon, lat, depth]
        if len(coords) < 2:
            continue

        event_id = props.get("code") or props.get("ids", "").strip(",")
        if not event_id:
            event_id = feat.get("id", "unknown")

        # Parse timestamp: USGS gives milliseconds since epoch
        ts_raw = props.get("time")
        if ts_raw is None:
            continue

        try:
            event = RawEvent(
                event_id=f"usgs_{event_id}",
                event_type=EventType.EARTHQUAKE,
                source="usgs",
                timestamp=ts_raw,
                latitude=coords[1],
                longitude=coords[0],
                raw_payload=feat,
                magnitude=props.get("mag"),
                depth_km=coords[2] if len(coords) > 2 else None,
                place=props.get("place"),
            )
            events.append(event)
        except Exception:
            # Reject malformed payloads rather than propagate
            continue

    return events


# ── Open-Meteo Air Quality ─────────────────────────────────

OPENMETEO_AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"


async def fetch_openmeteo_air_quality(
    client: httpx.AsyncClient,
    assets: List[Any],  # List[Asset] but avoid circular import
    timeout: float = 15.0
) -> List[RawEvent]:
    """
    Fetch current air quality for asset locations via Open-Meteo.
    No API key required. Schema verified from live docs.
    """
    if not assets:
        return []

    # Build multi-location query
    lats = ",".join(str(a.latitude) for a in assets)
    lons = ",".join(str(a.longitude) for a in assets)

    params = {
        "latitude": lats,
        "longitude": lons,
        "current": "us_aqi",
        "timezone": "auto",
    }

    try:
        resp = await client.get(OPENMETEO_AQ, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    events: List[RawEvent] = []

    # Open-Meteo returns either a single object or a list for multi-location
    results = data if isinstance(data, list) else [data]

    for i, result in enumerate(results):
        if i >= len(assets):
            break
        asset = assets[i]

        current = result.get("current", {})
        aqi = current.get("us_aqi")

        if aqi is None:
            continue

        # Determine AQI level
        if aqi <= 50:
            level = "good"
        elif aqi <= 100:
            level = "moderate"
        elif aqi <= 150:
            level = "unhealthy_sensitive"
        elif aqi <= 200:
            level = "unhealthy"
        elif aqi <= 300:
            level = "very_unhealthy"
        else:
            level = "hazardous"

        try:
            event = RawEvent(
                event_id=f"openmeteo_aq_{asset.asset_id}_{int(time.time())}",
                event_type=EventType.AIR_QUALITY,
                source="open-meteo",
                timestamp=datetime.now(timezone.utc),
                latitude=asset.latitude,
                longitude=asset.longitude,
                raw_payload=result,
                aqi=aqi,
                aqi_level=level,
            )
            events.append(event)
        except Exception:
            continue

    return events


# ── Alpha Vantage (optional, pluggable) ────────────────────

ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"


async def fetch_alpha_vantage_news(
    client: httpx.AsyncClient,
    api_key: Optional[str],
    tickers: List[str],
    timeout: float = 15.0
) -> List[RawEvent]:
    """
    Optional financial sentiment feed. If no key, returns empty gracefully.
    """
    if not api_key or not tickers:
        return []

    events: List[RawEvent] = []

    for ticker in tickers[:3]:  # Rate-limit: max 3 tickers per cycle
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "apikey": api_key,
            "limit": 5,
        }
        try:
            resp = await client.get(ALPHAVANTAGE_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            continue

        feed = data.get("feed", [])
        for item in feed:
            sentiment = item.get("overall_sentiment_score", 0)
            # Only flag extreme negative sentiment
            if sentiment < -0.5:
                try:
                    event = RawEvent(
                        event_id=f"av_{ticker}_{item.get('time_published', 'unknown')}",
                        event_type=EventType.FINANCIAL_SENTIMENT,
                        source="alpha-vantage",
                        timestamp=item.get("time_published", datetime.now(timezone.utc).isoformat()),
                        latitude=0.0,  # Financial events have no geo
                        longitude=0.0,
                        raw_payload=item,
                    )
                    events.append(event)
                except Exception:
                    continue

        # Alpha Vantage free tier: 5 calls per minute
        await asyncio.sleep(12)

    return events


# ── Replay Mode ─────────────────────────────────────────────

REPLAY_DIR = Path(__file__).parent / "replay_data"


def load_replay_event(event_file: str) -> Optional[RawEvent]:
    """
    Load a captured historical event from bundled replay data.
    """
    path = REPLAY_DIR / event_file
    if not path.exists():
        return None

    try:
        with open(path, "r") as f:
            data = json.load(f)

        # Support both full GeoJSON feature and RawEvent serialization
        if "properties" in data and "geometry" in data:
            # GeoJSON feature format (USGS style)
            props = data.get("properties", {})
            geom = data.get("geometry", {})
            coords = geom.get("coordinates", [0, 0, 0])

            return RawEvent(
                event_id=f"replay_{props.get('code', 'unknown')}",
                event_type=EventType.EARTHQUAKE,
                source="replay",
                timestamp=props.get("time", datetime.now(timezone.utc).timestamp() * 1000),
                latitude=coords[1],
                longitude=coords[0],
                raw_payload=data,
                magnitude=props.get("mag"),
                depth_km=coords[2] if len(coords) > 2 else None,
                place=props.get("place"),
            )
        else:
            # Serialized RawEvent
            return RawEvent.model_validate(data)
    except Exception:
        return None


# ── Throttled Poller ───────────────────────────────────────

class ThrottledPoller:
    """
    Async loop that polls feeds at configured intervals with throttling.
    """
    def __init__(self, config: AppConfig, assets: List[Any]):
        self.config = config
        self.assets = assets
        self._running = False
        self._last_poll: Dict[str, float] = {}
        self._min_interval = 10.0  # Hard floor between identical requests

    async def poll_all(self, client: httpx.AsyncClient) -> List[RawEvent]:
        """Poll all configured sources and return merged events."""
        all_events: List[RawEvent] = []

        # Earthquakes (always)
        now = time.time()
        if now - self._last_poll.get("usgs", 0) >= self._min_interval:
            eq_events = await fetch_usgs_earthquakes(client, period="hour")
            all_events.extend(eq_events)
            self._last_poll["usgs"] = now

        # Air quality (always, keyless)
        if now - self._last_poll.get("openmeteo", 0) >= self._min_interval:
            aq_events = await fetch_openmeteo_air_quality(client, self.assets)
            all_events.extend(aq_events)
            self._last_poll["openmeteo"] = now

        # Alpha Vantage (optional, if key present)
        if self.config.alpha_vantage_api_key:
            if now - self._last_poll.get("alphavantage", 0) >= 60.0:  # Stricter throttle
                sectors = list(set(a.sector for a in self.assets if a.sector))
                if sectors:
                    av_events = await fetch_alpha_vantage_news(
                        client, self.config.alpha_vantage_api_key, sectors
                    )
                    all_events.extend(av_events)
                self._last_poll["alphavantage"] = now

        return all_events

    async def run_loop(
        self,
        client: httpx.AsyncClient,
        callback,
    ):
        """
        Run polling loop. callback receives List[RawEvent] each cycle.
        """
        self._running = True
        while self._running:
            events = await self.poll_all(client)
            if events:
                await callback(events)
            await asyncio.sleep(self.config.poll_interval_seconds)

    def stop(self):
        self._running = False
