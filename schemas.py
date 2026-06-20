"""
schemas.py — All Pydantic models for Asset Exposure Watch.
Built from live API inspection, not assumptions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict
import math


class EventType(str, Enum):
    EARTHQUAKE = "earthquake"
    AIR_QUALITY = "air_quality"
    FINANCIAL_SENTIMENT = "financial_sentiment"
    UNKNOWN = "unknown"


class AssetType(str, Enum):
    FACILITY = "facility"
    WAREHOUSE = "warehouse"
    SUPPLY_NODE = "supply-node"
    MARKET_POSITION = "market-position"


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RunOutcome(str, Enum):
    STAND_DOWN = "stand_down"
    STAND_DOWN_WITH_REASON = "stand_down_with_reason"
    ACTIONABLE_EXPOSURE = "actionable_exposure"


# ── Configuration ──────────────────────────────────────────

class AppConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # Polling
    poll_interval_seconds: int = Field(default=60, ge=10, le=3600)

    # Triage thresholds
    earthquake_min_magnitude: float = Field(default=4.5, ge=0.0, le=10.0)
    air_quality_min_aqi: int = Field(default=150, ge=0, le=500)

    # Exposure scoring
    exposure_max_distance_km: float = Field(default=500.0, ge=1.0, le=2000.0)
    exposure_floor_score: float = Field(default=0.3, ge=0.0, le=1.0)
    distance_decay_factor: float = Field(default=0.05, ge=0.001, le=0.5)

    # Budget
    token_budget_ceiling: float = Field(default=1000.0, ge=0.0)

    # Optional third-party
    alpha_vantage_api_key: Optional[str] = None

    # Replay
    replay_mode: bool = False


# ── Raw Events (from external APIs) ─────────────────────────

class RawEvent(BaseModel):
    model_config = ConfigDict(extra="allow")

    event_id: str
    event_type: EventType
    source: str  # e.g., "usgs", "open-meteo"
    timestamp: datetime
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    raw_payload: Dict[str, Any]

    # Type-specific fields
    magnitude: Optional[float] = None  # for earthquakes
    depth_km: Optional[float] = None
    place: Optional[str] = None
    aqi: Optional[int] = None  # for air quality
    aqi_level: Optional[str] = None

    @field_validator("timestamp", mode="before")
    @classmethod
    def parse_timestamp(cls, v):
        if isinstance(v, (int, float)):
            return datetime.fromtimestamp(v / 1000.0 if v > 1e10 else v, tz=timezone.utc)
        if isinstance(v, str):
            return datetime.fromisoformat(v.replace("Z", "+00:00"))
        return v


# ── Flagged Events (post-triage) ───────────────────────────

class FlaggedEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    event_type: EventType
    source: str
    timestamp: datetime
    latitude: float
    longitude: float
    severity: SeverityLevel
    severity_score: float = Field(..., ge=0.0, le=1.0)
    triage_reason: str  # one-sentence explanation
    threshold_applied: str  # e.g., "magnitude >= 4.5"
    raw_event: RawEvent


# ── Asset ─────────────────────────────────────────────────

class Asset(BaseModel):
    model_config = ConfigDict(extra="allow")

    asset_id: str
    name: str
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    type: AssetType
    criticality: int = Field(..., ge=1, le=5)
    sector: Optional[str] = None

    @property
    def criticality_weight(self) -> float:
        """Normalize criticality 1-5 to 0.2-1.0."""
        return self.criticality / 5.0


# ── Exposure Findings ─────────────────────────────────────

class ExposureFinding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    finding_id: str
    event_id: str
    asset_id: str
    asset_name: str
    distance_km: float = Field(..., ge=0.0)
    event_severity_score: float
    asset_criticality_weight: float
    proximity_weight: float
    exposure_score: float = Field(..., ge=0.0, le=1.0)
    rationale: str  # one-sentence plain English
    recommended_action: str
    confidence: str  # caveats


# ── Recommended Actions ───────────────────────────────────

class ActionPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    action_id: str
    event_id: str
    finding_ids: List[str]
    title: str
    description: str
    priority: Literal["P1", "P2", "P3", "P4"]
    affected_assets: List[str]
    draft_jira_payload: Dict[str, Any]
    draft_pagerduty_payload: Dict[str, Any]
    executive_briefing: str
    created_at: datetime


# ── Audit Log ─────────────────────────────────────────────

class AuditLogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timestamp: datetime
    run_id: str
    agent: Literal["triage", "exposure", "action", "orchestrator"]
    event_id: Optional[str] = None
    transition: str  # e.g., "triage->stand_down", "triage->exposure"
    decision: str
    score_or_threshold: Optional[str] = None
    details: Optional[str] = None


# ── Shared State (LangGraph) ─────────────────────────────

class SharedState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    config: AppConfig
    started_at: datetime

    raw_events: List[RawEvent] = Field(default_factory=list)
    flagged_events: List[FlaggedEvent] = Field(default_factory=list)
    exposure_findings: List[ExposureFinding] = Field(default_factory=list)
    recommended_actions: List[ActionPayload] = Field(default_factory=list)
    audit_log: List[AuditLogEntry] = Field(default_factory=list)

    outcome: Optional[RunOutcome] = None
    outcome_reason: Optional[str] = None
    token_spend: float = 0.0

    finished_at: Optional[datetime] = None

    def log(self, agent: str, transition: str, decision: str, 
            event_id: Optional[str] = None, score_or_threshold: Optional[str] = None,
            details: Optional[str] = None):
        entry = AuditLogEntry(
            timestamp=datetime.now(timezone.utc),
            run_id=self.run_id,
            agent=agent,
            event_id=event_id,
            transition=transition,
            decision=decision,
            score_or_threshold=score_or_threshold,
            details=details
        )
        self.audit_log.append(entry)
