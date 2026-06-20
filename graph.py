"""
graph.py — LangGraph topology with three agents and conditional routing.
Most events route to "stand down". Only anomalies proceed to analysis.
"""
from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional, Annotated, Sequence, TypedDict
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

from schemas import (
    AppConfig, RawEvent, FlaggedEvent, ExposureFinding, ActionPayload,
    SharedState, EventType, SeverityLevel, RunOutcome, AuditLogEntry
)
from exposure import assess_exposure, score_event_severity
from ingestion import ThrottledPoller, load_replay_event


# ── LangGraph State ─────────────────────────────────────────

class GraphState(TypedDict):
    """LangGraph state dictionary. Must be JSON-serializable."""
    shared: Dict[str, Any]  # Serialized SharedState


def _deserialize_shared(data: Dict[str, Any]) -> SharedState:
    return SharedState.model_validate(data)


def _serialize_shared(state: SharedState) -> Dict[str, Any]:
    return state.model_dump(mode="json")


# ── Agent A: Triage ───────────────────────────────────────

def agent_triage(state: GraphState) -> GraphState:
    """
    Agent A — Triage: Does this event matter at all?
    Applies transparent thresholds. Routes to stand_down or exposure.
    """
    shared = _deserialize_shared(state["shared"])
    config = shared.config

    new_flagged: List[FlaggedEvent] = []

    for event in shared.raw_events:
        # Skip if already processed in this run
        if any(f.event_id == event.event_id for f in shared.flagged_events):
            continue

        if event.event_type == EventType.EARTHQUAKE:
            mag = event.magnitude or 0.0
            if mag >= config.earthquake_min_magnitude:
                # Severity mapping: 4.5-5.0 = medium, 5.0-6.0 = high, 6.0+ = critical
                if mag >= 6.0:
                    severity = SeverityLevel.CRITICAL
                    sev_score = 1.0
                elif mag >= 5.0:
                    severity = SeverityLevel.HIGH
                    sev_score = 0.75
                else:
                    severity = SeverityLevel.MEDIUM
                    sev_score = 0.5

                flagged = FlaggedEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    source=event.source,
                    timestamp=event.timestamp,
                    latitude=event.latitude,
                    longitude=event.longitude,
                    severity=severity,
                    severity_score=sev_score,
                    triage_reason=f"Magnitude {mag} >= threshold {config.earthquake_min_magnitude}",
                    threshold_applied=f"magnitude >= {config.earthquake_min_magnitude}",
                    raw_event=event,
                )
                new_flagged.append(flagged)
                shared.log(
                    agent="triage",
                    transition="triage->flagged",
                    decision="event flagged",
                    event_id=event.event_id,
                    score_or_threshold=f"mag={mag} >= {config.earthquake_min_magnitude}",
                    details=f"Severity={severity.value}, score={sev_score}"
                )
            else:
                shared.log(
                    agent="triage",
                    transition="triage->stand_down",
                    decision="event below threshold",
                    event_id=event.event_id,
                    score_or_threshold=f"mag={mag} < {config.earthquake_min_magnitude}",
                    details="No action required"
                )

        elif event.event_type == EventType.AIR_QUALITY:
            aqi = event.aqi or 0
            if aqi >= config.air_quality_min_aqi:
                if aqi >= 300:
                    severity = SeverityLevel.CRITICAL
                    sev_score = 1.0
                elif aqi >= 200:
                    severity = SeverityLevel.HIGH
                    sev_score = 0.75
                else:
                    severity = SeverityLevel.MEDIUM
                    sev_score = 0.5

                flagged = FlaggedEvent(
                    event_id=event.event_id,
                    event_type=event.event_type,
                    source=event.source,
                    timestamp=event.timestamp,
                    latitude=event.latitude,
                    longitude=event.longitude,
                    severity=severity,
                    severity_score=sev_score,
                    triage_reason=f"AQI {aqi} >= threshold {config.air_quality_min_aqi}",
                    threshold_applied=f"aqi >= {config.air_quality_min_aqi}",
                    raw_event=event,
                )
                new_flagged.append(flagged)
                shared.log(
                    agent="triage",
                    transition="triage->flagged",
                    decision="event flagged",
                    event_id=event.event_id,
                    score_or_threshold=f"aqi={aqi} >= {config.air_quality_min_aqi}",
                )
            else:
                shared.log(
                    agent="triage",
                    transition="triage->stand_down",
                    decision="event below threshold",
                    event_id=event.event_id,
                    score_or_threshold=f"aqi={aqi} < {config.air_quality_min_aqi}",
                )

        elif event.event_type == EventType.FINANCIAL_SENTIMENT:
            # Always flag financial sentiment if it reaches this stage (pre-filtered in ingestion)
            flagged = FlaggedEvent(
                event_id=event.event_id,
                event_type=event.event_type,
                source=event.source,
                timestamp=event.timestamp,
                latitude=event.latitude,
                longitude=event.longitude,
                severity=SeverityLevel.HIGH,
                severity_score=0.75,
                triage_reason="Extreme negative financial sentiment detected",
                threshold_applied="sentiment < -0.5",
                raw_event=event,
            )
            new_flagged.append(flagged)
            shared.log(
                agent="triage",
                transition="triage->flagged",
                decision="event flagged",
                event_id=event.event_id,
                score_or_threshold="sentiment < -0.5",
            )

    shared.flagged_events.extend(new_flagged)

    # If no events were flagged at all, route to stand_down
    if not shared.flagged_events:
        shared.outcome = RunOutcome.STAND_DOWN
        shared.outcome_reason = "No events tripped thresholds in this cycle."
        shared.log(
            agent="orchestrator",
            transition="triage->stand_down",
            decision="stand down",
            details="No events exceeded configured thresholds"
        )

    return {"shared": _serialize_shared(shared)}


# ── Agent B: Exposure ─────────────────────────────────────

def agent_exposure(state: GraphState, assets: List[Any]) -> GraphState:
    """
    Agent B — Exposure: Does it touch anything I own?
    Computes haversine distance, proximity weight, and exposure score.
    """
    shared = _deserialize_shared(state["shared"])
    config = shared.config

    if shared.outcome == RunOutcome.STAND_DOWN:
        return state  # Skip if already standing down

    all_findings: List[ExposureFinding] = []

    for flagged in shared.flagged_events:
        findings, reason = assess_exposure(flagged, assets, config)

        if findings:
            all_findings.extend(findings)
            shared.log(
                agent="exposure",
                transition="exposure->finding",
                decision=f"{len(findings)} assets exposed",
                event_id=flagged.event_id,
                score_or_threshold=f"exposure_floor={config.exposure_floor_score}",
                details=f"Top finding: {findings[0].rationale}"
            )
        else:
            shared.log(
                agent="exposure",
                transition="exposure->stand_down_with_reason",
                decision="no exposure",
                event_id=flagged.event_id,
                score_or_threshold=f"max_dist={config.exposure_max_distance_km} km",
                details=reason
            )

    shared.exposure_findings.extend(all_findings)

    if not all_findings:
        shared.outcome = RunOutcome.STAND_DOWN_WITH_REASON
        shared.outcome_reason = (
            f"Events tripped thresholds but no asset within {config.exposure_max_distance_km} km "
            f"cleared the exposure floor of {config.exposure_floor_score}."
        )
        shared.log(
            agent="orchestrator",
            transition="exposure->stand_down_with_reason",
            decision="stand down with reason",
            details=shared.outcome_reason
        )

    return {"shared": _serialize_shared(shared)}


# ── Agent C: Action ───────────────────────────────────────

def agent_action(state: GraphState) -> GraphState:
    """
    Agent C — Action: What do I do about it?
    Drafts structured action payloads and executive briefing.
    """
    shared = _deserialize_shared(state["shared"])
    config = shared.config

    if shared.outcome in (RunOutcome.STAND_DOWN, RunOutcome.STAND_DOWN_WITH_REASON):
        return state  # Skip if standing down

    # Group findings by event_id
    by_event: Dict[str, List[ExposureFinding]] = {}
    for finding in shared.exposure_findings:
        by_event.setdefault(finding.event_id, []).append(finding)

    for event_id, findings in by_event.items():
        findings.sort(key=lambda f: f.exposure_score, reverse=True)
        top = findings[0]

        # Determine overall priority
        max_score = findings[0].exposure_score
        if max_score >= 0.8:
            priority = "P1"
        elif max_score >= 0.6:
            priority = "P2"
        elif max_score >= 0.45:
            priority = "P3"
        else:
            priority = "P4"

        affected = [f.asset_name for f in findings]

        # Draft Jira-style payload
        jira_payload = {
            "fields": {
                "project": {"key": "RISK"},
                "summary": f"[{priority}] Exposure Alert: {top.event_id}",
                "description": (
                    f"h1. Asset Exposure Alert\n\n"
                    f"*Event:* {top.event_id}\n"
                    f"*Affected Assets:* {', '.join(affected)}\n"
                    f"*Max Exposure Score:* {max_score:.2f}\n"
                    f"*Top Rationale:* {top.rationale}\n\n"
                    f"h2. Recommended Actions\n"
                    + "\n".join(f"* {f.recommended_action}" for f in findings)
                ),
                "issuetype": {"name": "Incident"},
                "priority": {"name": priority},
                "labels": ["auto-exposure", "agent-generated"],
            }
        }

        # Draft PagerDuty-style payload
        pagerduty_payload = {
            "routing_key": "YOUR_INTEGRATION_KEY",
            "event_action": "trigger",
            "payload": {
                "summary": f"Asset Exposure Alert: {', '.join(affected[:3])}",
                "severity": "critical" if priority == "P1" else "error" if priority == "P2" else "warning",
                "source": "asset-exposure-watch",
                "custom_details": {
                    "event_id": top.event_id,
                    "affected_assets": affected,
                    "max_exposure_score": max_score,
                    "findings_count": len(findings),
                }
            }
        }

        # Executive briefing in Markdown
        briefing_lines = [
            "## Asset Exposure Alert — " + datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
            "",
            "### What Happened",
            "A " + top.event_id + " event has been detected with potential exposure to your asset portfolio.",
            "",
            "### What Is Exposed",
            "| Asset | Distance | Score | Action |",
            "|-------|----------|-------|--------|",
        ]
        for finding_item in findings[:5]:
            dist_str = str(round(finding_item.distance_km, 1)) + " km"
            score_str = str(round(finding_item.exposure_score, 2))
            row = "| " + finding_item.asset_name + " | " + dist_str + " | " + score_str + " | " + finding_item.recommended_action + " |"
            briefing_lines.append(row)
        briefing_lines.extend([
            "",
            "### Recommended Action",
            "Trigger " + priority + " response protocol for affected assets. Review draft tickets below.",
            "",
            "### Confidence & Caveats",
            "- Scoring uses straight-line (haversine) distance, not travel time or infrastructure topology.",
            "- Exposure model weights: event severity 40%, asset criticality 30%, proximity 30%.",
            "- Sample register may not reflect your actual asset footprint.",
            "",
            "### Draft Tickets",
            "- Jira: `" + json.dumps(jira_payload, indent=2) + "`",
            "- PagerDuty: `" + json.dumps(pagerduty_payload, indent=2) + "`",
        ])
        briefing = "\n".join(briefing_lines)

        action = ActionPayload(
            action_id=f"act_{event_id}_{uuid.uuid4().hex[:8]}",
            event_id=event_id,
            finding_ids=[f.finding_id for f in findings],
            title=f"Exposure Alert: {event_id}",
            description=f"{len(findings)} assets exposed. Max score: {max_score:.2f}",
            priority=priority,
            affected_assets=affected,
            draft_jira_payload=jira_payload,
            draft_pagerduty_payload=pagerduty_payload,
            executive_briefing=briefing,
            created_at=datetime.now(timezone.utc),
        )

        shared.recommended_actions.append(action)
        shared.log(
            agent="action",
            transition="action->draft_payload",
            decision="draft action created",
            event_id=event_id,
            score_or_threshold=f"priority={priority}",
            details=f"{len(findings)} findings, max_score={max_score:.2f}"
        )

    shared.outcome = RunOutcome.ACTIONABLE_EXPOSURE
    shared.outcome_reason = f"{len(shared.recommended_actions)} draft action(s) created."
    shared.log(
        agent="orchestrator",
        transition="action->actionable_exposure",
        decision="actionable exposure",
        details=shared.outcome_reason
    )

    return {"shared": _serialize_shared(shared)}


# ── Conditional Routing ───────────────────────────────────

def route_after_triage(state: GraphState) -> str:
    shared = _deserialize_shared(state["shared"])
    if shared.outcome == RunOutcome.STAND_DOWN:
        return "stand_down"
    return "exposure"


def route_after_exposure(state: GraphState) -> str:
    shared = _deserialize_shared(state["shared"])
    if shared.outcome == RunOutcome.STAND_DOWN_WITH_REASON:
        return "stand_down"
    return "action"


def finalize(state: GraphState) -> GraphState:
    shared = _deserialize_shared(state["shared"])
    shared.finished_at = datetime.now(timezone.utc)
    shared.log(
        agent="orchestrator",
        transition="finalize",
        decision="run complete",
        details=f"outcome={shared.outcome.value if shared.outcome else 'unknown'}"
    )
    return {"shared": _serialize_shared(shared)}


# ── Build Graph ───────────────────────────────────────────

def build_graph(assets: List[Any]) -> StateGraph:
    """Build and compile the LangGraph state machine."""

    workflow = StateGraph(GraphState)

    # Nodes
    workflow.add_node("triage", agent_triage)
    workflow.add_node("exposure", lambda state: agent_exposure(state, assets))
    workflow.add_node("action", agent_action)
    workflow.add_node("stand_down", finalize)
    workflow.add_node("finalize", finalize)

    # Edges
    workflow.set_entry_point("triage")

    workflow.add_conditional_edges(
        "triage",
        route_after_triage,
        {
            "stand_down": "stand_down",
            "exposure": "exposure",
        }
    )

    workflow.add_conditional_edges(
        "exposure",
        route_after_exposure,
        {
            "stand_down": "stand_down",
            "action": "action",
        }
    )

    workflow.add_edge("action", "finalize")
    workflow.add_edge("stand_down", "finalize")
    workflow.add_edge("finalize", END)

    return workflow.compile()


# ── Orchestrator: Run Once ────────────────────────────────

async def run_once(
    config: AppConfig,
    assets: List[Any],
    raw_events: List[RawEvent],
) -> SharedState:
    """
    Execute the full graph once with provided raw events.
    """
    run_id = f"run_{uuid.uuid4().hex[:12]}"
    shared = SharedState(
        run_id=run_id,
        config=config,
        started_at=datetime.now(timezone.utc),
        raw_events=raw_events,
    )

    graph = build_graph(assets)
    initial_state = {"shared": _serialize_shared(shared)}

    result = await graph.ainvoke(initial_state)

    return _deserialize_shared(result["shared"])


# ── Orchestrator: Replay Mode ─────────────────────────────

async def run_replay(
    config: AppConfig,
    assets: List[Any],
    replay_file: str,
) -> SharedState:
    """
    Replay a captured historical event through the full graph.
    Deterministic test path.
    """
    event = load_replay_event(replay_file)
    if event is None:
        raise FileNotFoundError(f"Replay file not found: {replay_file}")

    return await run_once(config, assets, [event])
