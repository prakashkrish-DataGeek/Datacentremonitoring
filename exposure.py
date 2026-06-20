"""
exposure.py — Transparent geospatial and criticality scoring.
Independently testable. Every score is explainable in one sentence.
"""
from __future__ import annotations

import math
from typing import List, Tuple
from schemas import Asset, RawEvent, FlaggedEvent, ExposureFinding, AppConfig


EARTH_RADIUS_KM = 6371.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Straight-line (great-circle) distance between two points on Earth.
    Transparent, inspectable arithmetic. No black boxes.
    """
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def proximity_weight(distance_km: float, max_distance_km: float, decay_factor: float) -> float:
    """
    Exponential distance decay: closer = higher weight.
    weight = exp(-decay_factor * distance_km), clipped at max_distance.
    """
    if distance_km >= max_distance_km:
        return 0.0
    return math.exp(-decay_factor * distance_km)


def compute_exposure_score(
    event_severity_score: float,
    asset_criticality_weight: float,
    proximity_weight: float,
) -> float:
    """
    Final exposure score = weighted sum of three factors, normalized to [0, 1].
    Weights: severity 0.4, criticality 0.3, proximity 0.3.
    """
    raw = (0.4 * event_severity_score) + (0.3 * asset_criticality_weight) + (0.3 * proximity_weight)
    # Normalize by max possible (0.4 + 0.3 + 0.3 = 1.0), so raw is already in [0,1]
    return min(1.0, max(0.0, raw))


def score_event_severity(flagged: FlaggedEvent) -> float:
    """
    Convert flagged event severity to a 0-1 score.
    """
    mapping = {
        "low": 0.25,
        "medium": 0.5,
        "high": 0.75,
        "critical": 1.0,
    }
    return mapping.get(flagged.severity.value, 0.5)


def assess_exposure(
    flagged_event: FlaggedEvent,
    assets: List[Asset],
    config: AppConfig,
) -> Tuple[List[ExposureFinding], str]:
    """
    For a single flagged event, compute exposure against all assets.
    Returns: (findings, stand_down_reason).
    If no asset clears the exposure floor, returns empty findings with reason.
    """
    findings: List[ExposureFinding] = []
    event_sev = score_event_severity(flagged_event)

    for asset in assets:
        dist = haversine_distance(
            flagged_event.latitude, flagged_event.longitude,
            asset.latitude, asset.longitude
        )

        prox = proximity_weight(dist, config.exposure_max_distance_km, config.distance_decay_factor)

        if prox <= 0.0:
            continue  # Beyond max distance

        score = compute_exposure_score(event_sev, asset.criticality_weight, prox)

        if score >= config.exposure_floor_score:
            rationale = (
                f"{asset.name} ({asset.asset_id}) is {dist:.1f} km from a "
                f"{flagged_event.event_type.value} event; "
                f"event severity={event_sev:.2f}, asset criticality={asset.criticality}/5, "
                f"proximity weight={prox:.2f} → exposure score={score:.2f}"
            )

            # Draft recommended action based on event type and score
            if score >= 0.8:
                action = f"IMMEDIATE: Inspect {asset.name} for structural damage and suspend operations until cleared."
                priority = "P1"
            elif score >= 0.6:
                action = f"URGENT: Dispatch inspection team to {asset.name} within 2 hours."
                priority = "P2"
            elif score >= 0.45:
                action = f"ADVISORY: Monitor {asset.name} closely and prepare contingency plans."
                priority = "P3"
            else:
                action = f"WATCH: Log {asset.name} for next routine check."
                priority = "P4"

            finding = ExposureFinding(
                finding_id=f"{flagged_event.event_id}_{asset.asset_id}",
                event_id=flagged_event.event_id,
                asset_id=asset.asset_id,
                asset_name=asset.name,
                distance_km=dist,
                event_severity_score=event_sev,
                asset_criticality_weight=asset.criticality_weight,
                proximity_weight=prox,
                exposure_score=score,
                rationale=rationale,
                recommended_action=action,
                confidence=f"Straight-line distance only; does not account for terrain, infrastructure, or travel time."
            )
            findings.append(finding)

    # Sort by exposure score descending
    findings.sort(key=lambda f: f.exposure_score, reverse=True)

    if not findings:
        reason = (
            f"Event {flagged_event.event_id} tripped threshold ({flagged_event.threshold_applied}), "
            f"but no asset within {config.exposure_max_distance_km} km cleared the exposure floor "
            f"of {config.exposure_floor_score}."
        )
        return findings, reason

    return findings, ""
