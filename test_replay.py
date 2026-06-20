"""
test_replay.py — Deterministic assertions over replay mode.
Proves: the same event always produces the same exposure findings.
"""
from __future__ import annotations

import asyncio
import pytest
from pathlib import Path

from schemas import AppConfig, Asset, EventType, RunOutcome, SeverityLevel
from exposure import haversine_distance, proximity_weight, compute_exposure_score
from graph import run_replay


# ── Sample Assets Subset for Testing ──────────────────────

TEST_ASSETS = [
    Asset(asset_id="LAX-01", name="LA Distribution", latitude=34.0522, longitude=-118.2437, type="warehouse", criticality=5, sector="logistics"),
    Asset(asset_id="SJC-08", name="San Jose Chip Foundry", latitude=37.3382, longitude=-121.8863, type="facility", criticality=5, sector="technology"),
    Asset(asset_id="BAK-11", name="Bakersfield Oil Terminal", latitude=35.3733, longitude=-119.0187, type="facility", criticality=4, sector="energy"),
    Asset(asset_id="RNO-12", name="Reno Logistics Park", latitude=39.5296, longitude=-119.8138, type="warehouse", criticality=3, sector="logistics"),
    Asset(asset_id="FAT-10", name="Fresno Agricultural Hub", latitude=36.7378, longitude=-119.7871, type="supply-node", criticality=3, sector="agriculture"),
    Asset(asset_id="LAS-06", name="Las Vegas Data Center", latitude=36.1699, longitude=-115.1398, type="facility", criticality=5, sector="technology"),
    Asset(asset_id="FAR-99", name="Fargo Remote Warehouse", latitude=46.8772, longitude=-96.7898, type="warehouse", criticality=2, sector="logistics"),
]


# ── Unit Tests: Exposure Math ─────────────────────────────

def test_haversine_distance_known_values():
    """Haversine distance against known reference."""
    # London to Paris ~344 km
    dist = haversine_distance(51.5074, -0.1278, 48.8566, 2.3522)
    assert 330 < dist < 360, f"Expected ~344 km, got {dist}"


def test_haversine_distance_zero():
    """Same point = 0 km."""
    assert haversine_distance(35.0, -117.0, 35.0, -117.0) == 0.0


def test_proximity_weight_at_zero():
    """At distance 0, weight should be 1.0."""
    assert proximity_weight(0.0, 500.0, 0.05) == 1.0


def test_proximity_weight_at_max():
    """At max distance, weight should be 0.0."""
    assert proximity_weight(500.0, 500.0, 0.05) == 0.0


def test_proximity_weight_beyond_max():
    """Beyond max distance, weight should be 0.0."""
    assert proximity_weight(600.0, 500.0, 0.05) == 0.0


def test_exposure_score_max():
    """All max inputs = score 1.0."""
    score = compute_exposure_score(1.0, 1.0, 1.0)
    assert score == 1.0


def test_exposure_score_zero():
    """All zero inputs = score 0.0."""
    score = compute_exposure_score(0.0, 0.0, 0.0)
    assert score == 0.0


def test_exposure_score_mid():
    """Mid inputs produce expected mid score."""
    score = compute_exposure_score(0.5, 0.5, 0.5)
    expected = 0.4 * 0.5 + 0.3 * 0.5 + 0.3 * 0.5
    assert abs(score - expected) < 1e-6


# ── Integration Tests: Replay Mode ─────────────────────────

@pytest.mark.asyncio
async def test_replay_ridgecrest_m52_flags_nearby_assets():
    """
    Replay Ridgecrest M5.2 earthquake.
    Must flag Bakersfield (35.37, -119.02) and Fresno (36.74, -119.79) as exposed.
    Must NOT flag Fargo (46.88, -96.79) as it is 800+ km away.
    """
    config = AppConfig(
        earthquake_min_magnitude=4.5,
        exposure_max_distance_km=500.0,
        exposure_floor_score=0.3,
        distance_decay_factor=0.05,
        replay_mode=True,
    )

    result = await run_replay(config, TEST_ASSETS, "ridgecrest_m52.json")

    # Must be actionable exposure
    assert result.outcome == RunOutcome.ACTIONABLE_EXPOSURE, (
        f"Expected actionable_exposure, got {result.outcome}"
    )

    # Must have findings
    assert len(result.exposure_findings) > 0, "Expected at least one exposure finding"

    # Find affected asset IDs
    affected_ids = {f.asset_id for f in result.exposure_findings}

    # Bakersfield should be affected (very close to Ridgecrest)
    assert "BAK-11" in affected_ids, (
        f"Bakersfield should be affected. Affected: {affected_ids}"
    )

    # Fresno should be affected (moderately close)
    assert "FAT-10" in affected_ids, (
        f"Fresno should be affected. Affected: {affected_ids}"
    )

    # Fargo should NOT be affected (800+ km away)
    assert "FAR-99" not in affected_ids, (
        f"Fargo should NOT be affected (too far). Affected: {affected_ids}"
    )

    # Verify rationale is present and explainable
    for finding in result.exposure_findings:
        assert finding.rationale, "Every finding must have a rationale"
        assert finding.distance_km > 0, "Distance must be positive"
        assert finding.exposure_score >= config.exposure_floor_score, (
            f"Score {finding.exposure_score} below floor {config.exposure_floor_score}"
        )

    # Verify audit trail exists
    assert len(result.audit_log) > 0, "Audit log must not be empty"

    # Verify action payload was drafted
    assert len(result.recommended_actions) > 0, "Expected draft action payload"


@pytest.mark.asyncio
async def test_replay_ridgecrest_m52_deterministic():
    """
    Same replay event must produce identical findings across multiple runs.
    """
    config = AppConfig(
        earthquake_min_magnitude=4.5,
        exposure_max_distance_km=500.0,
        exposure_floor_score=0.3,
        distance_decay_factor=0.05,
        replay_mode=True,
    )

    result1 = await run_replay(config, TEST_ASSETS, "ridgecrest_m52.json")
    result2 = await run_replay(config, TEST_ASSETS, "ridgecrest_m52.json")

    # Same number of findings
    assert len(result1.exposure_findings) == len(result2.exposure_findings)

    # Same asset IDs affected
    ids1 = sorted([f.asset_id for f in result1.exposure_findings])
    ids2 = sorted([f.asset_id for f in result2.exposure_findings])
    assert ids1 == ids2, f"Determinism failed: {ids1} vs {ids2}"

    # Same scores (within floating point tolerance)
    for f1, f2 in zip(
        sorted(result1.exposure_findings, key=lambda x: x.asset_id),
        sorted(result2.exposure_findings, key=lambda x: x.asset_id)
    ):
        assert abs(f1.exposure_score - f2.exposure_score) < 1e-9, (
            f"Score mismatch for {f1.asset_id}: {f1.exposure_score} vs {f2.exposure_score}"
        )


@pytest.mark.asyncio
async def test_replay_below_threshold_stands_down():
    """
    If we raise threshold above M5.2, the same event should stand down.
    """
    config = AppConfig(
        earthquake_min_magnitude=6.0,  # Above the M5.2 replay
        exposure_max_distance_km=500.0,
        exposure_floor_score=0.3,
        distance_decay_factor=0.05,
        replay_mode=True,
    )

    result = await run_replay(config, TEST_ASSETS, "ridgecrest_m52.json")

    assert result.outcome == RunOutcome.STAND_DOWN, (
        f"Expected stand_down (below threshold), got {result.outcome}"
    )
    assert len(result.exposure_findings) == 0, "No findings expected when standing down"


@pytest.mark.asyncio
async def test_replay_high_exposure_floor_stands_down_with_reason():
    """
    If exposure floor is impossibly high, event should stand down with reason.
    """
    config = AppConfig(
        earthquake_min_magnitude=4.5,
        exposure_max_distance_km=500.0,
        exposure_floor_score=0.99,  # Impossibly high
        distance_decay_factor=0.05,
        replay_mode=True,
    )

    result = await run_replay(config, TEST_ASSETS, "ridgecrest_m52.json")

    assert result.outcome == RunOutcome.STAND_DOWN_WITH_REASON, (
        f"Expected stand_down_with_reason, got {result.outcome}"
    )
    assert "exposure floor" in result.outcome_reason.lower(), (
        f"Reason should mention exposure floor: {result.outcome_reason}"
    )


# ── Asset Loading Tests ───────────────────────────────────

def test_load_sample_assets():
    """Verify sample asset register loads and validates."""
    from app import load_assets
    assets = load_assets("sample_assets.csv")
    assert len(assets) == 25, f"Expected 25 assets, got {len(assets)}"

    for asset in assets:
        assert asset.asset_id, "Asset must have ID"
        assert -90 <= asset.latitude <= 90, "Latitude out of range"
        assert -180 <= asset.longitude <= 180, "Longitude out of range"
        assert 1 <= asset.criticality <= 5, "Criticality out of range"


if __name__ == "__main__":
    # Run with: pytest test_replay.py -v
    # Or: python test_replay.py (runs basic tests)
    test_haversine_distance_known_values()
    test_haversine_distance_zero()
    test_proximity_weight_at_zero()
    test_proximity_weight_at_max()
    test_proximity_weight_beyond_max()
    test_exposure_score_max()
    test_exposure_score_zero()
    test_exposure_score_mid()
    print("All unit tests passed.")
