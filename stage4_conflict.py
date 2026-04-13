"""
stage4_conflict.py
------------------
Stage 4: Conflict Detection

Detects when individual fairness metrics *contradict* each other — e.g.
demographic parity says BIASED but equalised odds says FAIR.  This is a
manifestation of the Chouldechova (2017) impossibility result: it is
mathematically impossible to satisfy all fairness criteria simultaneously
when base rates differ between groups.

Decision logic
--------------
If normalised metric scores disagree by more than ``CONFLICT_THRESHOLD``
(default 0.25), a conflict is flagged and the pipeline recommends human
review rather than an automated verdict.
"""

from __future__ import annotations

from typing import NamedTuple

from stage3_scoring import ScoringResult
from utils import get_logger, fmt_float

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONFLICT_THRESHOLD: float = 0.40   # max allowed pairwise normalised-score spread
BASE_RATE_CONFLICT_PP: float = 0.15  # percentage-point base-rate gap that makes
                                      # impossibility realistic


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ConflictResult(NamedTuple):
    """Output of Stage 4 conflict detection."""

    conflict_detected: bool         # True → metrics fundamentally disagree
    max_score_spread: float         # largest pairwise distance between normalised scores
    conflicting_pair: tuple[str, str] | None  # metric names that disagree the most
    recommendation: str             # Human-readable summary


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def _find_max_spread(
    normalised_scores: dict[str, float],
) -> tuple[float, tuple[str, str] | None]:
    """
    Compute the largest pairwise difference among normalised metric scores.

    Parameters
    ----------
    normalised_scores : dict[str, float]
        Normalised fairness scores from Stage 3 (each in [0, 1]).

    Returns
    -------
    tuple[float, tuple[str, str] | None]
        (max_spread, (metric_high, metric_low)) or (0.0, None) when fewer
        than two metrics exist.
    """
    keys = list(normalised_scores.keys())
    if len(keys) < 2:
        return 0.0, None

    max_spread = 0.0
    pair: tuple[str, str] | None = None

    for i, k1 in enumerate(keys):
        for k2 in keys[i + 1:]:
            spread = abs(normalised_scores[k1] - normalised_scores[k2])
            if spread > max_spread:
                max_spread = spread
                pair = (k1, k2)

    return round(max_spread, 4), pair


def _build_recommendation(
    conflict_detected: bool,
    max_spread: float,
    conflicting_pair: tuple[str, str] | None,
) -> str:
    """
    Return the plain-English recommendation based on conflict status.

    Parameters
    ----------
    conflict_detected : bool
    max_spread : float
    conflicting_pair : tuple[str, str] | None

    Returns
    -------
    str
    """
    if not conflict_detected:
        return (
            "No significant conflict between fairness metrics. "
            "All normalised scores are within acceptable agreement "
            f"(max spread: {fmt_float(max_spread)}). "
            "Proceeding to Stage 5 for explanation."
        )

    m1, m2 = conflicting_pair  # type: ignore[misc]
    return (
        f"CONFLICT DETECTED: '{m1}' and '{m2}' disagree by "
        f"{fmt_float(max_spread)} (threshold: {fmt_float(CONFLICT_THRESHOLD)}). "
        "This may reflect the Chouldechova (2017) impossibility theorem — "
        "not all group-fairness criteria can be satisfied simultaneously "
        "when base rates differ. Human review is strongly recommended before "
        "an automated verdict is issued."
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_conflict_detection(scoring: ScoringResult) -> ConflictResult:
    """
    Execute Stage 4: Conflict Detection.

    Examines whether the normalised metric scores from Stage 3 are
    self-consistent.  A large spread indicates that different fairness
    definitions point in opposite directions — a common consequence of
    unequal base rates.

    Parameters
    ----------
    scoring : ScoringResult
        Output of Stage 3.

    Returns
    -------
    ConflictResult
    """
    logger.info("Stage 4 — Conflict Detection started.")

    max_spread, pair = _find_max_spread(scoring.normalised_scores)
    conflict_detected = max_spread > CONFLICT_THRESHOLD

    recommendation = _build_recommendation(conflict_detected, max_spread, pair)

    if conflict_detected:
        logger.warning("Conflict detected (spread=%.4f > %.2f).", max_spread, CONFLICT_THRESHOLD)
        if pair:
            logger.warning("  Conflicting metrics: %s vs %s", pair[0], pair[1])
    else:
        logger.info("No conflict detected (spread=%.4f).", max_spread)

    logger.info("Stage 4 complete.")

    return ConflictResult(
        conflict_detected=conflict_detected,
        max_score_spread=max_spread,
        conflicting_pair=pair,
        recommendation=recommendation,
    )
