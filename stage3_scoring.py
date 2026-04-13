"""
stage3_scoring.py
-----------------
Stage 3: Weighted Fairness Scoring

Normalises each fairness metric to [0, 1] (where 1 = perfectly fair) and
combines them into a single composite fairness score using weighted averaging.

Key design decisions:
  - Normalization bounds are anchored to practical fairness thresholds
    (EEOC 80% rule ≈ 20% gap), NOT to the theoretical maximum of 1.0.
    This ensures that a 10% outcome gap scores ~0.50 (moderate), not 0.90.
  - The outcome gap from Stage 1 screening is included as a scored metric
    with the highest weight, ensuring stage consistency.
  - Confidence shrinkage pulls small-group scores toward 0.5 (neutral).

Score interpretation:
  > 0.75  → FAIR       (minimal disparity, no action needed)
  0.40–0.75 → MODERATE (review recommended before deployment)
  < 0.40  → BIASED    (mitigation required)
"""

from __future__ import annotations

from typing import NamedTuple

from stage2_metrics import MetricsResult
from utils import get_logger, normalize_metric, clamp, fmt_float

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Weights (must sum to 1.0)
# ---------------------------------------------------------------------------

WEIGHTS: dict[str, float] = {
    "outcome_gap":        0.30,   # highest priority — direct Stage 1 signal
    "demographic_parity": 0.25,   # EEOC-aligned selection rate parity
    "disparity_ratio":    0.10,   # 4/5-rule proxy
    "tpr_difference":     0.20,   # equal opportunity (recall parity)
    "fpr_difference":     0.15,   # predictive equity (false alarm parity)
}

# Normalization bounds — anchored to practical fairness thresholds
# A 20% gap is the EEOC "adverse impact" trigger; we treat it as worst-case.
WORST_DIFF: float = 0.20         # worst-case for difference metrics
WORST_FPR_DIFF: float = 0.30    # FPR can have wider natural variance

# Thresholds for verdict labels
THRESHOLD_FAIR: float = 0.75
THRESHOLD_BIASED: float = 0.40

# Confidence shrinkage applies when any group n < this
SMALL_GROUP_N: int = 50
NEUTRAL_SCORE: float = 0.50   # shrink toward this for small groups


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ScoringResult(NamedTuple):
    """Output of Stage 3 fair-scoring."""

    normalised_scores: dict[str, float]  # per-metric normalised score
    weighted_score: float                # composite [0, 1]
    shrinkage_applied: bool              # True if any group n < 50
    verdict: str                         # FAIR | MODERATE | BIASED


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def _normalise_outcome_gap(value: float) -> float:
    """
    Normalise the outcome gap (positive-prediction rate difference).

    Best case: 0.0 (no gap) → 1.0
    Worst case: WORST_DIFF (EEOC threshold) → 0.0

    A 10% gap normalises to ~0.50 (moderate concern).

    Parameters
    ----------
    value : float
        Absolute outcome gap from Stage 1 screening.

    Returns
    -------
    float
    """
    return normalize_metric(value, worst=WORST_DIFF, best=0.0)


def _normalise_demographic_parity(value: float) -> float:
    """
    Normalise Demographic Parity Difference.

    Best case: 0.0 (perfect parity) → 1.0
    Worst case: WORST_DIFF (EEOC threshold) → 0.0

    Parameters
    ----------
    value : float

    Returns
    -------
    float
    """
    return normalize_metric(value, worst=WORST_DIFF, best=0.0)


def _normalise_disparity_ratio(value: float) -> float:
    """
    Normalise Disparity Ratio.

    Best case: 1.0 (equal rates) → 1.0
    Worst case: 0.0 (total disparity) → 0.0

    Parameters
    ----------
    value : float

    Returns
    -------
    float
    """
    return normalize_metric(value, worst=0.0, best=1.0)


def _normalise_tpr_difference(value: float) -> float:
    """
    Normalise TPR Difference (equal opportunity gap).

    Best case: 0.0 → 1.0 | Worst case: WORST_DIFF → 0.0

    Parameters
    ----------
    value : float

    Returns
    -------
    float
    """
    return normalize_metric(value, worst=WORST_DIFF, best=0.0)


def _normalise_fpr_difference(value: float) -> float:
    """
    Normalise FPR Difference (predictive equity gap).

    Best case: 0.0 → 1.0 | Worst case: WORST_FPR_DIFF → 0.0
    FPR uses a wider bound because false-positive rates have higher natural
    variance across groups.

    Parameters
    ----------
    value : float

    Returns
    -------
    float
    """
    return normalize_metric(value, worst=WORST_FPR_DIFF, best=0.0)


# ---------------------------------------------------------------------------
# Confidence shrinkage
# ---------------------------------------------------------------------------

def _apply_shrinkage(
    score: float,
    min_group_n: int,
) -> tuple[float, bool]:
    """
    Apply confidence shrinkage for small-sample groups.

    When *min_group_n* < ``SMALL_GROUP_N``, the composite score is pulled
    toward ``NEUTRAL_SCORE`` proportionally to sample scarcity.

    Parameters
    ----------
    score : float
        Raw composite fairness score.
    min_group_n : int
        Size of the smallest demographic group in the dataset.

    Returns
    -------
    tuple[float, bool]
        (adjusted_score, shrinkage_applied)
    """
    if min_group_n >= SMALL_GROUP_N:
        return score, False

    # Linear interpolation: shrink proportionally based on how far below 50 we are
    alpha = min_group_n / SMALL_GROUP_N          # 0 → 1 as n → 50
    shrunken = alpha * score + (1.0 - alpha) * NEUTRAL_SCORE
    logger.warning(
        "Confidence shrinkage applied (min group n=%d < %d). "
        "Score adjusted from %.4f to %.4f.",
        min_group_n, SMALL_GROUP_N, score, shrunken,
    )
    return clamp(shrunken), True


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------

def _assign_verdict(score: float) -> str:
    """
    Map a composite fairness score to a categorical verdict.

    Parameters
    ----------
    score : float
        Composite normalised fairness score in [0, 1].

    Returns
    -------
    str
        ``'FAIR'``, ``'MODERATE'``, or ``'BIASED'``.
    """
    if score > THRESHOLD_FAIR:
        return "FAIR"
    if score > THRESHOLD_BIASED:
        return "MODERATE"
    return "BIASED"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_scoring(metrics: MetricsResult, outcome_gap: float = 0.0) -> ScoringResult:
    """
    Execute Stage 3: Weighted Fairness Scoring.

    Parameters
    ----------
    metrics : MetricsResult
        Output of Stage 2 metric computation.
    outcome_gap : float
        Outcome gap from Stage 1 screening (absolute difference in
        positive-prediction rates between groups).

    Returns
    -------
    ScoringResult
    """
    logger.info("Stage 3 - Weighted Fairness Scoring started.")

    # Normalise each metric
    normalised: dict[str, float] = {
        "outcome_gap": _normalise_outcome_gap(outcome_gap),
        "demographic_parity": _normalise_demographic_parity(
            metrics.demographic_parity_diff
        ),
        "disparity_ratio": _normalise_disparity_ratio(metrics.disparity_ratio),
        "tpr_difference": _normalise_tpr_difference(metrics.tpr_difference),
        "fpr_difference": _normalise_fpr_difference(metrics.fpr_difference),
    }

    for metric_name, norm_val in normalised.items():
        logger.info("  %s -> normalised score: %s", metric_name, fmt_float(norm_val))

    # Weighted sum
    assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"
    raw_score = sum(normalised[k] * WEIGHTS[k] for k in WEIGHTS)
    raw_score = clamp(raw_score)

    # Identify minimum group size for shrinkage check
    min_group_n = min(gm.n for gm in metrics.per_group.values())
    composite_score, shrinkage_applied = _apply_shrinkage(raw_score, min_group_n)

    verdict = _assign_verdict(composite_score)

    logger.info("Raw composite score    : %s", fmt_float(raw_score))
    logger.info("Final fairness score   : %s", fmt_float(composite_score))
    logger.info("Verdict                : %s", verdict)
    logger.info("Stage 3 complete.")

    return ScoringResult(
        normalised_scores=normalised,
        weighted_score=composite_score,
        shrinkage_applied=shrinkage_applied,
        verdict=verdict,
    )
