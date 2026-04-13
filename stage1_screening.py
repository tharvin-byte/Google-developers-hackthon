"""
stage1_screening.py
-------------------
Stage 1: Fast Screening

Performs lightweight, model-agnostic screening to produce a preliminary risk
level (LOW / MODERATE / HIGH) without computing deep fairness metrics.

Checks:
  • Outcome gap  — difference in mean y_pred between groups
  • Feature distribution delta  — distribution divergence for numeric features
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
import pandas as pd

from utils import get_logger, fmt_pct, safe_divide

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSITIVE_ATTRIBUTE: str = "gender"
NUMERIC_FEATURES: list[str] = ["income", "experience"]

# Risk thresholds (based on outcome gap magnitude)
THRESHOLD_LOW: float = 0.05       # gap < this → LOW
THRESHOLD_HIGH: float = 0.15      # gap > this → HIGH
# 0.05 – 0.15 → MODERATE

RISK_LOW = "LOW"
RISK_MODERATE = "MODERATE"
RISK_HIGH = "HIGH"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ScreeningResult(NamedTuple):
    """Structured output from Stage 1 fast screening."""

    outcome_gap: float           # absolute difference in positive prediction rates
    risk_level: str              # LOW | MODERATE | HIGH
    group_means: dict[str, float]  # mean y_pred per group
    feature_deltas: dict[str, float]  # mean feature delta per feature
    distribution_skew_detected: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_outcome_gap(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """
    Compute the positive-prediction rate per demographic group and the
    maximum pairwise gap.

    Parameters
    ----------
    df : pd.DataFrame
        Validated dataset containing ``gender`` and ``y_pred`` columns.

    Returns
    -------
    tuple[float, dict[str, float]]
        (max_gap, {group_label: mean_y_pred})
    """
    groups = df[SENSITIVE_ATTRIBUTE].unique()
    group_means: dict[str, float] = {}

    for group in groups:
        mask = df[SENSITIVE_ATTRIBUTE] == group
        group_means[str(group)] = float(df.loc[mask, "y_pred"].mean())

    rates = list(group_means.values())
    max_gap = max(rates) - min(rates) if len(rates) > 1 else 0.0
    return max_gap, group_means


def _compute_feature_deltas(df: pd.DataFrame) -> tuple[dict[str, float], bool]:
    """
    Compute the absolute normalised mean difference per numeric feature between
    the two most prominent demographic groups.

    A *strong skew* is flagged when the normalised delta exceeds 0.20 for any
    feature.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[dict[str, float], bool]
        (feature_delta_per_feature, strong_skew_detected)
    """
    groups = df[SENSITIVE_ATTRIBUTE].unique()
    if len(groups) < 2:
        return {feat: 0.0 for feat in NUMERIC_FEATURES}, False

    g0, g1 = groups[0], groups[1]
    mask0 = df[SENSITIVE_ATTRIBUTE] == g0
    mask1 = df[SENSITIVE_ATTRIBUTE] == g1

    deltas: dict[str, float] = {}
    skew_detected = False

    for feat in NUMERIC_FEATURES:
        if feat not in df.columns:
            deltas[feat] = 0.0
            continue

        mean0 = df.loc[mask0, feat].mean()
        mean1 = df.loc[mask1, feat].mean()
        std_pool = df[feat].std() or 1.0

        normalised_delta = abs(mean0 - mean1) / std_pool
        deltas[feat] = round(float(normalised_delta), 4)

        if normalised_delta > 0.20:
            skew_detected = True
            logger.warning(
                "Strong distribution skew detected in '%s' "
                "(normalised delta=%.4f).",
                feat, normalised_delta,
            )

    return deltas, skew_detected


def _classify_risk(outcome_gap: float) -> str:
    """
    Map a numeric outcome gap to a categorical risk level.

    Parameters
    ----------
    outcome_gap : float
        Absolute difference in positive-prediction rates between groups.

    Returns
    -------
    str
        One of ``'LOW'``, ``'MODERATE'``, or ``'HIGH'``.
    """
    if outcome_gap < THRESHOLD_LOW:
        return RISK_LOW
    if outcome_gap > THRESHOLD_HIGH:
        return RISK_HIGH
    return RISK_MODERATE


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_screening(df: pd.DataFrame) -> ScreeningResult:
    """
    Execute Stage 1: Fast Screening.

    Computes outcome gap, feature distribution deltas, and classifies the
    preliminary risk level.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from Stage 0.

    Returns
    -------
    ScreeningResult
    """
    logger.info("Stage 1 - Fast Screening started.")

    outcome_gap, group_means = _compute_outcome_gap(df)
    feature_deltas, skew_detected = _compute_feature_deltas(df)
    risk_level = _classify_risk(outcome_gap)

    logger.info(
        "Outcome gap: %s | Risk level: %s",
        fmt_pct(outcome_gap), risk_level,
    )
    for group, rate in group_means.items():
        logger.info("  Group '%s' -> mean prediction rate: %s", group, fmt_pct(rate))
    for feat, delta in feature_deltas.items():
        logger.info("  Feature '%s' normalised delta: %.4f", feat, delta)

    logger.info(
        "Distribution skew: %s", "DETECTED" if skew_detected else "none"
    )
    logger.info("Stage 1 complete - preliminary risk level: %s", risk_level)

    return ScreeningResult(
        outcome_gap=outcome_gap,
        risk_level=risk_level,
        group_means=group_means,
        feature_deltas=feature_deltas,
        distribution_skew_detected=skew_detected,
    )
