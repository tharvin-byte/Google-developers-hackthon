"""
stage1_rules.py
---------------
Stage 1 Rule-Based Logic (HIGH-Risk Branch)

When Stage 1 fast screening classifies the outcome gap as HIGH, this module
applies additional rule-based checks to provide a preliminary diagnosis
*before* the deep fairness metrics in Stage 2 are computed.

Rules checked:
  1. Base-rate difference using y_true
  2. Group imbalance (sample count skew)
  3. Small group presence (n < 30)
  4. Whether the outcome gap is largely explained by the base-rate alone
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from utils import get_logger, safe_divide, fmt_pct

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSITIVE_ATTRIBUTE: str = "gender"

BASE_RATE_DIFF_THRESHOLD: float = 0.10   # flag if |base_rate_A - base_rate_B| > this
IMBALANCE_RATIO_THRESHOLD: float = 3.0  # flag if majority_group / minority_group > this
SMALL_GROUP_THRESHOLD: int = 30          # flag groups below this count
EXPLAINED_BY_BASE_RATE_TOLERANCE: float = 0.05  # gap vs base-rate within this → explained


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class RulesResult(NamedTuple):
    """Output of the Stage 1 rule-based checks."""

    base_rate_diff: float          # |base_rate_group_A - base_rate_group_B|
    group_imbalanced: bool         # True if sample-count ratio exceeds threshold
    small_group_present: bool      # True if any group has fewer than 30 samples
    gap_explained_by_base_rate: bool  # True if outcome gap is mostly a base-rate effect
    reasoning: str                 # Human-readable diagnostic string


# ---------------------------------------------------------------------------
# Individual rule checks
# ---------------------------------------------------------------------------

def _check_base_rate_difference(df: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """
    Compute the positive base rate (mean y_true) per group and return the
    maximum pairwise absolute difference.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[float, dict[str, float]]
        (max_base_rate_diff, {group_label: base_rate})
    """
    groups = df[SENSITIVE_ATTRIBUTE].unique()
    base_rates: dict[str, float] = {}

    for group in groups:
        mask = df[SENSITIVE_ATTRIBUTE] == group
        base_rates[str(group)] = float(df.loc[mask, "y_true"].mean())

    rates = list(base_rates.values())
    diff = max(rates) - min(rates) if len(rates) > 1 else 0.0
    return diff, base_rates


def _check_group_imbalance(df: pd.DataFrame) -> tuple[bool, dict[str, int]]:
    """
    Check whether the sample counts across groups are severely imbalanced.

    Returns True when the ratio of the largest to the smallest group exceeds
    ``IMBALANCE_RATIO_THRESHOLD``.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[bool, dict[str, int]]
        (is_imbalanced, {group_label: count})
    """
    counts: dict[str, int] = df[SENSITIVE_ATTRIBUTE].value_counts().to_dict()
    count_values = list(counts.values())

    if len(count_values) < 2:
        return False, counts

    ratio = safe_divide(max(count_values), min(count_values), default=1.0)
    imbalanced = ratio > IMBALANCE_RATIO_THRESHOLD

    if imbalanced:
        logger.warning(
            "Group imbalance detected — largest/smallest ratio: %.2f "
            "(threshold: %.1f).",
            ratio, IMBALANCE_RATIO_THRESHOLD,
        )
    return imbalanced, counts


def _check_small_groups(df: pd.DataFrame) -> bool:
    """
    Return True if any demographic group has fewer than ``SMALL_GROUP_THRESHOLD``
    samples.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    bool
    """
    counts: dict[str, int] = df[SENSITIVE_ATTRIBUTE].value_counts().to_dict()
    return any(n < SMALL_GROUP_THRESHOLD for n in counts.values())


def _outcome_gap_explained_by_base_rate(
    outcome_gap: float, base_rate_diff: float
) -> bool:
    """
    Determine whether the prediction outcome gap is largely a reflection of
    the genuine base-rate difference in the ground-truth labels.

    When the two values are within ``EXPLAINED_BY_BASE_RATE_TOLERANCE`` of each
    other the gap is considered "explained" by the base rate (i.e. the model
    may be learning real historical patterns rather than introducing new bias).

    Parameters
    ----------
    outcome_gap : float
        Absolute difference in positive-prediction rates between groups.
    base_rate_diff : float
        Absolute difference in actual positive rates (y_true) between groups.

    Returns
    -------
    bool
    """
    return abs(outcome_gap - base_rate_diff) <= EXPLAINED_BY_BASE_RATE_TOLERANCE


# ---------------------------------------------------------------------------
# Reasoning builder
# ---------------------------------------------------------------------------

def _build_reasoning(
    outcome_gap: float,
    base_rate_diff: float,
    base_rates: dict[str, float],
    group_imbalanced: bool,
    small_group_present: bool,
    gap_explained: bool,
) -> str:
    """
    Compose a human-readable reasoning string summarising the rule-check
    findings.

    Parameters
    ----------
    outcome_gap : float
    base_rate_diff : float
    base_rates : dict[str, float]
    group_imbalanced : bool
    small_group_present : bool
    gap_explained : bool

    Returns
    -------
    str
    """
    lines: list[str] = [
        f"HIGH risk detected. Outcome gap: {fmt_pct(outcome_gap)}."
    ]

    # Base-rate section
    rate_summary = ", ".join(
        f"{g}={fmt_pct(r)}" for g, r in base_rates.items()
    )
    lines.append(
        f"Ground-truth base rates — {rate_summary} "
        f"(difference: {fmt_pct(base_rate_diff)})."
    )

    if gap_explained:
        lines.append(
            "The prediction gap closely mirrors the base-rate difference; "
            "the model may be reflecting historical label imbalance."
        )
    else:
        lines.append(
            "The prediction gap exceeds the base-rate difference, "
            "suggesting the model amplifies existing disparities."
        )

    # Group imbalance section
    if group_imbalanced:
        lines.append(
            "Group imbalance detected: one group has significantly more samples. "
            "Results may be less stable for the minority group."
        )

    # Small group section
    if small_group_present:
        lines.append(
            "At least one group has fewer than 30 samples. "
            "Statistical estimates should be interpreted with caution."
        )

    lines.append(
        "Recommendation: proceed to Stage 2 for rigorous metric computation."
    )

    return " ".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_rules(df: pd.DataFrame, outcome_gap: float) -> RulesResult:
    """
    Execute Stage 1 rule-based checks for HIGH-risk cases.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from Stage 0.
    outcome_gap : float
        Outcome gap computed in Stage 1 screening.

    Returns
    -------
    RulesResult
    """
    logger.info("Stage 1 Rules - HIGH risk path entered.")

    base_rate_diff, base_rates = _check_base_rate_difference(df)
    group_imbalanced, _counts = _check_group_imbalance(df)
    small_group_present = _check_small_groups(df)
    gap_explained = _outcome_gap_explained_by_base_rate(outcome_gap, base_rate_diff)

    logger.info("Base-rate difference: %s", fmt_pct(base_rate_diff))
    logger.info("Group imbalanced: %s", group_imbalanced)
    logger.info("Small group present: %s", small_group_present)
    logger.info("Gap explained by base rate: %s", gap_explained)

    reasoning = _build_reasoning(
        outcome_gap=outcome_gap,
        base_rate_diff=base_rate_diff,
        base_rates=base_rates,
        group_imbalanced=group_imbalanced,
        small_group_present=small_group_present,
        gap_explained=gap_explained,
    )

    logger.info("Stage 1 Rules complete.")
    return RulesResult(
        base_rate_diff=base_rate_diff,
        group_imbalanced=group_imbalanced,
        small_group_present=small_group_present,
        gap_explained_by_base_rate=gap_explained,
        reasoning=reasoning,
    )
