"""
stage2_metrics.py
-----------------
Stage 2: Core Fairness Metrics

Implements fairness metrics manually (no Fairlearn dependency) across
demographic groups defined by the sensitive attribute.

Metrics computed per group:
  - Selection rate (positive prediction rate)
  - True Positive Rate (TPR / recall / sensitivity)
  - False Positive Rate (FPR)
  - False Negative Rate (FNR)
  - Accuracy

Derived fairness metrics:
  - Demographic Parity Difference (max selection rate − min selection rate)
  - Disparity Ratio               (min selection rate / max selection rate)
  - TPR Difference                (max TPR − min TPR)
  - FPR Difference                (max FPR − min FPR)
"""

from __future__ import annotations

from typing import NamedTuple

import pandas as pd

from utils import get_logger, safe_divide, fmt_pct, fmt_float

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSITIVE_ATTRIBUTE: str = "gender"

# EEOC-inspired 80% (4/5) rule threshold for disparity ratio
EEOC_RATIO_THRESHOLD: float = 0.80


# ---------------------------------------------------------------------------
# Per-group metric container
# ---------------------------------------------------------------------------

class GroupMetrics(NamedTuple):
    """All per-group classification-based fairness metrics."""

    group: str
    n: int                  # sample count
    selection_rate: float   # P(y_pred=1)
    tpr: float              # P(y_pred=1 | y_true=1)
    fpr: float              # P(y_pred=1 | y_true=0)
    fnr: float              # P(y_pred=0 | y_true=1)
    accuracy: float         # (TP + TN) / N


# ---------------------------------------------------------------------------
# Aggregate result container
# ---------------------------------------------------------------------------

class MetricsResult(NamedTuple):
    """All Stage 2 output."""

    per_group: dict[str, GroupMetrics]
    demographic_parity_diff: float   # max − min selection rate
    disparity_ratio: float           # min / max selection rate
    tpr_difference: float            # max − min TPR
    fpr_difference: float            # max − min FPR
    eeoc_flag: bool                  # True if disparity_ratio < 0.80


# ---------------------------------------------------------------------------
# Per-group computation
# ---------------------------------------------------------------------------

def _compute_confusion_components(
    y_true: pd.Series, y_pred: pd.Series
) -> tuple[int, int, int, int]:
    """
    Compute TP, FP, FN, TN from ground-truth and predicted label arrays.

    Parameters
    ----------
    y_true : pd.Series
        Actual binary labels (0 / 1).
    y_pred : pd.Series
        Predicted binary labels (0 / 1).

    Returns
    -------
    tuple[int, int, int, int]
        (TP, FP, FN, TN)
    """
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    return tp, fp, fn, tn


def _compute_group_metrics(df: pd.DataFrame, group_label: str) -> GroupMetrics:
    """
    Compute all classification-based fairness metrics for a single demographic
    group.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset — will be filtered to the specified group.
    group_label : str
        Value of ``SENSITIVE_ATTRIBUTE`` identifying this group.

    Returns
    -------
    GroupMetrics
    """
    mask = df[SENSITIVE_ATTRIBUTE] == group_label
    sub = df[mask]

    y_true: pd.Series = sub["y_true"].astype(int)
    y_pred: pd.Series = sub["y_pred"].astype(int)
    n = len(sub)

    tp, fp, fn, tn = _compute_confusion_components(y_true, y_pred)

    selection_rate = safe_divide(tp + fp, n)
    tpr = safe_divide(tp, tp + fn)         # sensitivity / recall
    fpr = safe_divide(fp, fp + tn)         # fall-out
    fnr = safe_divide(fn, fn + tp)         # miss rate
    accuracy = safe_divide(tp + tn, n)

    return GroupMetrics(
        group=str(group_label),
        n=n,
        selection_rate=round(selection_rate, 6),
        tpr=round(tpr, 6),
        fpr=round(fpr, 6),
        fnr=round(fnr, 6),
        accuracy=round(accuracy, 6),
    )


# ---------------------------------------------------------------------------
# Aggregate derivations
# ---------------------------------------------------------------------------

def _derive_aggregate_metrics(
    per_group: dict[str, GroupMetrics],
) -> tuple[float, float, float, float, bool]:
    """
    Derive cross-group aggregate fairness metrics from per-group statistics.

    Parameters
    ----------
    per_group : dict[str, GroupMetrics]

    Returns
    -------
    tuple
        (demographic_parity_diff, disparity_ratio, tpr_diff, fpr_diff, eeoc_flag)
    """
    selection_rates = [m.selection_rate for m in per_group.values()]
    tprs = [m.tpr for m in per_group.values()]
    fprs = [m.fpr for m in per_group.values()]

    dem_parity_diff = max(selection_rates) - min(selection_rates)
    disparity_ratio = safe_divide(min(selection_rates), max(selection_rates), default=0.0)
    tpr_diff = max(tprs) - min(tprs)
    fpr_diff = max(fprs) - min(fprs)
    eeoc_flag = disparity_ratio < EEOC_RATIO_THRESHOLD

    return dem_parity_diff, disparity_ratio, tpr_diff, fpr_diff, eeoc_flag


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_metrics(df: pd.DataFrame) -> MetricsResult:
    """
    Execute Stage 2: Core Fairness Metrics.

    Computes per-group and aggregate fairness metrics without any external
    fairness library.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from Stage 0.

    Returns
    -------
    MetricsResult
    """
    logger.info("Stage 2 - Core Fairness Metrics started.")

    groups = df[SENSITIVE_ATTRIBUTE].unique()
    per_group: dict[str, GroupMetrics] = {}

    for group in groups:
        gm = _compute_group_metrics(df, group)
        per_group[str(group)] = gm
        logger.info(
            "Group '%s' (n=%d) — Selection Rate: %s | TPR: %s | FPR: %s | "
            "FNR: %s | Accuracy: %s",
            gm.group, gm.n,
            fmt_pct(gm.selection_rate),
            fmt_pct(gm.tpr),
            fmt_pct(gm.fpr),
            fmt_pct(gm.fnr),
            fmt_pct(gm.accuracy),
        )

    dem_parity_diff, disparity_ratio, tpr_diff, fpr_diff, eeoc_flag = (
        _derive_aggregate_metrics(per_group)
    )

    logger.info("Demographic Parity Difference : %s", fmt_pct(dem_parity_diff))
    logger.info("Disparity Ratio               : %s", fmt_float(disparity_ratio))
    logger.info("TPR Difference                : %s", fmt_pct(tpr_diff))
    logger.info("FPR Difference                : %s", fmt_pct(fpr_diff))
    logger.info(
        "EEOC 80%% Rule                : %s",
        "VIOLATED" if eeoc_flag else "SATISFIED",
    )

    logger.info("Stage 2 complete.")

    return MetricsResult(
        per_group=per_group,
        demographic_parity_diff=dem_parity_diff,
        disparity_ratio=disparity_ratio,
        tpr_difference=tpr_diff,
        fpr_difference=fpr_diff,
        eeoc_flag=eeoc_flag,
    )
