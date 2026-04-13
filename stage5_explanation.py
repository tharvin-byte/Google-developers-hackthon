"""
stage5_explanation.py
---------------------
Stage 5: Decision Engine & Explanation

Uses SHAP (TreeExplainer via a lightweight surrogate model) to identify the
top features driving predictions, compares their importance across demographic
groups, and produces a plain-English explanation of the audit findings.

Produces:
  - Metric-based reasoning (outcome gap, disparity ratio, FPR/FNR)
  - SHAP-based feature influence insight (human-readable, no raw values)
  - Most-affected group identification (selection rate + error rates)
  - Conflict-aware explanations with confidence levels
  - Verdict-specific remediation recommendations

NOTE:
  - We do NOT print raw SHAP values.
  - Only simple, human-readable explanations are produced.
  - A surrogate DecisionTreeClassifier is trained solely for SHAP
    interpretability — it is NOT the model being audited.
"""

from __future__ import annotations

import warnings
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from stage2_metrics import MetricsResult, GroupMetrics
from stage3_scoring import ScoringResult
from stage4_conflict import ConflictResult
from utils import get_logger, fmt_pct, fmt_float

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SENSITIVE_ATTRIBUTE: str = "gender"
FEATURE_COLUMNS: list[str] = ["income", "experience", "education_level"]
TOP_N_FEATURES: int = 3         # number of top features to report
SURROGATE_RANDOM_STATE: int = 42

# Thresholds for flagging notable error-rate differences
ERROR_RATE_THRESHOLD: float = 0.05


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ExplanationResult(NamedTuple):
    """Output of Stage 5.

    Carries all data required to render a professional fairness audit report,
    including SHAP insights, affected-group analysis, conflict-aware
    explanations, confidence level, and actionable recommendations.
    """

    top_features: list[str]                                # ordered list of most important features
    group_feature_importance: dict[str, dict[str, float]]  # {group: {feature: importance}}
    explanation: str                                        # human-readable explanation paragraph
    remediation: list[str]                                 # actionable recommendations
    most_affected_group: str                               # e.g. "Female applicants"
    confidence: str                                        # "High" or "Low"
    confidence_reason: str                                 # reason for the confidence level
    shap_insight: str                                      # human-readable SHAP summary


# ---------------------------------------------------------------------------
# SHAP helpers
# ---------------------------------------------------------------------------

def _train_surrogate(df: pd.DataFrame) -> tuple[DecisionTreeClassifier, pd.DataFrame]:
    """
    Train a lightweight surrogate decision tree on the dataset to enable
    SHAP-based feature importance.

    The surrogate mirrors the *existing* ``y_pred`` column rather than
    ``y_true`` — we want to explain what drives the model's **predictions**,
    not what drives the ground truth.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[DecisionTreeClassifier, pd.DataFrame]
        (trained_model, feature_matrix)
    """
    # Encode education_level as numeric if it is categorical
    df_work = df.copy()

    # Always encode education_level to numeric (handles str, category, etc.)
    if not pd.api.types.is_numeric_dtype(df_work["education_level"]):
        df_work["education_level"] = (
            df_work["education_level"].astype("category").cat.codes
        )

    X = df_work[FEATURE_COLUMNS].copy()
    y = df_work["y_pred"].astype(int)

    surrogate = DecisionTreeClassifier(
        max_depth=5,
        random_state=SURROGATE_RANDOM_STATE,
    )
    surrogate.fit(X, y)
    return surrogate, X


def _compute_shap_importance(
    surrogate: DecisionTreeClassifier,
    X: pd.DataFrame,
    df: pd.DataFrame,
) -> tuple[list[str], dict[str, dict[str, float]]]:
    """
    Use SHAP TreeExplainer to compute per-group feature importances and
    identify the top contributing features.

    Parameters
    ----------
    surrogate : DecisionTreeClassifier
        Fitted surrogate model.
    X : pd.DataFrame
        Feature matrix used for the surrogate.
    df : pd.DataFrame
        Original dataset with the sensitive attribute column.

    Returns
    -------
    tuple[list[str], dict[str, dict[str, float]]]
        (top_features_ordered, {group: {feature: mean_abs_shap}})
    """
    import shap  # imported here to localise the dependency

    # Suppress SHAP's verbose deprecation / feature-perturbation warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        explainer = shap.TreeExplainer(surrogate)
        shap_values = explainer.shap_values(X)

    # shap_values may be:
    #   - a list of arrays (one per class) from older SHAP
    #   - a 3D ndarray (n_samples, n_features, n_classes) from newer SHAP
    #   - a 2D ndarray (n_samples, n_features) for regression / single output
    if isinstance(shap_values, list):
        shap_matrix = np.array(shap_values[1])
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_matrix = shap_values[:, :, 1]  # positive class slice
    else:
        shap_matrix = np.array(shap_values)

    # Global top features (across ALL groups)
    global_importance = np.abs(shap_matrix).mean(axis=0)
    feature_rank = list(np.argsort(global_importance)[::-1])
    top_features = [FEATURE_COLUMNS[int(i)] for i in feature_rank[:TOP_N_FEATURES]]

    # Per-group importance
    groups = df[SENSITIVE_ATTRIBUTE].unique()
    group_importance: dict[str, dict[str, float]] = {}

    for group in groups:
        mask = (df[SENSITIVE_ATTRIBUTE] == group).values
        group_shap = np.abs(shap_matrix[mask]).mean(axis=0)
        group_importance[str(group)] = {
            FEATURE_COLUMNS[i]: round(float(group_shap[i]), 4)
            for i in range(len(FEATURE_COLUMNS))
        }

    return top_features, group_importance


# ---------------------------------------------------------------------------
# Affected-group identification
# ---------------------------------------------------------------------------

def _identify_affected_group(
    per_group: dict[str, GroupMetrics],
) -> tuple[str, str, float, str]:
    """
    Identify which demographic group is most negatively affected by examining
    both selection rates and error rates (FPR / FNR).

    Parameters
    ----------
    per_group : dict[str, GroupMetrics]

    Returns
    -------
    tuple[str, str, float, str]
        (disadvantaged_group, advantaged_group, rate_gap, human_reason)
    """
    sorted_groups = sorted(
        per_group.values(), key=lambda gm: gm.selection_rate
    )
    disadvantaged = sorted_groups[0]
    advantaged = sorted_groups[-1]
    gap = advantaged.selection_rate - disadvantaged.selection_rate

    # Build a human-readable reason incorporating error rates
    reasons: list[str] = ["lower approval rate"]

    fnr_diff = abs(disadvantaged.fnr - advantaged.fnr)
    if fnr_diff > ERROR_RATE_THRESHOLD and disadvantaged.fnr > advantaged.fnr:
        reasons.append("higher false negative rate")

    fpr_diff = abs(disadvantaged.fpr - advantaged.fpr)
    if fpr_diff > ERROR_RATE_THRESHOLD and disadvantaged.fpr > advantaged.fpr:
        reasons.append("higher false positive rate")

    if len(reasons) == 1:
        reason_text = reasons[0]
    elif len(reasons) == 2:
        reason_text = f"{reasons[0]} and {reasons[1]}"
    else:
        reason_text = f"{', '.join(reasons[:-1])}, and {reasons[-1]}"

    return disadvantaged.group, advantaged.group, gap, reason_text


# ---------------------------------------------------------------------------
# SHAP insight builder
# ---------------------------------------------------------------------------

def _build_shap_insight(
    top_features: list[str],
    group_importance: dict[str, dict[str, float]],
) -> str:
    """
    Generate a human-readable SHAP feature-importance summary.

    Compares feature impact across demographic groups and produces a
    natural-language sentence — never exposes raw SHAP values.

    Parameters
    ----------
    top_features : list[str]
        Ordered list of the most influential features.
    group_importance : dict[str, dict[str, float]]
        Per-group mean absolute SHAP values for each feature.

    Returns
    -------
    str
        Human-readable insight paragraph.
    """
    # Build natural-language feature list
    if len(top_features) >= 2:
        feat_str = ", ".join(top_features[:-1]) + f" and {top_features[-1]}"
    else:
        feat_str = top_features[0] if top_features else "unknown features"

    insight = (
        f"The disparity is primarily influenced by {feat_str} "
        f"differences across groups."
    )

    # Cross-group comparison for the most important feature
    if len(group_importance) >= 2 and top_features:
        groups = list(group_importance.keys())
        top_feat = top_features[0]
        vals = {g: group_importance[g].get(top_feat, 0.0) for g in groups}

        max_g = max(vals, key=lambda k: vals[k])
        min_g = min(vals, key=lambda k: vals[k])

        if vals[min_g] > 0:
            ratio = vals[max_g] / vals[min_g]
            if ratio > 1.2:
                insight += (
                    f" The feature '{top_feat}' has a notably stronger influence "
                    f"on predictions for {max_g} applicants compared to "
                    f"{min_g} applicants."
                )

    return insight


# ---------------------------------------------------------------------------
# Explanation builder
# ---------------------------------------------------------------------------

def _build_explanation(
    verdict: str,
    top_features: list[str],
    disadvantaged_group: str,
    rate_gap: float,
    affected_reason: str,
    conflict_detected: bool,
    metrics: MetricsResult,
) -> str:
    """
    Compose a detailed, human-readable explanation paragraph incorporating
    the final verdict, underlying metrics, and conflict status.

    The explanation is grounded in concrete metric values (outcome gap,
    disparity ratio, FPR/FNR differences) so the reader can understand
    *why* the verdict was assigned.

    Parameters
    ----------
    verdict : str
        Final verdict from Stage 3 (FAIR / MODERATE / BIASED).
    top_features : list[str]
        Top SHAP features driving predictions.
    disadvantaged_group : str
        Name of the most-affected group.
    rate_gap : float
        Selection-rate gap between disadvantaged and advantaged groups.
    affected_reason : str
        Human-readable reason why the group is most affected.
    conflict_detected : bool
        Whether Stage 4 flagged conflicting metrics.
    metrics : MetricsResult
        Full Stage 2 output for metric-level detail.

    Returns
    -------
    str
        Multi-sentence explanation paragraph.
    """
    # Verdict → severity description
    severity_map = {
        "FAIR": "minimal bias",
        "MODERATE": "noticeable bias that warrants attention",
        "BIASED": "significant bias requiring immediate action",
    }
    severity = severity_map.get(verdict, "potential bias")

    sentences: list[str] = []

    # 1. Opening severity statement
    sentences.append(f"The model exhibits {severity}.")

    # 2. Metric-based reasoning: outcome gap + disparity ratio
    sentences.append(
        f"The outcome gap between groups is {fmt_pct(rate_gap)}, "
        f"with a disparity ratio of {fmt_float(metrics.disparity_ratio, 2)}."
    )

    # 3. Error-rate analysis
    if metrics.fpr_difference > ERROR_RATE_THRESHOLD:
        sentences.append(
            f"The false positive rate (FPR) differs by "
            f"{fmt_pct(metrics.fpr_difference)} across groups, "
            f"indicating unequal error distribution."
        )

    if metrics.tpr_difference > ERROR_RATE_THRESHOLD:
        sentences.append(
            f"The true positive rate (TPR) gap of "
            f"{fmt_pct(metrics.tpr_difference)} suggests the model's "
            f"predictive accuracy varies across demographics."
        )

    # 4. Most-affected group
    sentences.append(
        f"{disadvantaged_group.capitalize()} applicants are the most affected "
        f"group due to {affected_reason}."
    )

    # 5. Conflict note (required text from spec)
    if conflict_detected:
        sentences.append(
            "Fairness metrics show conflicting signals. While some metrics "
            "suggest fairness, others indicate disparity. This reflects a "
            "known trade-off in fairness evaluation and requires human review."
        )

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Remediation builder
# ---------------------------------------------------------------------------

def _build_remediation(
    verdict: str,
    top_features: list[str],
    disadvantaged_group: str,
) -> list[str]:
    """
    Generate a prioritised list of remediation recommendations tailored
    to the verdict severity.

    Parameters
    ----------
    verdict : str
        Final verdict (FAIR / MODERATE / BIASED).
    top_features : list[str]
        Top SHAP features driving disparity.
    disadvantaged_group : str
        Name of the most-affected group.

    Returns
    -------
    list[str]
        Ordered list of actionable recommendations.
    """
    recs: list[str] = []

    if verdict == "FAIR":
        recs.append(
            "No immediate action required. Continue monitoring."
        )

    elif verdict == "MODERATE":
        recs.append(
            "Review feature influence and decision thresholds. "
            "Consider auditing feature distributions."
        )
        recs.append(
            f"Investigate whether {', '.join(top_features)} contribute to "
            f"disparate outcomes for {disadvantaged_group} applicants."
        )
        recs.append(
            "Consider threshold calibration to reduce the selection-rate gap."
        )

    elif verdict == "BIASED":
        recs.append(
            "Apply mitigation strategies such as rebalancing data, "
            "adjusting thresholds, or reviewing feature selection."
        )
        recs.append(
            f"The features {', '.join(top_features)} are the primary drivers "
            f"of disparity — consider removing or re-weighting them."
        )
        recs.append(
            f"Investigate systemic data collection bias affecting "
            f"'{disadvantaged_group}' applicants."
        )
        recs.append(
            "Model retraining with fairness constraints (e.g. equalised odds "
            "post-processing) is strongly recommended before deployment."
        )

    recs.append(
        "Schedule periodic fairness re-audits as new data is collected."
    )

    return recs


# ---------------------------------------------------------------------------
# Confidence level
# ---------------------------------------------------------------------------

def _determine_confidence(conflict_detected: bool) -> tuple[str, str]:
    """
    Determine confidence level based on metric agreement.

    Parameters
    ----------
    conflict_detected : bool
        Whether Stage 4 detected conflicting metrics.

    Returns
    -------
    tuple[str, str]
        (confidence_label, human_reason)
    """
    if conflict_detected:
        return (
            "Low",
            "Metric disagreement detected — different fairness criteria "
            "point in opposite directions.",
        )
    return (
        "High",
        "All fairness metrics are consistent and agree on the assessment.",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_explanation(
    df: pd.DataFrame,
    metrics: MetricsResult,
    scoring: ScoringResult,
    conflict: ConflictResult,
) -> ExplanationResult:
    """
    Execute Stage 5: Decision Engine & Explanation.

    This stage always runs — including when a conflict was detected in
    Stage 4.  Conflict information is folded into the explanation and
    reflected by a lower confidence level.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset from Stage 0.
    metrics : MetricsResult
        Output of Stage 2.
    scoring : ScoringResult
        Output of Stage 3.
    conflict : ConflictResult
        Output of Stage 4.

    Returns
    -------
    ExplanationResult
    """
    logger.info("Stage 5 — Decision Engine & Explanation started.")

    # 1. Train surrogate and compute SHAP importance
    surrogate, X = _train_surrogate(df)
    top_features, group_importance = _compute_shap_importance(surrogate, X, df)

    logger.info("Top features by SHAP importance: %s", top_features)
    for group, imp in group_importance.items():
        logger.info("  Group '%s': %s", group, imp)

    # 2. Identify the most-affected group
    disadvantaged, advantaged, rate_gap, affected_reason = (
        _identify_affected_group(metrics.per_group)
    )
    most_affected = f"{disadvantaged.capitalize()} applicants"
    logger.info(
        "Most affected group: '%s' | Advantaged: '%s' | Gap: %s | Reason: %s",
        disadvantaged, advantaged, fmt_pct(rate_gap), affected_reason,
    )

    # 3. Generate SHAP insight (human-readable, no raw values)
    shap_insight = _build_shap_insight(top_features, group_importance)
    logger.info("SHAP insight: %s", shap_insight)

    # 4. Determine confidence level
    confidence, confidence_reason = _determine_confidence(
        conflict.conflict_detected
    )
    logger.info("Confidence: %s (%s)", confidence, confidence_reason)

    # 5. Build the main explanation paragraph
    explanation = _build_explanation(
        verdict=scoring.verdict,
        top_features=top_features,
        disadvantaged_group=disadvantaged,
        rate_gap=rate_gap,
        affected_reason=affected_reason,
        conflict_detected=conflict.conflict_detected,
        metrics=metrics,
    )

    # 6. Build remediation checklist
    remediation = _build_remediation(
        verdict=scoring.verdict,
        top_features=top_features,
        disadvantaged_group=disadvantaged,
    )

    logger.info("Stage 5 complete.")

    return ExplanationResult(
        top_features=top_features,
        group_feature_importance=group_importance,
        explanation=explanation,
        remediation=remediation,
        most_affected_group=most_affected,
        confidence=confidence,
        confidence_reason=confidence_reason,
        shap_insight=shap_insight,
    )
