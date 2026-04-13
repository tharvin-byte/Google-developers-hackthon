"""
main.py
-------
FairSight - AI Fairness Audit Pipeline Orchestrator

Loads ``dataset.csv`` and runs every stage in order:

  Stage 0 -> Data Validation
  Stage 1 -> Fast Screening  (+Rule-based logic for HIGH risk)
  Stage 2 -> Core Fairness Metrics
  Stage 3 -> Weighted Scoring + Confidence Layer
  Stage 4 -> Conflict Detection
  Stage 5 -> Decision Engine & Explanation  (always runs)

Prints a clean, professional audit report to stdout.
"""

from __future__ import annotations

import io
import sys
import textwrap

# Force UTF-8 output on Windows consoles to avoid cp1252 encoding errors
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding="utf-8", errors="replace"
    )

from stage0_validation import run_validation
from stage1_screening import run_screening
from stage1_rules import run_rules
from stage2_metrics import run_metrics
from stage3_scoring import run_scoring
from stage4_conflict import run_conflict_detection
from stage5_explanation import run_explanation
from utils import section_banner, fmt_pct, fmt_float


# ---------------------------------------------------------------------------
# Report formatting constants
# ---------------------------------------------------------------------------

_REPORT_WIDTH: int = 62
_DIVIDER: str = "=" * _REPORT_WIDTH
_THIN_DIVIDER: str = "-" * _REPORT_WIDTH


# ---------------------------------------------------------------------------
# Report formatting helpers
# ---------------------------------------------------------------------------

def _header(title: str) -> str:
    """Return a boxed header banner for the report."""
    padded = f"  {title}  ".center(_REPORT_WIDTH)
    return f"\n{_DIVIDER}\n{padded}\n{_DIVIDER}"


def _section(title: str) -> str:
    """Return a section heading with thin dividers."""
    label = f"-- {title} "
    return f"\n{label}{'-' * max(0, _REPORT_WIDTH - len(label))}"


def _wrap(text: str, indent: int = 3) -> str:
    """Word-wrap text with a leading indent for report sections."""
    prefix = " " * indent
    return textwrap.fill(
        text,
        width=_REPORT_WIDTH - indent,
        initial_indent=prefix,
        subsequent_indent=prefix,
    )


def _verdict_indicator(verdict: str) -> str:
    """Return a visual indicator for the verdict severity."""
    icons = {
        "FAIR": "[OK]  FAIR",
        "MODERATE": "[!!]  MODERATE",
        "BIASED": "[XX]  BIASED",
    }
    return icons.get(verdict, verdict)


def _risk_indicator(risk: str) -> str:
    """Return a visual indicator for the risk level."""
    icons = {
        "LOW": "[OK]  LOW",
        "MODERATE": "[!!]  MODERATE",
        "HIGH": "[XX]  HIGH",
    }
    return icons.get(risk, risk)


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def _print_report(
    outcome_gap: float,
    risk_level: str,
    rules_reasoning: str | None,
    fairness_score: float,
    verdict: str,
    conflict_detected: bool,
    conflict_recommendation: str,
    conflicting_pair: tuple[str, str] | None,
    most_affected_group: str,
    confidence: str,
    confidence_reason: str,
    shap_insight: str,
    explanation: str,
    remediation: list[str],
) -> None:
    """
    Print the final audit report in a professional, human-readable format.

    Parameters
    ----------
    outcome_gap : float
        Absolute selection-rate gap from Stage 1.
    risk_level : str
        Preliminary risk level from Stage 1.
    rules_reasoning : str | None
        Rule-based reasoning from Stage 1 (HIGH risk only).
    fairness_score : float
        Composite weighted fairness score from Stage 3.
    verdict : str
        Final verdict (FAIR / MODERATE / BIASED).
    conflict_detected : bool
        Whether Stage 4 detected conflicting metrics.
    conflict_recommendation : str
        Human-readable conflict summary from Stage 4.
    conflicting_pair : tuple[str, str] | None
        The pair of metrics that disagree the most.
    most_affected_group : str
        Human-readable label for the most-affected group.
    confidence : str
        Confidence level (High / Low).
    confidence_reason : str
        Reason for the confidence level.
    shap_insight : str
        Human-readable SHAP feature insight.
    explanation : str
        Full explanation paragraph from Stage 5.
    remediation : list[str]
        Ordered list of recommendations.
    """
    # -- Title --
    print(_header("FAIRSIGHT - FAIRNESS AUDIT REPORT"))

    # -- Overview --
    print(_section("Overview"))
    print(f"   Outcome Gap      : {fmt_pct(outcome_gap)}")
    print(f"   Risk Level       : {_risk_indicator(risk_level)}")
    print(f"   Fairness Score   : {fmt_float(fairness_score)} / 1.00")
    print(f"   Final Verdict    : {_verdict_indicator(verdict)}")
    print(f"   Confidence       : {confidence}")

    # ── Stage 1 Rule-Based Analysis (HIGH risk only) ──
    if rules_reasoning:
        print(_section("Rule-Based Analysis (HIGH Risk)"))
        print(_wrap(rules_reasoning))

    # ── Conflict Analysis ──
    print(_section("Conflict Analysis"))
    print(f"   Conflict Detected : {'YES' if conflict_detected else 'NO'}")
    if conflict_detected and conflicting_pair:
        print(f"   Conflicting Pair  : {conflicting_pair[0]} vs {conflicting_pair[1]}")
    print(f"   Confidence Reason : {confidence_reason}")
    if conflict_detected:
        print(_wrap(conflict_recommendation))

    # ── Most Affected Group ──
    print(_section("Most Affected Group"))
    print(f"   {most_affected_group}")

    # ── SHAP Feature Analysis ──
    print(_section("SHAP Feature Analysis"))
    print(_wrap(shap_insight))

    # ── Explanation ──
    print(_section("Explanation"))
    print(_wrap(explanation))

    # ── Recommendations ──
    print(_section("Recommendations"))
    for i, rec in enumerate(remediation, 1):
        # Indent wrapped lines under the number
        prefix = f"   {i}. "
        subsequent = " " * len(prefix)
        wrapped = textwrap.fill(
            rec,
            width=_REPORT_WIDTH - len(prefix),
            initial_indent=prefix,
            subsequent_indent=subsequent,
        )
        print(wrapped)

    # ── Footer ──
    print(f"\n{_THIN_DIVIDER}")
    print("   Generated by FairSight — AI Fairness Audit Pipeline")
    print(f"{_THIN_DIVIDER}")
    print(_header("END OF REPORT"))


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Orchestrate the full FairSight fairness audit pipeline.
    """
    print(section_banner("FairSight - AI Fairness Audit Pipeline"))

    # ── Stage 0: Data Validation ──
    print(section_banner("Stage 0 - Data Validation"))
    validation = run_validation("dataset.csv")
    if not validation.passed:
        print("[FAIL] Stage 0 validation failed. Aborting pipeline.")
        sys.exit(1)
    df = validation.dataframe

    # ── Stage 1: Fast Screening ──
    print(section_banner("Stage 1 - Fast Screening"))
    screening = run_screening(df)

    rules_reasoning: str | None = None
    if screening.risk_level == "HIGH":
        print(section_banner("Stage 1 - Rule-Based Logic (HIGH Risk)"))
        rules_result = run_rules(df, screening.outcome_gap)
        rules_reasoning = rules_result.reasoning

    # ── Stage 2: Core Fairness Metrics ──
    print(section_banner("Stage 2 - Core Fairness Metrics"))
    metrics = run_metrics(df)

    # ── Stage 3: Weighted Scoring ──
    print(section_banner("Stage 3 - Weighted Scoring"))
    scoring = run_scoring(metrics, outcome_gap=screening.outcome_gap)

    # ── Stage 4: Conflict Detection ──
    print(section_banner("Stage 4 - Conflict Detection"))
    conflict = run_conflict_detection(scoring)

    # ── Stage 5: Decision Engine & Explanation (always runs) ──
    print(section_banner("Stage 5 - Decision Engine & Explanation"))
    if conflict.conflict_detected:
        print("   [!] Conflict detected — Stage 5 will include conflict analysis.")
    explanation_result = run_explanation(df, metrics, scoring, conflict)

    final_verdict = scoring.verdict

    # ── Final Report ──
    _print_report(
        outcome_gap=screening.outcome_gap,
        risk_level=screening.risk_level,
        rules_reasoning=rules_reasoning,
        fairness_score=scoring.weighted_score,
        verdict=final_verdict,
        conflict_detected=conflict.conflict_detected,
        conflict_recommendation=conflict.recommendation,
        conflicting_pair=conflict.conflicting_pair,
        most_affected_group=explanation_result.most_affected_group,
        confidence=explanation_result.confidence,
        confidence_reason=explanation_result.confidence_reason,
        shap_insight=explanation_result.shap_insight,
        explanation=explanation_result.explanation,
        remediation=explanation_result.remediation,
    )


if __name__ == "__main__":
    main()
