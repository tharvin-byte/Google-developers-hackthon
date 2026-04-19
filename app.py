"""
app.py
------
FastAPI backend for FairSight AI Fairness Audit Pipeline.
Exposes logic from the 6-stage pipeline as a REST API.
"""

import os
import shutil
import tempfile
from typing import Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing pipeline logic
import pandas as pd
from stage0_validation import run_validation
from stage1_screening import run_screening
from stage1_rules import run_rules
from stage2_metrics import run_metrics
from stage3_scoring import run_scoring
from stage4_conflict import run_conflict_detection
from stage5_explanation import run_explanation

app = FastAPI(title="FairSight API", version="1.0.0")

# Enable CORS for Angular frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify the frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AuditResponse(BaseModel):
    status: str
    results: dict[str, Any]

@app.post("/api/audit", response_model=AuditResponse)
async def audit_dataset(file: UploadFile = File(...)):
    if not file.filename.endswith('.csv'):
        raise HTTPException(status_code=400, detail="Only CSV files are allowed.")

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # ── Stage 0: Data Validation ──
        validation = run_validation(tmp_path)
        if not validation.passed:
            # We still return the validation results so the UI can show warnings/errors
            return {
                "status": "validation_failed",
                "results": {
                    "stage0": {
                        "passed": False,
                        "group_sizes": validation.group_sizes,
                        "warnings": validation.warnings
                    }
                }
            }

        df = validation.dataframe

        # ── Stage 1: Fast Screening ──
        screening = run_screening(df)
        rules_reasoning = None
        if screening.risk_level == "HIGH":
            rules_result = run_rules(df, screening.outcome_gap)
            rules_reasoning = rules_result.reasoning

        # ── Stage 2: Core Fairness Metrics ──
        metrics = run_metrics(df)

        # ── Stage 3: Weighted Scoring ──
        scoring = run_scoring(metrics, outcome_gap=screening.outcome_gap)

        # ── Stage 4: Conflict Detection ──
        conflict = run_conflict_detection(scoring)

        # ── Stage 5: Decision Engine & Explanation ──
        explanation_result = run_explanation(df, metrics, scoring, conflict)

        # Compile final response
        results = {
            "stage0": {
                "passed": True,
                "group_sizes": validation.group_sizes,
                "warnings": validation.warnings
            },
            "stage1": {
                "outcome_gap": screening.outcome_gap,
                "risk_level": screening.risk_level,
                "rules_reasoning": rules_reasoning
            },
            "stage2": {
                "disparity_ratio": metrics.disparity_ratio,
                "fpr_difference": metrics.fpr_difference,
                "tpr_difference": metrics.tpr_difference,
                "per_group": {k: v._asdict() for k, v in metrics.per_group.items()}
            },
            "stage3": {
                "weighted_score": scoring.weighted_score,
                "verdict": scoring.verdict,
                "normalised_scores": scoring.normalised_scores
            },
            "stage4": {
                "conflict_detected": conflict.conflict_detected,
                "recommendation": conflict.recommendation,
                "conflicting_pair": conflict.conflicting_pair
            },
            "stage5": {
                "explanation": explanation_result.explanation,
                "remediation": explanation_result.remediation,
                "most_affected_group": explanation_result.most_affected_group,
                "confidence": explanation_result.confidence,
                "confidence_reason": explanation_result.confidence_reason,
                "shap_insight": explanation_result.shap_insight,
                "top_features": explanation_result.top_features
            }
        }

        return AuditResponse(status="success", results=results)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Cleanup temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
