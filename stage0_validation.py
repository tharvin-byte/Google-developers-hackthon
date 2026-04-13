"""
stage0_validation.py
--------------------
Stage 0: Data Validation

Loads dataset.csv, verifies the presence of required columns, handles missing
values, and ensures that each demographic group has a statistically reliable
sample size (n >= 30).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import NamedTuple

import pandas as pd

from utils import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REQUIRED_COLUMNS: list[str] = [
    "gender",
    "income",
    "experience",
    "education_level",
    "y_true",
    "y_pred",
]

MIN_GROUP_SIZE: int = 30          # flag groups smaller than this
SENSITIVE_ATTRIBUTE: str = "gender"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

class ValidationResult(NamedTuple):
    """Structured output returned by :func:`run_validation`."""

    dataframe: pd.DataFrame
    group_sizes: dict[str, int]
    warnings: list[str]
    passed: bool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _check_required_columns(df: pd.DataFrame) -> list[str]:
    """
    Return a list of column names that are missing from *df*.

    Parameters
    ----------
    df : pd.DataFrame
        Loaded dataset.

    Returns
    -------
    list[str]
        Missing column names; empty list when all required columns are present.
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def _drop_missing_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Drop rows that contain any NaN in the required columns and report the count.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[pd.DataFrame, int]
        Cleaned dataframe and the number of rows dropped.
    """
    before = len(df)
    df_clean = df.dropna(subset=REQUIRED_COLUMNS).copy()
    dropped = before - len(df_clean)
    return df_clean, dropped


def _validate_binary_column(df: pd.DataFrame, column: str) -> list[str]:
    """
    Ensure *column* contains only binary values (0 / 1).

    Parameters
    ----------
    df : pd.DataFrame
    column : str

    Returns
    -------
    list[str]
        Warning messages; empty if the column is valid.
    """
    unique_vals = set(df[column].unique())
    if not unique_vals.issubset({0, 1}):
        return [
            f"Column '{column}' contains non-binary values: {unique_vals}. "
            "Expected only 0 and 1."
        ]
    return []


def _check_group_sizes(df: pd.DataFrame) -> tuple[dict[str, int], list[str]]:
    """
    Compute per-group sample sizes and flag groups below MIN_GROUP_SIZE.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    tuple[dict[str, int], list[str]]
        Dictionary of {group_label: count} and a list of warning strings.
    """
    sizes: dict[str, int] = (
        df[SENSITIVE_ATTRIBUTE].value_counts().to_dict()
    )
    warnings: list[str] = []

    for group, count in sizes.items():
        if count < MIN_GROUP_SIZE:
            warnings.append(
                f"Group '{group}' has only {count} samples "
                f"(minimum recommended: {MIN_GROUP_SIZE}). "
                "Statistical estimates may be unreliable."
            )

    return sizes, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_validation(dataset_path: str = "dataset.csv") -> ValidationResult:
    """
    Execute Stage 0: Data Validation.

    Steps:
      1. Load ``dataset_path`` into a DataFrame.
      2. Verify the presence of all required columns (hard failure on missing).
      3. Drop rows with missing values in required columns.
      4. Validate that ``y_true`` and ``y_pred`` are binary.
      5. Compute group sizes and flag small groups.

    Parameters
    ----------
    dataset_path : str
        Relative or absolute path to the CSV dataset.

    Returns
    -------
    ValidationResult
        Named tuple containing the cleaned DataFrame, group sizes, warnings,
        and a boolean ``passed`` flag.

    Raises
    ------
    SystemExit
        When required columns are missing (unrecoverable error).
    FileNotFoundError
        When ``dataset_path`` does not exist.
    """
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: '{dataset_path}'. "
            "Ensure dataset.csv is in the working directory."
        )

    logger.info("Loading dataset from '%s'.", dataset_path)
    df = pd.read_csv(path, encoding="utf-8")
    logger.info("Dataset loaded — %d rows, %d columns.", *df.shape)

    # 1. Column presence check
    missing_cols = _check_required_columns(df)
    if missing_cols:
        logger.error("Missing required columns: %s", missing_cols)
        sys.exit(1)
    logger.info("All required columns present: %s", REQUIRED_COLUMNS)

    # 2. Handle missing values
    df, dropped = _drop_missing_rows(df)
    if dropped > 0:
        logger.warning(
            "%d row(s) dropped due to missing values in required columns.",
            dropped,
        )
    else:
        logger.info("No missing values found in required columns.")

    # 3. Validate binary columns
    all_warnings: list[str] = []
    for binary_col in ("y_true", "y_pred"):
        all_warnings.extend(_validate_binary_column(df, binary_col))

    for w in all_warnings:
        logger.warning(w)

    # 4. Group size check
    group_sizes, size_warnings = _check_group_sizes(df)
    all_warnings.extend(size_warnings)
    for w in size_warnings:
        logger.warning(w)

    logger.info(
        "Group sizes — %s",
        ", ".join(f"{k}: {v}" for k, v in group_sizes.items()),
    )

    passed = len([w for w in all_warnings if "non-binary" in w]) == 0

    logger.info(
        "Stage 0 validation %s.", "PASSED" if passed else "FAILED (see warnings)"
    )

    return ValidationResult(
        dataframe=df,
        group_sizes=group_sizes,
        warnings=all_warnings,
        passed=passed,
    )
