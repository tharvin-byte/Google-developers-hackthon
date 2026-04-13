"""
utils.py
--------
Shared utility functions for the FairSight fairness auditing pipeline.
Provides logging, formatting, and common data helpers used across all stages.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------

def get_logger(name: str) -> logging.Logger:
    """
    Return a consistently configured logger for a given module name.

    Parameters
    ----------
    name : str
        Usually ``__name__`` of the calling module.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                fmt="[%(levelname)s] %(name)s - %(message)s"
            )
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def section_banner(title: str, width: int = 60) -> str:
    """
    Return a formatted section banner string.

    Parameters
    ----------
    title : str
        Section title to display.
    width : int
        Total banner width in characters.

    Returns
    -------
    str
    """
    border = "=" * width
    padded = f"  {title}  ".center(width)
    return f"\n{border}\n{padded}\n{border}"


def fmt_pct(value: float) -> str:
    """
    Format a float as a percentage string rounded to two decimal places.

    Parameters
    ----------
    value : float
        Numeric value (e.g. 0.123 → '12.30%').

    Returns
    -------
    str
    """
    return f"{value * 100:.2f}%"


def fmt_float(value: float, decimals: int = 4) -> str:
    """
    Format a float to a fixed number of decimal places.

    Parameters
    ----------
    value : float
        Numeric value to format.
    decimals : int
        Number of decimal places.

    Returns
    -------
    str
    """
    return f"{value:.{decimals}f}"


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp *value* to the closed interval [lo, hi].

    Parameters
    ----------
    value : float
    lo : float
    hi : float

    Returns
    -------
    float
    """
    return max(lo, min(hi, value))


def normalize_metric(
    value: float,
    worst: float,
    best: float,
) -> float:
    """
    Min-max normalise *value* so that *best* → 1.0 and *worst* → 0.0.

    Parameters
    ----------
    value : float
        Raw metric value to normalise.
    worst : float
        The value that represents the worst-case scenario.
    best : float
        The value that represents the best-case scenario (fairness).

    Returns
    -------
    float
        Normalised score in [0.0, 1.0].
    """
    span = abs(best - worst)
    if span == 0:
        return 1.0
    score = 1.0 - abs(value - best) / span
    return clamp(score)


# ---------------------------------------------------------------------------
# Misc helpers
# ---------------------------------------------------------------------------

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Divide numerator by denominator, returning *default* on zero-division.

    Parameters
    ----------
    numerator : float
    denominator : float
    default : float
        Value returned when *denominator* is zero.

    Returns
    -------
    float
    """
    if denominator == 0:
        return default
    return numerator / denominator


def dict_pretty(d: dict[str, Any], indent: int = 2) -> str:
    """
    Return a human-readable multi-line representation of a flat dictionary.

    Parameters
    ----------
    d : dict
        Dictionary to format.
    indent : int
        Leading spaces per item.

    Returns
    -------
    str
    """
    pad = " " * indent
    lines = [f"{pad}{k}: {v}" for k, v in d.items()]
    return "\n".join(lines)
