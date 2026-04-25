"""Pure-math clinical calculators.

Kept free of LangChain so they can be unit-tested in isolation. The
LangChain `@tool` wrappers in `tools.py` delegate to these functions.

EDUCATIONAL USE ONLY. These are simplified formulas for teaching, not
clinical decision support.
"""

from __future__ import annotations


def bmi(height_m: float, weight_kg: float) -> float:
    """Body Mass Index = weight_kg / (height_m ** 2)."""
    if height_m <= 0:
        raise ValueError("height_m must be > 0")
    if weight_kg <= 0:
        raise ValueError("weight_kg must be > 0")
    return weight_kg / (height_m ** 2)


def mean_arterial_pressure(sbp: float, dbp: float) -> float:
    """MAP approximation = (SBP + 2 * DBP) / 3, in mmHg."""
    if sbp <= 0 or dbp <= 0:
        raise ValueError("sbp and dbp must be > 0")
    return (sbp + 2 * dbp) / 3


def anion_gap(na: float, cl: float, hco3: float) -> float:
    """Anion gap = Na - (Cl + HCO3), in mEq/L. Sign is preserved."""
    return na - (cl + hco3)
