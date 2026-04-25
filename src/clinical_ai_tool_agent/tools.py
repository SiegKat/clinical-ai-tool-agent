"""LangChain Tool wrappers around the pure-math calculators.

Pydantic-validated input schemas; one-line output strings the LLM can
quote back. The actual math lives in `calculators.py`.
"""

from __future__ import annotations


def build_tools():
    """Return the list of LangChain Tools. Lazy import keeps the package
    importable in environments without LangChain (e.g. unit tests)."""
    from langchain_core.tools import tool
    from pydantic import BaseModel, Field

    from clinical_ai_tool_agent import calculators

    class BMIPayload(BaseModel):
        height_m: float = Field(..., gt=0, description="Height in meters")
        weight_kg: float = Field(..., gt=0, description="Weight in kilograms")

    class MAPPayload(BaseModel):
        sbp: float = Field(..., gt=0, description="Systolic BP (mmHg)")
        dbp: float = Field(..., gt=0, description="Diastolic BP (mmHg)")

    class AnionGapPayload(BaseModel):
        na: float = Field(..., description="Sodium (mEq/L)")
        cl: float = Field(..., description="Chloride (mEq/L)")
        hco3: float = Field(..., description="Bicarbonate (mEq/L)")

    @tool("calc_bmi", args_schema=BMIPayload)
    def calc_bmi(height_m: float, weight_kg: float) -> str:
        """BMI = kg / m^2. Educational estimate; not for diagnosis."""
        return f"BMI={calculators.bmi(height_m, weight_kg):.1f} (educational estimate)"

    @tool("calc_map", args_schema=MAPPayload)
    def calc_map(sbp: float, dbp: float) -> str:
        """Mean Arterial Pressure ~= (SBP + 2*DBP)/3 mmHg."""
        return (
            f"MAP~={calculators.mean_arterial_pressure(sbp, dbp):.0f} mmHg "
            "(educational estimate)"
        )

    @tool("calc_anion_gap", args_schema=AnionGapPayload)
    def calc_anion_gap(na: float, cl: float, hco3: float) -> str:
        """Anion Gap = Na - (Cl + HCO3) mEq/L."""
        return (
            f"Anion gap={calculators.anion_gap(na, cl, hco3):.1f} "
            "(educational estimate)"
        )

    return [calc_bmi, calc_map, calc_anion_gap]


def tool_summaries(tools) -> str:
    """One line per tool, used in the ReAct system prompt."""
    lines = []
    for t in tools:
        schema = getattr(t, "args", None)
        if schema:
            lines.append(f"- {t.name}: args schema = {schema}")
        else:
            lines.append(f"- {t.name}")
    return "\n".join(lines)
