import math

import pytest

from clinical_ai_tool_agent.calculators import (
    anion_gap,
    bmi,
    mean_arterial_pressure,
)


def test_bmi_known_values():
    # 1.75 m, 70 kg -> 22.857...
    assert math.isclose(bmi(1.75, 70), 22.857, abs_tol=0.01)
    # 1.70 m, 70 kg -> 24.221...
    assert math.isclose(bmi(1.70, 70), 24.221, abs_tol=0.01)


def test_bmi_invalid_inputs():
    with pytest.raises(ValueError):
        bmi(0, 70)
    with pytest.raises(ValueError):
        bmi(1.75, 0)
    with pytest.raises(ValueError):
        bmi(-1.0, 70)


def test_map_normal_bp():
    # 120/80 -> (120 + 160) / 3 = 93.33
    assert math.isclose(mean_arterial_pressure(120, 80), 93.333, abs_tol=0.01)


def test_map_diastolic_weighted_more_than_systolic():
    # The same average BP but higher diastolic should yield higher MAP.
    same_average_high_dbp = mean_arterial_pressure(100, 90)  # avg 95
    same_average_low_dbp = mean_arterial_pressure(140, 50)   # avg 95
    assert same_average_high_dbp > same_average_low_dbp


def test_map_invalid_inputs():
    with pytest.raises(ValueError):
        mean_arterial_pressure(0, 80)
    with pytest.raises(ValueError):
        mean_arterial_pressure(120, 0)


def test_anion_gap_normal_range():
    # Na 138, Cl 100, HCO3 22 -> 16
    assert anion_gap(138, 100, 22) == 16


def test_anion_gap_can_be_negative():
    # Pathological-looking inputs still compute (sign preserved).
    assert anion_gap(130, 110, 25) == -5
