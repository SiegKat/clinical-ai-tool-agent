from clinical_ai_tool_agent.parser import parse_action_or_final


def test_parses_well_formed_action():
    text = """Some preamble.
Action: calc_bmi
Action Input: {"height_m": 1.75, "weight_kg": 70}
"""
    name, args, final = parse_action_or_final(text)
    assert name == "calc_bmi"
    assert args == {"height_m": 1.75, "weight_kg": 70}
    assert final is None


def test_parses_final_answer():
    text = "Final Answer: MAP is approximately 93 mmHg.\n"
    name, args, final = parse_action_or_final(text)
    assert name is None and args is None
    assert final == "MAP is approximately 93 mmHg."


def test_final_answer_takes_precedence_over_action():
    text = """Final Answer: done.

Action: calc_bmi
Action Input: {"height_m": 1.75, "weight_kg": 70}
"""
    name, args, final = parse_action_or_final(text)
    assert final == "done."
    assert name is None


def test_returns_all_none_when_neither_present():
    name, args, final = parse_action_or_final("just rambling text")
    assert (name, args, final) == (None, None, None)


def test_empty_input_returns_all_none():
    assert parse_action_or_final("") == (None, None, None)


def test_invalid_json_args_returns_none_args():
    text = """Action: calc_bmi
Action Input: {height_m: 1.75}
"""
    name, args, _ = parse_action_or_final(text)
    assert name == "calc_bmi"
    assert args is None


def test_action_with_hyphenated_tool_name():
    text = """Action: some-tool-name
Action Input: {"x": 1}
"""
    name, _, _ = parse_action_or_final(text)
    assert name == "some-tool-name"
