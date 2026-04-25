from clinical_ai_tool_agent.knowledge import load_corpus


def test_load_corpus_returns_non_empty_facts():
    facts = load_corpus()
    assert len(facts) >= 5
    for f in facts:
        assert isinstance(f, str)
        assert f.strip() == f
        assert not f.startswith("#")


def test_corpus_mentions_core_calculators():
    text = " ".join(load_corpus()).lower()
    for keyword in ("bmi", "anion gap", "map"):
        assert keyword in text, f"corpus is missing reference to {keyword!r}"
