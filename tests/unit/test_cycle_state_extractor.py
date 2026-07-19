from src.cycles.state_extractor import extract_state_as_of


def test_extract_state_as_of_returns_combo_and_states():
    state = extract_state_as_of("btc", "2024-01-01", mode="closed_only")
    assert state.asset == "btc"
    assert state.combo
    assert "1h" in state.states
