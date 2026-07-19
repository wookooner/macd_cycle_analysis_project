import pandas as pd

from src.backtest.validation import validate_decision_table, validate_manifest


def test_proxy_and_low_sample_blocked_from_decision_table():
    df = pd.DataFrame({"rule_kind": ["proxy", "production"], "sample_class": ["reliable_sample", "low_sample"]})
    codes = {issue["code"] for issue in validate_decision_table(df)}
    assert "proxy_in_decision_table" in codes
    assert "low_sample_in_decision_table" in codes


def test_duplicate_hash_detected():
    issues = validate_manifest([{"sha256": "same"}, {"sha256": "same"}])
    assert issues and issues[0]["code"] == "duplicate_output_hash"
