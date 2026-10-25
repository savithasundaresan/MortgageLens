import pandas as pd
import pytest
from src.analysis import approval_summary, two_proportion_z_test


def test_approval_rate_excludes_withdrawn():
    df = pd.DataFrame({"action_taken": ["Originated", "Denied", "Withdrawn"]})
    result = approval_summary(df).iloc[0]
    assert result["applications"] == 2
    assert result["origination_share"] == 0.5


def test_two_proportion_test_returns_valid_p_value():
    result = two_proportion_z_test(80, 100, 60, 100)
    assert 0 <= result["p_value"] <= 1
    assert result["difference"] == pytest.approx(0.2)


def test_invalid_counts_raise():
    with pytest.raises(ValueError):
        two_proportion_z_test(5, 4, 2, 10)
