import pandas as pd
import pytest

from src.analysis import approval_summary
from src.sql_analysis import run_sql_summaries


def sample_applications():
    return pd.DataFrame(
        {
            "application_year": [2024, 2024, 2024, 2023],
            "state": ["CA", "CA", "CA", "NY"],
            "action_taken": [
                "Originated",
                "Denied",
                "Withdrawn",
                "Originated",
            ],
        }
    )


def test_sql_state_summary_counts_eligible_outcomes():
    df = sample_applications()

    result = run_sql_summaries(df)["approval_by_state"]
    california = result.loc[result["state"] == "CA"].iloc[0]

    assert california["eligible_applications"] == 2
    assert california["originated"] == 1
    assert california["denied"] == 1
    assert california["origination_share"] == pytest.approx(0.5)


def test_sql_and_python_origination_rates_match():
    df = sample_applications()

    sql_result = run_sql_summaries(df)["approval_by_state"]
    python_result = approval_summary(df, group_col="state")

    merged = sql_result.merge(
        python_result,
        left_on="state",
        right_on="group",
        suffixes=("_sql", "_python"),
    )

    assert len(merged) == 2

    for _, row in merged.iterrows():
        assert row["origination_share_sql"] == pytest.approx(
            row["origination_share_python"]
        )
