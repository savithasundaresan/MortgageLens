import pandas as pd
import pytest
from src.data_prep import clean_applications, summarize_data_quality


def sample():
    return pd.DataFrame(
        [
            {
                "application_id": "A1",
                "application_year": 2024,
                "state": "CA",
                "loan_purpose": "Home purchase",
                "loan_type": "Conventional",
                "applicant_race": "Asian",
                "applicant_ethnicity": "Not Hispanic",
                "applicant_sex": "Female",
                "applicant_age_band": "25-34",
                "income_thousands": 100,
                "loan_amount_thousands": 400,
                "property_value_thousands": 500,
                "debt_to_income_ratio": 30,
                "loan_to_value_ratio": 80,
                "interest_rate": 6.5,
                "action_taken": "Originated",
                "denial_reason": "Not applicable",
                "approved": 1,
            }
        ]
    )


def test_cleaner_adds_derived_columns():
    out = clean_applications(sample())
    assert "is_originated" in out
    assert out.loc[0, "is_originated"] == 1
    assert out.loc[0, "approval_eligible"]


def test_duplicate_application_id_removed():
    df = pd.concat([sample(), sample()], ignore_index=True)
    assert len(clean_applications(df)) == 1


def test_missing_required_column_raises():
    df = sample().drop(columns=["state"])
    with pytest.raises(ValueError):
        clean_applications(df)


def test_negative_income_becomes_missing():
    df = sample()
    df.loc[0, "income_thousands"] = -5
    assert pd.isna(clean_applications(df).loc[0, "income_thousands"])


def test_quality_report_counts_duplicates_before_cleaning():
    df = pd.concat([sample(), sample()], ignore_index=True)

    report = summarize_data_quality(df)

    assert report["row_count"] == 2
    assert report["duplicate_application_ids"] == 1


def test_quality_report_counts_original_missing_values():
    df = sample()
    df.loc[0, "income_thousands"] = None
    df.loc[0, "applicant_race"] = None

    report = summarize_data_quality(df)
    missingness = report["missingness"].set_index("field")

    assert missingness.loc["income_thousands", "missing_count"] == 1
    assert missingness.loc["applicant_race", "missing_count"] == 1


def test_quality_report_detects_invalid_numeric_values():
    df = sample()
    df["income_thousands"] = "not-a-number"

    report = summarize_data_quality(df)
    invalid = report["invalid_numeric"].set_index("field")

    assert invalid.loc["income_thousands", "invalid_numeric_count"] == 1
