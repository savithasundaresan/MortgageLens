import pandas as pd
import pytest
from src.data_prep import clean_applications

def sample():
    return pd.DataFrame([{
        "application_id":"A1", "application_year":2024, "state":"CA",
        "loan_purpose":"Home purchase", "loan_type":"Conventional",
        "applicant_race":"Asian", "applicant_ethnicity":"Not Hispanic",
        "applicant_sex":"Female", "applicant_age_band":"25-34",
        "income_thousands":100, "loan_amount_thousands":400,
        "property_value_thousands":500, "debt_to_income_ratio":30,
        "loan_to_value_ratio":80, "interest_rate":6.5,
        "action_taken":"Originated", "denial_reason":"Not applicable", "approved":1
    }])

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
