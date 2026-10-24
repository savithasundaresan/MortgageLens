from __future__ import annotations
import pandas as pd

REQUIRED_COLUMNS = {
    "application_id", "application_year", "state", "loan_purpose", "loan_type",
    "applicant_race", "applicant_ethnicity", "applicant_sex", "applicant_age_band",
    "income_thousands", "loan_amount_thousands", "property_value_thousands",
    "debt_to_income_ratio", "loan_to_value_ratio", "interest_rate",
    "action_taken", "denial_reason", "approved"
}

NUMERIC_COLUMNS = [
    "application_year", "income_thousands", "loan_amount_thousands",
    "property_value_thousands", "debt_to_income_ratio", "loan_to_value_ratio",
    "interest_rate", "approved"
]

def validate_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the project's analysis-friendly application schema."""
    validate_columns(df)
    cleaned = df.copy()
    cleaned = cleaned.drop_duplicates(subset=["application_id"], keep="first")
    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")
    for col in ["state", "loan_purpose", "loan_type", "applicant_race",
                "applicant_ethnicity", "applicant_sex", "applicant_age_band",
                "action_taken", "denial_reason"]:
        cleaned[col] = cleaned[col].fillna("Unknown").astype(str).str.strip()
    # Impossible negative monetary values are treated as missing, not silently corrected.
    for col in ["income_thousands", "loan_amount_thousands", "property_value_thousands"]:
        cleaned.loc[cleaned[col] < 0, col] = float("nan")
    cleaned.loc[~cleaned["approved"].isin([0, 1]), "approved"] = float("nan")
    cleaned["approval_eligible"] = cleaned["action_taken"].isin(["Originated", "Denied"])
    cleaned["is_denied"] = (cleaned["action_taken"] == "Denied").astype(int)
    cleaned["is_originated"] = (cleaned["action_taken"] == "Originated").astype(int)
    cleaned["income_to_loan_ratio"] = (
        cleaned["income_thousands"] / cleaned["loan_amount_thousands"].replace(0, float("nan"))
    )
    return cleaned.reset_index(drop=True)

def load_and_clean(path: str) -> pd.DataFrame:
    return clean_applications(pd.read_csv(path))
