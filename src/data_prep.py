from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {
    "application_id",
    "application_year",
    "state",
    "loan_purpose",
    "loan_type",
    "applicant_race",
    "applicant_ethnicity",
    "applicant_sex",
    "applicant_age_band",
    "income_thousands",
    "loan_amount_thousands",
    "property_value_thousands",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "interest_rate",
    "action_taken",
    "denial_reason",
    "approved",
}

NUMERIC_COLUMNS = [
    "application_year",
    "income_thousands",
    "loan_amount_thousands",
    "property_value_thousands",
    "debt_to_income_ratio",
    "loan_to_value_ratio",
    "interest_rate",
    "approved",
]

CATEGORICAL_COLUMNS = [
    "state",
    "loan_purpose",
    "loan_type",
    "applicant_race",
    "applicant_ethnicity",
    "applicant_sex",
    "applicant_age_band",
    "action_taken",
    "denial_reason",
]

ELIGIBLE_OUTCOMES = {"Originated", "Denied"}


def validate_columns(df: pd.DataFrame) -> None:
    """Raise an error if required analysis columns are missing."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")


def clean_applications(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean the analysis-friendly application schema."""
    validate_columns(df)

    cleaned = df.copy()

    # A missing application ID cannot safely identify a unique application.
    cleaned["application_id"] = cleaned["application_id"].astype("string").str.strip()
    cleaned = cleaned.dropna(subset=["application_id"])
    cleaned = cleaned[cleaned["application_id"].ne("")]
    cleaned = cleaned.drop_duplicates(subset=["application_id"], keep="first")

    # Convert numeric columns without treating missing values as zero.
    for col in NUMERIC_COLUMNS:
        cleaned[col] = pd.to_numeric(cleaned[col], errors="coerce")

    # Normalize categories.
    for col in CATEGORICAL_COLUMNS:
        cleaned[col] = (
            cleaned[col]
            .fillna("Unknown")
            .astype(str)
            .str.strip()
            .replace("", "Unknown")
        )

    # Impossible negative monetary amounts become missing.
    for col in [
        "income_thousands",
        "loan_amount_thousands",
        "property_value_thousands",
    ]:
        cleaned.loc[cleaned[col] < 0, col] = float("nan")

    # Remove physically invalid negative values from rate/ratio fields.
    # Do not impose arbitrary upper limits here; investigate those separately.
    for col in [
        "debt_to_income_ratio",
        "loan_to_value_ratio",
        "interest_rate",
    ]:
        cleaned.loc[cleaned[col] < 0, col] = float("nan")

    # Only originated and denied applications have an observed binary
    # outcome for this project's initial approval/denial analysis.
    cleaned["approval_eligible"] = cleaned["action_taken"].isin(ELIGIBLE_OUTCOMES)

    cleaned["is_denied"] = pd.Series(pd.NA, index=cleaned.index, dtype="Int64")
    cleaned["is_originated"] = pd.Series(pd.NA, index=cleaned.index, dtype="Int64")

    eligible = cleaned["approval_eligible"]

    cleaned.loc[eligible, "is_denied"] = (
        cleaned.loc[eligible, "action_taken"].eq("Denied").astype(int)
    )
    cleaned.loc[eligible, "is_originated"] = (
        cleaned.loc[eligible, "action_taken"].eq("Originated").astype(int)
    )

    # Derive the expected target from the recorded action.
    expected_approved = cleaned["action_taken"].map({"Originated": 1.0, "Denied": 0.0})

    # Keep the source target for eligible rows, but flag inconsistencies.
    cleaned["target_mismatch"] = (
        eligible
        & cleaned["approved"].notna()
        & cleaned["approved"].ne(expected_approved)
    )

    # Use the observed action as the analysis target for eligible rows.
    # All other outcomes remain missing, not rejected/denied.
    cleaned["approved"] = expected_approved.where(eligible)

    # Descriptive ratio only; zero loan amounts are undefined.
    cleaned["income_to_loan_ratio"] = cleaned["income_thousands"] / cleaned[
        "loan_amount_thousands"
    ].replace(0, float("nan"))

    return cleaned.reset_index(drop=True)


def load_and_clean(path: str) -> pd.DataFrame:
    """Read a CSV and return its cleaned analysis dataset."""
    return clean_applications(pd.read_csv(path))


def summarize_data_quality(df: pd.DataFrame) -> dict:
    """Summarize data-quality issues before cleaning modifies the data."""
    quality = df.copy()

    # Count duplicate IDs before clean_applications removes them.
    if "application_id" in quality.columns:
        ids = quality["application_id"].astype("string").str.strip()
        duplicate_ids = int(ids.notna().where(ids.ne(""), False).duplicated().sum())
        missing_ids = int((ids.isna() | ids.eq("")).sum())
    else:
        duplicate_ids = 0
        missing_ids = 0

    # Fields where missingness is useful to monitor.
    monitored_columns = [
        "application_year",
        "state",
        "loan_purpose",
        "income_thousands",
        "loan_amount_thousands",
        "property_value_thousands",
        "debt_to_income_ratio",
        "loan_to_value_ratio",
        "interest_rate",
        "applicant_race",
        "applicant_ethnicity",
        "applicant_sex",
        "applicant_age_band",
    ]

    monitored_columns = [col for col in monitored_columns if col in quality.columns]

    missingness = pd.DataFrame(
        {
            "field": monitored_columns,
            "missing_count": [
                int(quality[col].isna().sum()) for col in monitored_columns
            ],
            "missing_percent": [
                float(quality[col].isna().mean() * 100) for col in monitored_columns
            ],
        }
    ).sort_values("missing_percent", ascending=False)

    # Count non-empty values that cannot be parsed as numeric.
    numeric_columns = [col for col in NUMERIC_COLUMNS if col in quality.columns]

    invalid_numeric = []
    for col in numeric_columns:
        original = quality[col]
        converted = pd.to_numeric(original, errors="coerce")
        invalid_count = int((original.notna() & converted.isna()).sum())
        invalid_numeric.append(
            {
                "field": col,
                "invalid_numeric_count": invalid_count,
            }
        )

    invalid_numeric = pd.DataFrame(invalid_numeric)

    return {
        "row_count": len(quality),
        "duplicate_application_ids": duplicate_ids,
        "missing_application_ids": missing_ids,
        "missingness": missingness,
        "invalid_numeric": invalid_numeric,
    }
