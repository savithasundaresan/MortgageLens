import pandas as pd

from src.reporting import build_report_context, generate_html, generate_pdf


def _test_frame():
    """Small deterministic fixture for testing report generation."""
    return pd.DataFrame({
        "application_id": ["a", "b", "c", "d"],
        "application_year": [2024, 2024, 2024, 2024],
        "state": ["CA"] * 4,
        "loan_purpose": ["Home purchase"] * 4,
        "loan_type": ["Conventional"] * 4,
        "applicant_race": ["Group A", "Group A", "Group B", "Group B"],
        "applicant_ethnicity": ["Unknown"] * 4,
        "applicant_sex": ["Female", "Male", "Female", "Male"],
        "applicant_age_band": ["25-34"] * 4,
        "income_thousands": [100.0, 120.0, 80.0, 90.0],
        "loan_amount_thousands": [400.0, 450.0, 300.0, 350.0],
        "property_value_thousands": [500.0, 550.0, 400.0, 450.0],
        "debt_to_income_ratio": [30.0, 35.0, 40.0, 32.0],
        "loan_to_value_ratio": [80.0, 82.0, 75.0, 78.0],
        "interest_rate": [6.0, 6.2, 6.5, 6.1],
        "action_taken": ["Originated", "Denied", "Originated", "Denied"],
        "denial_reason": ["Unknown", "Debt-to-income ratio", "Unknown", "Credit history"],
    })


def test_reports_generate_pdf_and_html_formats():
    context = build_report_context(_test_frame(), "Test source", "Test fixture data")
    assert context["rows"] == 4
    assert context["eligible"] == 4
    assert context["origination_share"] == 0.5
    assert b"MortgageLens" in generate_html(context)
    assert generate_pdf(context).startswith(b"%PDF")


def test_detailed_report_notes_missing_model_and_cv_results():
    context = build_report_context(_test_frame(), "Test source", "Test fixture data")
    html = generate_html(context, detailed=True).decode("utf-8")
    assert "No inferential hypothesis test is calculated" in html
    assert "Held-out model evaluation" in html
    assert "Cross-validation" in html
