import pandas as pd
from src.modeling import build_pipeline


def test_pipeline_rejects_unknown_model():
    import pytest

    with pytest.raises(ValueError):
        build_pipeline("neural_magic")


def test_feature_schema_is_documented():
    from src.modeling import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    assert "applicant_race" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES
    assert "applicant_ethnicity" not in NUMERIC_FEATURES + CATEGORICAL_FEATURES
