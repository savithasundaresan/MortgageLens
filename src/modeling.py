from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

BASE_NUMERIC_FEATURES = [
    "income_thousands",
    "loan_amount_thousands",
    "property_value_thousands",
    "loan_to_value_ratio",
]

BASE_CATEGORICAL_FEATURES = [
    "loan_purpose",
    "loan_type",
    "state",
]

# Backward-compatible feature names used by the tests and documentation.
NUMERIC_FEATURES = BASE_NUMERIC_FEATURES + ["debt_to_income_ratio"]
CATEGORICAL_FEATURES = BASE_CATEGORICAL_FEATURES


def build_pipeline(
    model_kind: str = "logistic",
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> Pipeline:
    """Build a leakage-conscious preprocessing and classification pipeline."""

    numeric_features = (
        numeric_features
        if numeric_features is not None
        else BASE_NUMERIC_FEATURES + ["debt_to_income_ratio"]
    )
    categorical_features = (
        categorical_features
        if categorical_features is not None
        else BASE_CATEGORICAL_FEATURES
    )

    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ]
    )

    if model_kind == "logistic":
        estimator = LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
    elif model_kind == "random_forest":
        estimator = RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
    else:
        raise ValueError("model_kind must be 'logistic' or 'random_forest'")

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("model", estimator),
        ]
    )


def _calculate_metrics(
    y_true: pd.Series,
    predictions: np.ndarray,
    probabilities: np.ndarray | None,
) -> dict:
    """Calculate classification metrics using 0=denied and 1=originated."""

    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "roc_auc": float("nan"),
    }

    if probabilities is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))

    return metrics


def train_and_evaluate(df: pd.DataFrame) -> dict:
    """
    Compare a majority-class baseline with two classifiers.

    Target:
        1 = originated
        0 = denied

    This predicts recorded HMDA outcomes, not creditworthiness or
    whether a lending decision was fair.
    """

    required = {
        "action_taken",
        *BASE_NUMERIC_FEATURES,
        "loan_purpose",
        "loan_type",
        "state",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing modeling columns: {sorted(missing)}")

    data = df[df["action_taken"].isin(["Originated", "Denied"])].copy()

    if len(data) < 30:
        raise ValueError("Need at least 30 originated/denied records.")

    data["target"] = data["action_taken"].eq("Originated").astype(int)

    if data["target"].nunique() != 2:
        raise ValueError("Need both originated and denied outcomes.")

    # Use reported DTI categories when available. Do not treat
    # approximate category midpoints as exact numeric observations.
    numeric_features = [col for col in BASE_NUMERIC_FEATURES if col in data.columns]
    categorical_features = [
        col for col in BASE_CATEGORICAL_FEATURES if col in data.columns
    ]

    if (
        "debt_to_income_category" in data.columns
        and data["debt_to_income_category"].notna().any()
    ):
        categorical_features.append("debt_to_income_category")
    elif "debt_to_income_ratio" in data.columns:
        numeric_features.append("debt_to_income_ratio")

    features = numeric_features + categorical_features
    X = data[features].copy()
    y = data["target"].astype(int)

    # Normalize categorical missing values; numeric missing values
    # remain NaN and are imputed inside each training pipeline.
    for col in categorical_features:
        X[col] = X[col].where(X[col].notna(), np.nan)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    results = {}

    # Majority-class benchmark: shows how a trivial classifier performs.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)

    results["majority_baseline"] = {
        "model": baseline,
        "metrics": _calculate_metrics(y_test, baseline_pred, None),
        "confusion_matrix": confusion_matrix(
            y_test, baseline_pred, labels=[0, 1]
        ).tolist(),
        "test_rows": int(len(y_test)),
        "train_rows": int(len(y_train)),
        "features": features,
        "class_balance": {
            "denied": int((y == 0).sum()),
            "originated": int((y == 1).sum()),
        },
    }

    for kind in ["logistic", "random_forest"]:
        pipeline = build_pipeline(
            kind,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )
        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        # Retain test outcomes and predictions for error and threshold analysis.
        test_results = {
            "y_true": y_test.to_numpy().tolist(),
            "y_pred": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "test_indices": X_test.index.tolist(),
        }

        results[kind] = {
            "model": pipeline,
            "metrics": _calculate_metrics(y_test, predictions, probabilities),
            "confusion_matrix": confusion_matrix(
                y_test, predictions, labels=[0, 1]
            ).tolist(),
            "test_rows": int(len(y_test)),
            "train_rows": int(len(y_train)),
            "features": features,
            "class_balance": {
                "denied": int((y == 0).sum()),
                "originated": int((y == 1).sum()),
            },
            "test_results": test_results,
        }

    return results


def cross_validate_models(df: pd.DataFrame, n_splits: int = 5) -> dict:
    """Evaluate classifiers with stratified cross-validation.

    Target:
        1 = originated
        0 = denied

    This evaluates recorded historical outcomes, not creditworthiness.
    """
    required = {
        "action_taken",
        *BASE_NUMERIC_FEATURES,
        "loan_purpose",
        "loan_type",
        "state",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing modeling columns: {sorted(missing)}")

    data = df[df["action_taken"].isin(["Originated", "Denied"])].copy()

    if len(data) < n_splits:
        raise ValueError(f"Need at least {n_splits} eligible records.")

    y = data["action_taken"].eq("Originated").astype(int)

    if y.value_counts().min() < n_splits:
        raise ValueError("Each outcome class must have at least n_splits records.")

    numeric_features = [col for col in BASE_NUMERIC_FEATURES if col in data.columns]
    categorical_features = [
        col for col in BASE_CATEGORICAL_FEATURES if col in data.columns
    ]

    if (
        "debt_to_income_category" in data.columns
        and data["debt_to_income_category"].notna().any()
    ):
        categorical_features.append("debt_to_income_category")
    elif "debt_to_income_ratio" in data.columns:
        numeric_features.append("debt_to_income_ratio")

    features = numeric_features + categorical_features
    X = data[features].copy()

    for col in categorical_features:
        X[col] = X[col].where(X[col].notna(), np.nan)

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    results = {}

    for kind in ["logistic", "random_forest"]:
        pipeline = build_pipeline(
            kind,
            numeric_features=numeric_features,
            categorical_features=categorical_features,
        )

        scores = cross_validate(
            pipeline,
            X,
            y,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
            error_score="raise",
        )

        results[kind] = {
            metric: {
                "mean": float(np.mean(scores[f"test_{metric}"])),
                "std": float(np.std(scores[f"test_{metric}"], ddof=1)),
            }
            for metric in scoring
        }

    return results


def compare_dti_sensitivity(
    df: pd.DataFrame,
    n_splits: int = 5,
) -> dict:
    """Compare cross-validation performance with and without DTI.

    Target:
        1 = originated
        0 = denied

    DTI information is excluded from the second model configuration.
    Demographic variables are not used as predictors.
    """
    required = {
        "action_taken",
        *BASE_NUMERIC_FEATURES,
        "loan_purpose",
        "loan_type",
        "state",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing modeling columns: {sorted(missing)}")

    data = df[df["action_taken"].isin(["Originated", "Denied"])].copy()

    if len(data) < n_splits:
        raise ValueError(f"Need at least {n_splits} eligible records.")

    y = data["action_taken"].eq("Originated").astype(int)

    if y.value_counts().min() < n_splits:
        raise ValueError("Each outcome class must have at least n_splits records.")

    base_numeric = [col for col in BASE_NUMERIC_FEATURES if col in data.columns]
    base_categorical = [col for col in BASE_CATEGORICAL_FEATURES if col in data.columns]

    configurations = {
        "With DTI": {
            "numeric": base_numeric.copy(),
            "categorical": base_categorical.copy(),
        },
        "Without DTI": {
            "numeric": base_numeric.copy(),
            "categorical": base_categorical.copy(),
        },
    }

    # Match the existing modeling logic: use reported DTI category
    # when available; otherwise use the numeric DTI ratio.
    if (
        "debt_to_income_category" in data.columns
        and data["debt_to_income_category"].notna().any()
    ):
        configurations["With DTI"]["categorical"].append("debt_to_income_category")
    elif "debt_to_income_ratio" in data.columns:
        configurations["With DTI"]["numeric"].append("debt_to_income_ratio")

    cv = StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=42,
    )

    scoring = {
        "balanced_accuracy": "balanced_accuracy",
        "roc_auc": "roc_auc",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
    }

    results = {}

    for configuration, feature_sets in configurations.items():
        numeric_features = feature_sets["numeric"]
        categorical_features = feature_sets["categorical"]
        features = numeric_features + categorical_features
        X = data[features].copy()

        for col in categorical_features:
            X[col] = X[col].where(X[col].notna(), np.nan)

        results[configuration] = {}

        for kind in ["logistic", "random_forest"]:
            pipeline = build_pipeline(
                kind,
                numeric_features=numeric_features,
                categorical_features=categorical_features,
            )

            scores = cross_validate(
                pipeline,
                X,
                y,
                cv=cv,
                scoring=scoring,
                n_jobs=1,
                error_score="raise",
            )

            results[configuration][kind] = {
                metric: {
                    "mean": float(np.mean(scores[f"test_{metric}"])),
                    "std": float(
                        np.std(
                            scores[f"test_{metric}"],
                            ddof=1,
                        )
                    ),
                }
                for metric in scoring
            }

    return results
