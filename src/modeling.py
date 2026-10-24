from __future__ import annotations
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "income_thousands", "loan_amount_thousands", "property_value_thousands",
    "debt_to_income_ratio", "loan_to_value_ratio"
]
CATEGORICAL_FEATURES = ["loan_purpose", "loan_type", "state"]

def build_pipeline(model_kind: str = "logistic") -> Pipeline:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric, NUMERIC_FEATURES),
        ("categorical", categorical, CATEGORICAL_FEATURES)
    ])
    if model_kind == "logistic":
        estimator = LogisticRegression(max_iter=1000, class_weight="balanced")
    elif model_kind == "random_forest":
        estimator = RandomForestClassifier(n_estimators=150, min_samples_leaf=5,
                                           class_weight="balanced", random_state=42,
                                           n_jobs=-1)
    else:
        raise ValueError("model_kind must be 'logistic' or 'random_forest'")
    return Pipeline([("preprocessor", preprocessor), ("model", estimator)])

def train_and_evaluate(df: pd.DataFrame) -> dict:
    """Demonstration model for originated vs denied cases, not credit underwriting."""
    data = df[df["action_taken"].isin(["Originated", "Denied"])].copy()
    data["target"] = (data["action_taken"] == "Originated").astype(int)
    data = data.dropna(subset=["target"])
    if len(data) < 30 or data["target"].nunique() != 2:
        raise ValueError("Need at least 30 eligible rows and both target classes.")
    features = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = data["target"].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )
    results = {}
    for kind in ["logistic", "random_forest"]:
        pipeline = build_pipeline(kind)
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        results[kind] = {
            "model": pipeline,
            "metrics": {
                "accuracy": float(accuracy_score(y_test, pred)),
                "precision": float(precision_score(y_test, pred, zero_division=0)),
                "recall": float(recall_score(y_test, pred, zero_division=0)),
                "f1": float(f1_score(y_test, pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_test, proba)),
            },
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
            "test_rows": int(len(y_test)),
        }
    return results
