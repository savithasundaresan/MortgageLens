from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
import numpy as np
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import hashlib
import base64
import fitz  # PyMuPDF: render PDF pages as images for reliable in-app preview

from src.data_prep import clean_applications, summarize_data_quality
from src.analysis import approval_summary, denial_reason_summary
from src.modeling import (
    train_and_evaluate,
    cross_validate_models,
    compare_dti_sensitivity,
)
from src.sql_analysis import run_sql_summaries
from src.reporting import build_report_context, generate_html, generate_pdf

BASE_DIR = Path(__file__).resolve().parent
HMDA_DATA = BASE_DIR / "data" / "processed" / "mortgage_applications_hmda.csv"

st.set_page_config(page_title="MortgageLens", page_icon="🏠", layout="wide")
st.title("🏠 MortgageLens")
st.caption("Mortgage Credit Risk & Fair Lending Analytics")

st.warning(
    "Educational portfolio project only. Descriptive group differences are not proof "
    "of discrimination. This dashboard is not a credit decision system and must not "
    "be used to make actual lending decisions."
)


@st.cache_data
def read_csv_file(path_or_buffer):
    return pd.read_csv(path_or_buffer)


st.sidebar.header("Data source")
source_options = []
if HMDA_DATA.exists():
    source_options.append("2024 California HMDA data")
source_options.append("Upload a CSV")
source = st.sidebar.radio("Choose dataset", source_options)

try:
    if source == "2024 California HMDA data":
        raw = read_csv_file(HMDA_DATA)
        source_caption = (
            "Source: public 2024 HMDA data, California, one institution. "
            "Purchased loans are reported as a separate action outcome and are not "
            "counted as new applications in the originated-versus-denied analysis."
        )
    else:
        uploaded = st.sidebar.file_uploader("Upload analysis-ready CSV", type=["csv"])
        if uploaded is None:
            st.info(
                "Upload a CSV containing the MortgageLens analysis-ready columns to continue."
            )
            st.stop()
        raw = pd.read_csv(uploaded)
        source_caption = f"Source: uploaded file ({uploaded.name})."

    df = clean_applications(raw)
except Exception as exc:
    st.error(f"Could not load this file: {exc}")
    st.stop()

st.caption(source_caption)

if source == "2024 California HMDA data":
    st.info(
        "Interpretation note: the source contains one institution and 2024 California records, "
        "not all California lenders. HMDA action outcomes are not interchangeable. The "
        "origination share below uses only originated and denied applications."
    )
    if "debt_to_income_category" in df.columns:
        st.caption(
            "Data note: the source DTI category is retained. The numeric DTI field is an "
            "approximate category midpoint where available, not an exact borrower-level ratio."
        )

st.sidebar.header("Filters")
years = sorted(df["application_year"].dropna().astype(int).unique().tolist())
selected_years = st.sidebar.multiselect("Application year", years, default=years)
states = sorted(df["state"].dropna().unique().tolist())
selected_states = st.sidebar.multiselect("State", states, default=states)
purposes = sorted(df["loan_purpose"].dropna().unique().tolist())
selected_purposes = st.sidebar.multiselect("Loan purpose", purposes, default=purposes)

filtered = df[
    df["application_year"].isin(selected_years)
    & df["state"].isin(selected_states)
    & df["loan_purpose"].isin(selected_purposes)
].copy()

# Fingerprint the cleaned dataset so results cannot be reused for different data.
dataset_hash = pd.util.hash_pandas_object(df, index=True).values.tobytes()

dataset_fingerprint = hashlib.sha256(
    str(tuple(df.columns)).encode("utf-8") + dataset_hash
).hexdigest()

current_model_signature = (
    source,
    dataset_fingerprint,
    tuple(selected_years),
    tuple(selected_states),
    tuple(selected_purposes),
)

eligible = filtered[filtered["action_taken"].isin(["Originated", "Denied"])]
n_eligible = len(eligible)
n_originated = int((eligible["action_taken"] == "Originated").sum())
n_denied = int((eligible["action_taken"] == "Denied").sum())
origination_share = n_originated / n_eligible if n_eligible else float("nan")
denial_share = n_denied / n_eligible if n_eligible else float("nan")
median_loan = filtered["loan_amount_thousands"].median()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total records", f"{len(filtered):,}")
k2.metric("Originated + denied", f"{n_eligible:,}")
k3.metric(
    "Origination share*",
    f"{origination_share:.1%}" if pd.notna(origination_share) else "N/A",
)
k4.metric(
    "Median loan amount", f"${median_loan:,.0f}K" if pd.notna(median_loan) else "N/A"
)
st.caption(
    "*Origination share = originated / (originated + denied). Approved-not-accepted, "
    "withdrawn, incomplete, and purchased records are excluded from this denominator. "
    "This is not a market-wide approval rate."
)

tab_overview, tab_fair, tab_model, tab_sql, tab_data, tab_report = st.tabs(
    [
        "Overview & EDA",
        "Fair Lending Review",
        "Model Evaluation",
        "SQL Analytics",
        "Data Quality & Export",
        "Portfolio Report",
    ]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        outcome_counts = (
            filtered["action_taken"]
            .value_counts(dropna=False)
            .rename_axis("Outcome")
            .reset_index(name="Records")
        )
        st.plotly_chart(
            px.bar(
                outcome_counts,
                x="Outcome",
                y="Records",
                title="Reported action outcomes",
            ),
            use_container_width=True,
        )
    with right:
        rate_df = approval_summary(filtered, "loan_purpose")
        if not rate_df.empty:
            fig = px.bar(
                rate_df,
                x="group",
                y="origination_share",
                text="origination_share",
                title="Origination share by loan purpose (originated vs denied only)",
                labels={
                    "group": "Loan purpose",
                    "origination_share": "Origination share",
                },
            )
            fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
            fig.update_yaxes(tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No originated or denied applications in the current filters.")

    left, right = st.columns(2)
    with left:
        reason_df = denial_reason_summary(filtered)
        if not reason_df.empty:
            st.plotly_chart(
                px.bar(
                    reason_df,
                    x="denial_reason",
                    y="count",
                    title="Recorded denial reasons",
                    labels={"count": "Denied applications", "denial_reason": "Reason"},
                ),
                use_container_width=True,
            )
        else:
            st.info("No denied applications in the current filters.")
    with right:
        if filtered["interest_rate"].notna().any():
            st.plotly_chart(
                px.histogram(
                    filtered.dropna(subset=["interest_rate"]),
                    x="interest_rate",
                    color="loan_type",
                    nbins=25,
                    title="Interest-rate distribution (records with a reported rate)",
                ),
                use_container_width=True,
            )
        else:
            st.info("No interest-rate values in the current filters.")

    scatter_data = filtered.dropna(subset=["income_thousands", "loan_amount_thousands"])
    if not scatter_data.empty:
        st.plotly_chart(
            px.scatter(
                scatter_data,
                x="income_thousands",
                y="loan_amount_thousands",
                color="action_taken",
                title="Income vs. loan amount",
                labels={
                    "income_thousands": "Income ($ thousands)",
                    "loan_amount_thousands": "Loan amount ($ thousands)",
                },
            ),
            use_container_width=True,
        )

with tab_fair:
    st.subheader("Descriptive group-level outcome comparisons")
    st.write(
        "These are unadjusted comparisons among originated and denied applications only. "
        "Differences are not proof of discrimination or causation. Group sizes, missing data, "
        "loan characteristics, and other factors should be considered."
    )
    demographic = st.selectbox(
        "Compare groups by",
        [
            "applicant_race",
            "applicant_ethnicity",
            "applicant_sex",
            "applicant_age_band",
        ],
    )
    fair = approval_summary(filtered, demographic).sort_values(
        "origination_share", ascending=False
    )
    if not fair.empty:
        fair["origination_share_pct"] = fair["origination_share"] * 100
        st.dataframe(fair, use_container_width=True, hide_index=True)
        fig = px.bar(
            fair,
            x="group",
            y="origination_share",
            hover_data=["applications", "originated", "denied"],
            title=f"Origination share by {demographic.replace('applicant_', '').replace('_', ' ')}",
        )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No originated or denied applications in the current filters.")
    st.caption(
        "Demographic fields are used here for descriptive fairness review only and are excluded "
        "from the predictive model. Small groups and unknown/missing categories require caution."
    )

with tab_model:
    st.subheader("Predictive Modeling")
    st.write(
        "Target: originated (1) versus denied (0). Applicant race, ethnicity, "
        "sex, and age are excluded from the predictive model. This is an "
        "educational exercise, not a credit decision system."
    )

    if source == "2024 California HMDA data":
        st.warning(
            "DTI information has substantial missingness. The model uses the "
            "reported DTI category when available, rather than treating "
            "approximate category midpoints as exact values. Review the DTI "
            "sensitivity analysis when interpreting the results."
        )

    # ============================================================
    # 1. MODEL PERFORMANCE
    # ============================================================
    with st.expander("1. Model Performance", expanded=True):
        st.markdown("#### Held-out test-set evaluation")
        st.caption(
            "Train two models and compare their predictions on a held-out "
            "test set. The positive class is originated; the negative class is denied."
        )

        if st.button("Train and evaluate models", type="primary"):
            try:
                with st.spinner("Training and evaluating models..."):
                    results = train_and_evaluate(filtered)

                st.session_state["model_results"] = results
                st.session_state["model_results_signature"] = current_model_signature

            except Exception as exc:
                st.error(f"Model training failed: {exc}")

        results = st.session_state.get("model_results")
        saved_signature = st.session_state.get("model_results_signature")

        if results is None:
            st.info("Click the button above to train both models.")

        elif saved_signature != current_model_signature:
            st.warning(
                "Your dataset or filters have changed. Train the models again "
                "to refresh the results."
            )

        else:
            rows = []

            for name, result in results.items():
                rows.append(
                    {
                        "Model": name.replace("_", " ").title(),
                        **result["metrics"],
                        "Test rows": result["test_rows"],
                    }
                )

            metrics_df = pd.DataFrame(rows)

            st.markdown("#### Model comparison")
            st.dataframe(
                metrics_df.round(4),
                use_container_width=True,
                hide_index=True,
            )

            metric_chart = metrics_df.melt(
                id_vars="Model",
                value_vars=[
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                ],
                var_name="Metric",
                value_name="Score",
            )

            st.plotly_chart(
                px.bar(
                    metric_chart,
                    x="Model",
                    y="Score",
                    color="Metric",
                    barmode="group",
                    title="Held-out Test-Set Metrics",
                    range_y=[0, 1],
                ),
                use_container_width=True,
            )

            st.markdown("#### Confusion matrices")

            for name, result in results.items():
                st.markdown(
                    f"**{name.replace('_', ' ').title()}** "
                    "(actual rows × predicted columns)"
                )

                confusion_df = pd.DataFrame(
                    result["confusion_matrix"],
                    index=["Actual denied", "Actual originated"],
                    columns=["Predicted denied", "Predicted originated"],
                )

                st.dataframe(
                    confusion_df,
                    use_container_width=False,
                )

            st.caption(
                "Accuracy alone can hide errors in the minority class. "
                "Consider balanced accuracy, precision, recall, specificity, "
                "and the confusion matrices together."
            )

        st.divider()

        # --------------------------------------------------------
        # FIVE-FOLD CROSS-VALIDATION
        # --------------------------------------------------------
        st.markdown("#### Five-Fold Cross-Validation")

        st.write(
            "Cross-validation evaluates each model across five stratified "
            "folds. Results show the mean and standard deviation across folds."
        )

        if st.button(
            "Run cross-validation",
            key="run_cross_validation",
        ):
            try:
                with st.spinner("Evaluating models across five folds..."):
                    cv_results = cross_validate_models(
                        filtered,
                        n_splits=5,
                    )

                st.session_state["cv_results"] = cv_results
                st.session_state["cv_results_signature"] = current_model_signature

            except Exception as exc:
                st.error(f"Cross-validation failed: {exc}")

        cv_results = st.session_state.get("cv_results")
        cv_signature = st.session_state.get("cv_results_signature")

        if cv_results is None:
            st.info("Run cross-validation to compare performance across folds.")

        elif cv_signature != current_model_signature:
            st.warning(
                "Your dataset or filters have changed. Run cross-validation "
                "again to refresh these results."
            )

        else:
            cv_rows = []

            for model_name, metric_results in cv_results.items():
                row = {"Model": model_name.replace("_", " ").title()}

                for metric_name, values in metric_results.items():
                    row[f"{metric_name}_mean"] = values["mean"]
                    row[f"{metric_name}_std"] = values["std"]

                cv_rows.append(row)

            cv_df = pd.DataFrame(cv_rows)

            st.dataframe(
                cv_df.round(4),
                use_container_width=True,
                hide_index=True,
            )

            plot_metric = st.selectbox(
                "Cross-validation metric to compare",
                [
                    "balanced_accuracy",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                ],
                format_func=lambda value: value.replace("_", " ").title(),
                key="cv_metric",
            )

            cv_chart_df = pd.DataFrame(
                [
                    {
                        "Model": model_name.replace("_", " ").title(),
                        "Mean": values[plot_metric]["mean"],
                        "Std. Dev.": values[plot_metric]["std"],
                    }
                    for model_name, values in cv_results.items()
                ]
            )

            st.plotly_chart(
                px.bar(
                    cv_chart_df,
                    x="Model",
                    y="Mean",
                    error_y="Std. Dev.",
                    title=(
                        "Five-Fold Cross-Validation: "
                        f"{plot_metric.replace('_', ' ').title()}"
                    ),
                    range_y=[0, 1],
                ),
                use_container_width=True,
            )

            st.caption(
                "Error bars represent one standard deviation across folds. "
                "Cross-validation and held-out test-set results are different "
                "evaluations and should be interpreted separately."
            )

    # ============================================================
    # 2. DECISION THRESHOLD ANALYSIS
    # ============================================================
    with st.expander(
        "2. Decision Threshold Analysis",
        expanded=False,
    ):
        st.write(
            "Explore how changing the probability threshold affects "
            "classification metrics and the types of errors made. "
            "This is retrospective analysis, not a lending decision tool."
        )

        threshold_model_name = st.selectbox(
            "Choose model",
            ["logistic", "random_forest"],
            format_func=lambda name: name.replace("_", " ").title(),
            key="threshold_model",
        )

        results = st.session_state.get("model_results")
        saved_signature = st.session_state.get("model_results_signature")

        if results is None:
            st.info("Train the models in Model Performance first.")

        elif saved_signature != current_model_signature:
            st.info(
                "The dataset or filters have changed. Retrain the models "
                "before viewing threshold results."
            )

        else:
            threshold_result = results[threshold_model_name]["test_results"]

            y_true = np.array(threshold_result["y_true"])
            probabilities = np.array(threshold_result["probabilities"])

            threshold_rows = []

            for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
                predictions = (probabilities >= threshold).astype(int)

                tn, fp, fn, tp = confusion_matrix(
                    y_true,
                    predictions,
                    labels=[0, 1],
                ).ravel()

                threshold_rows.append(
                    {
                        "threshold": threshold,
                        "accuracy": accuracy_score(y_true, predictions),
                        "balanced_accuracy": balanced_accuracy_score(
                            y_true, predictions
                        ),
                        "precision": precision_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        ),
                        "recall": recall_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        ),
                        "specificity": (tn / (tn + fp) if (tn + fp) else 0.0),
                        "f1": f1_score(
                            y_true,
                            predictions,
                            zero_division=0,
                        ),
                        "False positives": fp,
                        "False negatives": fn,
                    }
                )

            threshold_df = pd.DataFrame(threshold_rows)

            st.dataframe(
                threshold_df.round(4),
                use_container_width=True,
                hide_index=True,
            )

            threshold_chart_df = threshold_df.melt(
                id_vars="threshold",
                value_vars=[
                    "balanced_accuracy",
                    "precision",
                    "recall",
                    "specificity",
                    "f1",
                ],
                var_name="Metric",
                value_name="Score",
            )

            st.plotly_chart(
                px.line(
                    threshold_chart_df,
                    x="threshold",
                    y="Score",
                    color="Metric",
                    markers=True,
                    title="Performance at Different Thresholds",
                    labels={
                        "threshold": "Probability threshold",
                    },
                ),
                use_container_width=True,
            )

            st.caption(
                "False positives are denied applications predicted as originated. "
                "False negatives are originated applications predicted as denied."
            )

    # ============================================================
    # 3. MODEL DIAGNOSTICS
    # ============================================================
    with st.expander(
        "3. Model Diagnostics",
        expanded=False,
    ):
        # --------------------------------------------------------
        # DTI SENSITIVITY ANALYSIS
        # --------------------------------------------------------
        st.markdown("#### DTI Sensitivity Analysis")

        st.write(
            "Compare five-fold cross-validation performance with and "
            "without debt-to-income (DTI) information. This helps assess "
            "how model performance changes when DTI is excluded."
        )

        if st.button(
            "Run DTI sensitivity analysis",
            key="run_dti_sensitivity",
        ):
            try:
                with st.spinner("Comparing models with and without DTI..."):
                    dti_results = compare_dti_sensitivity(
                        filtered,
                        n_splits=5,
                    )

                st.session_state["dti_results"] = dti_results
                st.session_state["dti_results_signature"] = current_model_signature

            except Exception as exc:
                st.error(f"DTI sensitivity analysis failed: {exc}")

        dti_results = st.session_state.get("dti_results")
        dti_signature = st.session_state.get("dti_results_signature")

        if dti_results is None:
            st.info("Run DTI sensitivity analysis to compare both configurations.")

        elif dti_signature != current_model_signature:
            st.info(
                "The dataset or filters have changed. Rerun the DTI "
                "sensitivity analysis to refresh the results."
            )

        else:
            dti_rows = []

            for configuration, model_results in dti_results.items():
                for model_name, model_metrics in model_results.items():
                    for metric in [
                        "balanced_accuracy",
                        "roc_auc",
                        "precision",
                        "recall",
                        "f1",
                    ]:
                        dti_rows.append(
                            {
                                "Configuration": configuration,
                                "Model": model_name.replace("_", " ").title(),
                                "Metric": metric.replace("_", " ").title(),
                                "Mean": model_metrics[metric]["mean"],
                                "Std. Dev.": model_metrics[metric]["std"],
                            }
                        )

            dti_table = pd.DataFrame(dti_rows)

            st.dataframe(
                dti_table.round(4),
                use_container_width=True,
                hide_index=True,
            )

            selected_dti_metric = st.selectbox(
                "DTI comparison metric",
                [
                    "balanced_accuracy",
                    "roc_auc",
                    "precision",
                    "recall",
                    "f1",
                ],
                format_func=lambda value: value.replace("_", " ").title(),
                key="dti_sensitivity_metric",
            )

            dti_chart_rows = []

            for configuration, model_results in dti_results.items():
                for model_name, model_metrics in model_results.items():
                    dti_chart_rows.append(
                        {
                            "Configuration": configuration,
                            "Model": model_name.replace("_", " ").title(),
                            "Mean score": model_metrics[selected_dti_metric]["mean"],
                            "Standard deviation": model_metrics[selected_dti_metric][
                                "std"
                            ],
                        }
                    )

            st.plotly_chart(
                px.bar(
                    pd.DataFrame(dti_chart_rows),
                    x="Model",
                    y="Mean score",
                    color="Configuration",
                    barmode="group",
                    error_y="Standard deviation",
                    title=(
                        f"{selected_dti_metric.replace('_', ' ').title()}: "
                        "With vs. Without DTI"
                    ),
                    range_y=[0, 1],
                ),
                use_container_width=True,
            )

            st.caption(
                "Bars show mean cross-validation scores; error bars show "
                "standard deviation across folds. These results describe "
                "historical application outcomes, not creditworthiness or fairness."
            )

        st.divider()

        # --------------------------------------------------------
        # MODEL ERROR ANALYSIS
        # --------------------------------------------------------
        st.markdown("#### Model Error Analysis")

        st.write(
            "Inspect individual test-set predictions to understand where "
            "each model makes mistakes."
        )

        error_model_name = st.selectbox(
            "Choose model for error analysis",
            ["logistic", "random_forest"],
            format_func=lambda name: name.replace("_", " ").title(),
            key="error_analysis_model",
        )

        saved_results = st.session_state.get("model_results")
        saved_signature = st.session_state.get("model_results_signature")

        if saved_results is None:
            st.info("Train the models in Model Performance first.")

        elif saved_signature != current_model_signature:
            st.info(
                "The dataset or filters have changed. Retrain the models "
                "before inspecting errors."
            )

        else:
            model_result = saved_results[error_model_name]
            test_results = model_result["test_results"]

            y_true_error = np.array(test_results["y_true"])
            y_pred_error = np.array(test_results["y_pred"])
            probabilities_error = np.array(test_results["probabilities"])
            test_indices = test_results["test_indices"]

            error_df = filtered.loc[test_indices].copy()

            error_df["Actual Outcome"] = np.where(
                y_true_error == 1,
                "Originated",
                "Denied",
            )

            error_df["Predicted Outcome"] = np.where(
                y_pred_error == 1,
                "Originated",
                "Denied",
            )

            error_df["Predicted Origination Probability"] = probabilities_error

            error_df["Error Type"] = np.select(
                [
                    (y_true_error == 0) & (y_pred_error == 1),
                    (y_true_error == 1) & (y_pred_error == 0),
                ],
                [
                    "False Positive",
                    "False Negative",
                ],
                default="Correct Prediction",
            )

            total_errors = int((error_df["Error Type"] != "Correct Prediction").sum())
            false_positives = int((error_df["Error Type"] == "False Positive").sum())
            false_negatives = int((error_df["Error Type"] == "False Negative").sum())

            metric1, metric2, metric3 = st.columns(3)
            metric1.metric(
                "Misclassified records",
                f"{total_errors:,}",
            )
            metric2.metric(
                "False positives",
                f"{false_positives:,}",
            )
            metric3.metric(
                "False negatives",
                f"{false_negatives:,}",
            )

            error_filter = st.selectbox(
                "Filter predictions",
                [
                    "All test records",
                    "Misclassified only",
                    "False positives",
                    "False negatives",
                    "Correct predictions",
                ],
                key="error_type_filter",
            )

            if error_filter == "Misclassified only":
                display_df = error_df[error_df["Error Type"] != "Correct Prediction"]

            elif error_filter == "False positives":
                display_df = error_df[error_df["Error Type"] == "False Positive"]

            elif error_filter == "False negatives":
                display_df = error_df[error_df["Error Type"] == "False Negative"]

            elif error_filter == "Correct predictions":
                display_df = error_df[error_df["Error Type"] == "Correct Prediction"]

            else:
                display_df = error_df

            preferred_columns = [
                "application_id",
                "action_taken",
                "Actual Outcome",
                "Predicted Outcome",
                "Predicted Origination Probability",
                "Error Type",
                "loan_purpose",
                "loan_type",
                "income_thousands",
                "loan_amount_thousands",
                "debt_to_income_category",
                "debt_to_income_ratio",
                "loan_to_value_ratio",
            ]

            visible_columns = [
                column for column in preferred_columns if column in display_df.columns
            ]

            st.dataframe(
                display_df[visible_columns],
                use_container_width=True,
                hide_index=True,
            )

            misclassified_df = error_df[error_df["Error Type"] != "Correct Prediction"]

            st.download_button(
                "Download misclassified records as CSV",
                data=misclassified_df.to_csv(index=False).encode("utf-8"),
                file_name=(f"mortgagelens_{error_model_name}_errors.csv"),
                mime="text/csv",
            )

            st.caption(
                "Predicted probabilities are model estimates, not verified "
                "probabilities of repayment or default. Error types describe "
                "predictions against historical originated-versus-denied labels."
            )

with tab_sql:
    st.subheader("SQL-powered summaries (DuckDB)")
    try:
        summaries = run_sql_summaries(filtered)
        st.markdown("**Origination share by state**")
        st.dataframe(
            summaries["approval_by_state"], use_container_width=True, hide_index=True
        )
        st.markdown("**Annual trend with prior-year comparison (LAG window function)**")
        st.dataframe(
            summaries["annual_trend"], use_container_width=True, hide_index=True
        )
    except Exception as exc:
        st.error(f"Could not run SQL summaries: {exc}")

with tab_data:
    st.subheader("Data quality checks")

    # Assess the original source before cleaning.
    raw_quality = summarize_data_quality(raw)

    st.markdown("**Original source data — before cleaning**")
    st.caption(
        "These checks cover the full loaded source dataset, before "
        "duplicate removal, missing-category replacement, and filters."
    )

    q1, q2, q3, q4 = st.columns(4)

    q1.metric(
        "Source rows",
        f"{raw_quality['row_count']:,}",
    )
    q2.metric(
        "Duplicate application IDs",
        f"{raw_quality['duplicate_application_ids']:,}",
    )
    q3.metric(
        "Missing application IDs",
        f"{raw_quality['missing_application_ids']:,}",
    )
    q4.metric(
        "Invalid numeric values",
        f"{int(raw_quality['invalid_numeric']['invalid_numeric_count'].sum()):,}",
    )

    st.markdown("**Original missingness by field**")
    st.dataframe(
        raw_quality["missingness"],
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("**Invalid numeric values by field**")
    st.dataframe(
        raw_quality["invalid_numeric"],
        use_container_width=True,
        hide_index=True,
    )

    st.divider()
    st.markdown("**Remaining missingness in cleaned, filtered data**")

    missing_cols = [
        c
        for c in [
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
        if c in filtered.columns
    ]
    missingness = pd.DataFrame(
        {
            "field": missing_cols,
            "missing_count": [int(filtered[c].isna().sum()) for c in missing_cols],
            "missing_percent": [
                float(filtered[c].isna().mean() * 100) for c in missing_cols
            ],
        }
    ).sort_values("missing_percent", ascending=False)

    st.markdown("**Missingness by field**")
    st.dataframe(missingness, use_container_width=True, hide_index=True)
    if not missingness.empty:
        st.plotly_chart(
            px.bar(
                missingness,
                x="field",
                y="missing_percent",
                title="Missing values by field (%)",
                labels={"field": "Field", "missing_percent": "Missing (%)"},
            ),
            use_container_width=True,
        )
    st.markdown("**Data preview (first 500 rows)**")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button(
        "Download filtered data as CSV",
        filtered.to_csv(index=False).encode("utf-8"),
        file_name="mortgagelens_filtered.csv",
        mime="text/csv",
    )


with tab_report:
    st.subheader("Generate portfolio-ready reports")
    st.write(
        "Create a concise one-page portfolio summary and a detailed technical report "
        "from the currently selected dataset and filters. Reports include model results "
        "and cross-validation only when those analyses have been run for the current filters."
    )
    st.caption(
        "HTML is suitable for viewing in a browser or printing to PDF. PDF files are generated directly. "
        "Both formats include limitations and responsible-use notes."
    )

    current_results = st.session_state.get("model_results")
    if st.session_state.get("model_results_signature") != current_model_signature:
        current_results = None
    current_cv = st.session_state.get("cv_results")
    if st.session_state.get("cv_results_signature") != current_model_signature:
        current_cv = None

    report_context = build_report_context(
        filtered=filtered,
        source=source,
        source_caption=source_caption,
        model_results=current_results,
        cv_results=current_cv,
    )

    if current_results is None:
        st.info(
            "For the strongest report, open Model Evaluation and click "
            "'Train and evaluate models' first. You can still generate reports without model results."
        )
    if current_cv is None:
        st.caption(
            "Five-fold cross-validation will appear in the detailed report after you run it."
        )

    # Generate each report once so the view links and download buttons use identical content.
    summary_pdf = generate_pdf(report_context, detailed=False)
    summary_html = generate_html(report_context, detailed=False)
    technical_pdf = generate_pdf(report_context, detailed=True)
    technical_html = generate_html(report_context, detailed=True)

    one_col, detail_col = st.columns(2)
    with one_col:
        st.markdown("**One-page portfolio report**")
        st.write(
            "Project objective, dataset, methods, current model metrics, and key limitations."
        )
        st.download_button(
            "Download one-page report · PDF",
            data=summary_pdf,
            file_name="mortgagelens_portfolio_summary.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "Download one-page report · HTML",
            data=summary_html,
            file_name="mortgagelens_portfolio_summary.html",
            mime="text/html",
            use_container_width=True,
        )
    with detail_col:
        st.markdown("**Detailed technical report**")
        st.write(
            "Outcome counts, missingness, demographic comparisons, model metrics, confusion matrices, cross-validation, and SQL summaries."
        )
        st.download_button(
            "Download technical report · PDF",
            data=technical_pdf,
            file_name="mortgagelens_technical_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
        st.download_button(
            "Download technical report · HTML",
            data=technical_html,
            file_name="mortgagelens_technical_report.html",
            mime="text/html",
            use_container_width=True,
        )

    st.markdown("**View report here**")
    preview_kind = st.radio(
        "Report format",
        ["One-page HTML", "Detailed HTML", "One-page PDF", "Detailed PDF"],
        horizontal=True,
        key="report_preview_kind",
    )
    preview_map = {
        "One-page HTML": (summary_html, "text/html"),
        "Detailed HTML": (technical_html, "text/html"),
        "One-page PDF": (summary_pdf, "application/pdf"),
        "Detailed PDF": (technical_pdf, "application/pdf"),
    }
    preview_bytes, preview_mime = preview_map[preview_kind]
    if preview_mime == "text/html":
        # Render HTML directly in the Streamlit page instead of opening a data: URL.
        st.components.v1.html(
            preview_bytes.decode("utf-8"),
            height=850,
            scrolling=True,
        )
    else:
        # Browser PDF plugins often block embedded data: URLs in Streamlit.
        # Render each PDF page to a PNG and show it directly in the app instead.
        try:
            pdf_document = fitz.open(stream=preview_bytes, filetype="pdf")
            st.caption(
                f"PDF preview · {pdf_document.page_count} page(s). "
                "Use the download button above if you need the original PDF file."
            )
            for page_number, page in enumerate(pdf_document, start=1):
                pixmap = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
                page_png = pixmap.tobytes("png")
                st.image(page_png, caption=f"Page {page_number}", use_container_width=True)
            pdf_document.close()
        except Exception as exc:
            st.error(f"Could not preview the PDF: {exc}")
            st.info(
                "The PDF download button above still provides the original report. "
                "Make sure PyMuPDF is installed (`pip install pymupdf`)."
            )

    st.divider()
    st.markdown("**Report preview**")
    report_share = (
        f"{report_context['origination_share']:.1%}"
        if report_context["origination_share"] is not None
        else "N/A"
    )
    st.markdown(
        f"- **Source:** {source}\n"
        f"- **Filtered records:** {len(filtered):,}\n"
        f"- **Originated + denied:** {report_context['eligible']:,}\n"
        f"- **Origination share:** {report_share}"
    )
