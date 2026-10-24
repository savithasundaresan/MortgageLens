from pathlib import Path
import pandas as pd
import plotly.express as px
import streamlit as st
from src.data_prep import clean_applications
from src.analysis import approval_summary, denial_reason_summary
from src.modeling import train_and_evaluate
from src.sql_analysis import run_sql_summaries

st.set_page_config(page_title="MortgageLens", page_icon="🏠", layout="wide")
st.title("🏠 MortgageLens")
st.caption("Mortgage Credit Risk & Fair Lending Analytics")
st.warning(
    "Educational portfolio project. The bundled data are synthetic. "
    "Descriptive group differences are not proof of discrimination. "
    "Do not use these models for actual lending decisions."
)

DEFAULT_DATA = Path(__file__).parent / "data" / "mortgage_applications_sample.csv"

@st.cache_data
def load_default():
    return pd.read_csv(DEFAULT_DATA)

uploaded = st.sidebar.file_uploader("Upload analysis-ready CSV", type=["csv"])
try:
    raw = pd.read_csv(uploaded) if uploaded is not None else load_default()
    df = clean_applications(raw)
except Exception as exc:
    st.error(f"Could not load this file: {exc}")
    st.stop()

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

eligible = filtered[filtered["action_taken"].isin(["Originated", "Denied"])]
n_eligible = len(eligible)
n_originated = int((eligible["action_taken"] == "Originated").sum())
n_denied = int((eligible["action_taken"] == "Denied").sum())
approval_rate = n_originated / n_eligible if n_eligible else 0
median_loan = filtered["loan_amount_thousands"].median()

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total records", f"{len(filtered):,}")
k2.metric("Approval rate*", f"{approval_rate:.1%}" if n_eligible else "N/A")
k3.metric("Denials*", f"{n_denied:,}")
k4.metric("Median requested loan", f"${median_loan:,.0f}K" if pd.notna(median_loan) else "N/A")
st.caption("*Approval rate denominator includes originated and denied applications only; withdrawals and incomplete files are excluded.")

tab_overview, tab_fair, tab_model, tab_sql, tab_data = st.tabs(
    ["Overview & EDA", "Fair Lending Review", "Model Evaluation", "SQL Analytics", "Data Quality & Export"]
)

with tab_overview:
    left, right = st.columns(2)
    with left:
        rate_df = approval_summary(filtered, "loan_purpose")
        fig = px.bar(rate_df, x="group", y="approval_rate", text="approval_rate",
                     title="Approval rate by loan purpose", labels={"group":"Loan purpose","approval_rate":"Approval rate"})
        fig.update_traces(texttemplate="%{text:.1%}", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        reason_df = denial_reason_summary(filtered)
        if not reason_df.empty:
            st.plotly_chart(px.bar(reason_df, x="denial_reason", y="count",
                                   title="Recorded denial reasons", labels={"count":"Applications","denial_reason":"Reason"}),
                            use_container_width=True)
        else:
            st.info("No denied applications in the current filters.")
    if filtered["interest_rate"].notna().any():
        st.plotly_chart(px.histogram(filtered.dropna(subset=["interest_rate"]), x="interest_rate",
                                     color="loan_type", nbins=25, title="Interest-rate distribution for records with a rate"),
                        use_container_width=True)
    st.plotly_chart(px.scatter(filtered.dropna(subset=["income_thousands", "loan_amount_thousands"]),
                               x="income_thousands", y="loan_amount_thousands", color="action_taken",
                               title="Income vs. requested loan amount",
                               labels={"income_thousands":"Income ($ thousands)", "loan_amount_thousands":"Loan amount ($ thousands)"}),
                    use_container_width=True)

with tab_fair:
    st.subheader("Descriptive group-level outcome comparisons")
    st.write("These are unadjusted descriptive comparisons. Differences may reflect many factors and require deeper analysis.")
    demographic = st.selectbox("Compare groups by", ["applicant_race", "applicant_ethnicity", "applicant_sex", "applicant_age_band"])
    fair = approval_summary(filtered, demographic)
    fair = fair.sort_values("approval_rate", ascending=False)
    fair["approval_rate_pct"] = fair["approval_rate"] * 100
    st.dataframe(fair, use_container_width=True, hide_index=True)
    fig = px.bar(fair, x="group", y="approval_rate", hover_data=["applications", "originated", "denied"],
                 title=f"Observed approval rate by {demographic.replace('applicant_', '').replace('_', ' ')}")
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Review group sample sizes, missing demographic data, loan purpose/type, income, geography, and other potential confounders before drawing conclusions.")

with tab_model:
    st.subheader("Predictive modeling demonstration")
    st.write("Target: originated (1) vs. denied (0). Features exclude applicant race, ethnicity, sex, and age. This is not a credit decision system.")
    if st.button("Train and evaluate models", type="primary"):
        try:
            results = train_and_evaluate(filtered)
            rows = []
            for name, result in results.items():
                rows.append({"model": name.replace("_", " ").title(), **result["metrics"], "test_rows": result["test_rows"]})
            metrics = pd.DataFrame(rows)
            st.dataframe(metrics, use_container_width=True, hide_index=True)
            st.plotly_chart(px.bar(metrics.melt(id_vars="model", value_vars=["precision", "recall", "f1", "roc_auc"]),
                                   x="model", y="value", color="variable", barmode="group",
                                   title="Model metric comparison", range_y=[0,1]), use_container_width=True)
            for name, result in results.items():
                st.markdown(f"**{name.replace('_',' ').title()} confusion matrix** (actual rows × predicted columns)")
                st.dataframe(pd.DataFrame(result["confusion_matrix"],
                                          index=["Actual denied", "Actual originated"],
                                          columns=["Predicted denied", "Predicted originated"]),
                             use_container_width=False)
            st.info("Metrics depend on the sample, target definition, features, split, and threshold. Accuracy alone is insufficient; review false positives and false negatives.")
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
    else:
        st.info("Click the button to train both models on the currently filtered data.")

with tab_sql:
    st.subheader("SQL-powered summaries (DuckDB)")
    summaries = run_sql_summaries(filtered)
    st.markdown("**Approval rate by state**")
    st.dataframe(summaries["approval_by_state"], use_container_width=True, hide_index=True)
    st.markdown("**Annual trend with prior-year comparison (LAG window function)**")
    st.dataframe(summaries["annual_trend"], use_container_width=True, hide_index=True)

with tab_data:
    st.subheader("Data quality checks")
    q1, q2, q3 = st.columns(3)
    q1.metric("Duplicate application IDs", int(filtered["application_id"].duplicated().sum()))
    q2.metric("Missing income", int(filtered["income_thousands"].isna().sum()))
    q3.metric("Missing interest rate", int(filtered["interest_rate"].isna().sum()))
    st.markdown("**Preview**")
    st.dataframe(filtered.head(500), use_container_width=True, hide_index=True)
    st.download_button("Download filtered data as CSV", filtered.to_csv(index=False).encode("utf-8"),
                       file_name="mortgagelens_filtered.csv", mime="text/csv")
