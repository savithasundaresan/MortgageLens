from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from typing import Any

import pandas as pd


def _fmt(value: Any, key: str = "") -> str:
    if value is None:
        return "Not run"
    try:
        if pd.isna(value):
            return "N/A"
    except (TypeError, ValueError):
        pass
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        metric_keys = {"origination_share", "rate_a", "rate_b", "difference", "accuracy",
                       "balanced_accuracy", "precision", "recall", "specificity", "f1", "roc_auc"}
        if key in metric_keys or any(key.startswith(metric + " ") for metric in metric_keys - {"difference"}):
            return f"{value:.2%}" if key != "difference" else f"{value:+.2%}"
        if key == "missing_percent":
            return f"{value:.1f}%"
        return f"{value:,.4f}" if isinstance(value, float) else f"{value:,}"
    return str(value)


def _table_html(df: pd.DataFrame | None, max_rows: int = 30) -> str:
    if df is None or df.empty:
        return "<p><em>No results available for this section.</em></p>"
    shown = df.head(max_rows).copy()
    headers = "".join(f"<th>{escape(str(c))}</th>" for c in shown.columns)
    rows = []
    for _, row in shown.iterrows():
        cells = "".join(f"<td>{escape(_fmt(v, str(c)))}</td>" for c, v in row.items())
        rows.append(f"<tr>{cells}</tr>")
    suffix = f"<p><em>Showing {len(shown)} of {len(df)} rows.</em></p>" if len(df) > max_rows else ""
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>{suffix}"


def build_report_context(
    filtered: pd.DataFrame,
    source: str,
    source_caption: str,
    model_results: dict | None = None,
    cv_results: dict | None = None,
) -> dict:
    eligible = filtered[filtered["action_taken"].isin(["Originated", "Denied"])]
    originated = int((eligible["action_taken"] == "Originated").sum())
    denied = int((eligible["action_taken"] == "Denied").sum())
    n_eligible = len(eligible)
    median_loan = filtered["loan_amount_thousands"].median()
    outcome_counts = (
        filtered["action_taken"].value_counts(dropna=False)
        .rename_axis("Action outcome").reset_index(name="Records")
    )
    demographics = {}
    from src.analysis import approval_summary
    for col in ["applicant_race", "applicant_ethnicity", "applicant_sex", "applicant_age_band"]:
        if col in filtered.columns:
            demographics[col.replace("applicant_", "").replace("_", " ").title()] = (
                approval_summary(filtered, col).sort_values("origination_share", ascending=False)
            )
    try:
        from src.sql_analysis import run_sql_summaries
        sql_summaries = run_sql_summaries(filtered)
    except Exception:
        sql_summaries = {}
    missing_cols = [c for c in [
        "income_thousands", "loan_amount_thousands", "property_value_thousands",
        "debt_to_income_ratio", "loan_to_value_ratio", "interest_rate",
        "applicant_race", "applicant_ethnicity", "applicant_sex", "applicant_age_band"
    ] if c in filtered.columns]
    missingness = pd.DataFrame({
        "Field": missing_cols,
        "Missing count": [int(filtered[c].isna().sum()) for c in missing_cols],
        "Missing percent": [float(filtered[c].isna().mean() * 100) for c in missing_cols],
    }).sort_values("Missing percent", ascending=False)
    model_rows = []
    confusion_tables = {}
    if model_results:
        for name, result in model_results.items():
            model_rows.append({"Model": name.replace("_", " ").title(),
                               **result.get("metrics", {}),
                               "Test rows": result.get("test_rows", "")})
            cm = result.get("confusion_matrix")
            if cm is not None:
                confusion_tables[name.replace("_", " ").title()] = pd.DataFrame(
                    cm, index=["Actual denied", "Actual originated"],
                    columns=["Predicted denied", "Predicted originated"])
    model_table = pd.DataFrame(model_rows)
    cv_rows = []
    if cv_results:
        for name, metrics in cv_results.items():
            row = {"Model": name.replace("_", " ").title()}
            for metric, values in metrics.items():
                row[f"{metric} mean"] = values.get("mean")
                row[f"{metric} std"] = values.get("std")
            cv_rows.append(row)
    return {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "source": source,
        "source_caption": source_caption,
        "rows": len(filtered),
        "columns": len(filtered.columns),
        "eligible": n_eligible,
        "originated": originated,
        "denied": denied,
        "origination_share": originated / n_eligible if n_eligible else None,
        "median_loan": median_loan,
        "outcomes": outcome_counts,
        "demographics": demographics,
        "sql": sql_summaries,
        "missingness": missingness,
        "models": model_table,
        "confusion": confusion_tables,
        "cv": pd.DataFrame(cv_rows),
    }


def _report_html(ctx: dict, detailed: bool) -> str:
    share = "N/A" if ctx["origination_share"] is None else f'{ctx["origination_share"]:.1%}'
    median = "N/A" if pd.isna(ctx["median_loan"]) else f'${ctx["median_loan"]:,.0f}K'
    title = "MortgageLens — Technical Analysis Report" if detailed else "MortgageLens — Portfolio Summary"
    common = f"""
    <h1>{escape(title)}</h1>
    <p class="muted">Generated {escape(ctx['generated_at'])} · {escape(ctx['source'])}</p>
    <p>{escape(ctx['source_caption'])}</p>
    <div class="kpis">
      <div><strong>{ctx['rows']:,}</strong><span>Filtered records</span></div>
      <div><strong>{ctx['eligible']:,}</strong><span>Originated + denied</span></div>
      <div><strong>{share}</strong><span>Origination share*</span></div>
      <div><strong>{escape(median)}</strong><span>Median loan amount</span></div>
    </div>
    <h2>Objective and approach</h2>
    <p>Explore reported mortgage application outcomes, assess data quality, compare unadjusted group-level outcomes, and demonstrate classification and SQL analytics. The binary modeling task distinguishes recorded originated versus denied outcomes; it does not predict default or measure borrower creditworthiness.</p>
    """
    if not detailed:
        model = _table_html(ctx["models"]) if not ctx["models"].empty else "<p>Model evaluation has not been run for the current filters. Open Model Evaluation and run the models, then generate the report again.</p>"
        return f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title>
        <style>{_css()}</style></head><body class="onepage">{common}
        <h2>Methods</h2><p>Python, pandas, scikit-learn, Plotly and DuckDB; data validation and cleaning; exploratory summaries; unadjusted demographic comparisons; Logistic Regression and Random Forest with a held-out test set. Applicant race, ethnicity, sex and age are excluded from predictive features.</p>
        <h2>Model results</h2>{model}
        <h2>Interpretation and limitations</h2><p>*Origination share = originated / (originated + denied); other action outcomes are excluded. Group differences are descriptive, not proof of discrimination or causation. HMDA action outcomes reflect reported actions, not creditworthiness. Public HMDA data from a limited institution, geography, and period may not represent the whole lending market. Results depend on current dashboard filters.</p>
        <p class="muted">MortgageLens · Educational portfolio project · Not for lending decisions</p></body></html>"""
    sections = [
        "<h2>Action outcome distribution</h2>" + _table_html(ctx["outcomes"]),
        "<h2>Data quality: remaining missingness in filtered, cleaned data</h2>" + _table_html(ctx["missingness"]),
        "<h2>Demographic outcome comparisons</h2>" + "".join(
            f"<h3>{escape(name)}</h3>{_table_html(df)}" for name, df in ctx["demographics"].items()
        ),
        "<h2>Held-out model evaluation</h2>" + (
            _table_html(ctx["models"]) if not ctx["models"].empty else
            "<p>Model evaluation has not been run for the current filters.</p>"
        ),
        "<h2>Confusion matrices</h2>" + "".join(
            f"<h3>{escape(name)}</h3>{_table_html(df)}" for name, df in ctx["confusion"].items()
        ),
        "<h2>Cross-validation</h2>" + (
            _table_html(ctx["cv"]) if not ctx["cv"].empty else
            "<p>Cross-validation has not been run for the current filters.</p>"
        ),
        "<h2>SQL summaries — origination share by state</h2>" + _table_html(ctx["sql"].get("approval_by_state")),
        "<h2>SQL summaries — annual trend</h2>" + _table_html(ctx["sql"].get("annual_trend")),
        "<h2>Methods and interpretation</h2><p>Data cleaning validates required columns, removes duplicate application IDs, converts numeric fields, replaces missing categorical values with Unknown, and treats impossible negative amounts/rates as missing. The model compares Logistic Regression and Random Forest against a majority-class baseline on a held-out test set; five-fold stratified cross-validation is included only if run in the dashboard.</p><p>No inferential hypothesis test is calculated by this report. Demographic group comparisons are unadjusted descriptive statistics. Small groups, missingness, confounding variables, and institutional context limit interpretation.</p>",
        "<h2>Limitations and responsible use</h2><ul><li>The target is recorded originated versus denied, not loan repayment, default, or creditworthiness.</li><li>Applicant race, ethnicity, sex, and age are excluded from predictive features but shown for descriptive review only.</li><li>Differences in group-level outcomes do not establish discrimination, causation, or legal compliance.</li><li>Public HMDA data may cover a limited institution, geography, and period, which limits generalizability.</li><li>Where DTI is represented by category midpoints, it is approximate rather than an exact borrower-level ratio.</li><li>Results reflect the currently selected source and dashboard filters.</li></ul>",
    ]
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{escape(title)}</title><style>{_css()}</style></head><body>{common}{''.join(sections)}<p class="muted">MortgageLens · Educational portfolio project · Not for lending decisions</p></body></html>"""


def _css() -> str:
    return """body{font:14px Arial,sans-serif;line-height:1.45;color:#17202a;max-width:1000px;margin:30px auto;padding:0 24px}h1{font-size:26px;margin-bottom:4px}h2{font-size:18px;border-bottom:1px solid #ccd1d1;padding-bottom:5px;margin-top:24px}h3{font-size:15px}.muted{color:#566573;font-size:11px}.kpis{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin:18px 0}.kpis div{border:1px solid #d5d8dc;border-radius:6px;padding:12px}.kpis strong,.kpis span{display:block}.kpis strong{font-size:22px}.kpis span{font-size:11px;color:#566573}.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:11px;margin:10px 0}th,td{border:1px solid #d5d8dc;padding:5px 7px;text-align:left;vertical-align:top}th{background:#f2f4f4}.onepage{max-width:780px;font-size:12px}.onepage h1{font-size:24px}.onepage h2{font-size:15px;margin-top:15px}.onepage .kpis strong{font-size:18px}@media print{body{margin:10mm auto;padding:0}.onepage{font-size:10pt}.onepage h2{margin-top:10px}.kpis div{padding:6px}}"""


def generate_html(ctx: dict, detailed: bool = False) -> bytes:
    return _report_html(ctx, detailed).encode("utf-8")


def generate_pdf(ctx: dict, detailed: bool = False) -> bytes:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

    output = BytesIO()
    doc = SimpleDocTemplate(output, pagesize=letter, rightMargin=0.48*inch,
                            leftMargin=0.48*inch, topMargin=0.45*inch, bottomMargin=0.45*inch)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="SmallBody", parent=styles["BodyText"], fontSize=8, leading=10, spaceAfter=5))
    styles.add(ParagraphStyle(name="SmallHead", parent=styles["Heading2"], fontSize=11, leading=13, spaceBefore=8, spaceAfter=4))
    story = []
    title = "MortgageLens — Technical Analysis Report" if detailed else "MortgageLens — Portfolio Summary"
    story += [Paragraph(title, styles["Title"]), Paragraph(
        f"Generated {ctx['generated_at']} · {ctx['source']}", styles["SmallBody"]),
        Paragraph(ctx["source_caption"], styles["SmallBody"]), Spacer(1, 5)]
    share = "N/A" if ctx["origination_share"] is None else f'{ctx["origination_share"]:.1%}'
    median = "N/A" if pd.isna(ctx["median_loan"]) else f'${ctx["median_loan"]:,.0f}K'
    kpi = [["Filtered records", "Originated + denied", "Origination share*", "Median loan amount"],
           [f'{ctx["rows"]:,}', f'{ctx["eligible"]:,}', share, median]]
    t = Table(kpi, colWidths=[1.8*inch]*4)
    t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF0F6")),
                           ("GRID",(0,0),(-1,-1),0.4,colors.grey),
                           ("FONTNAME",(0,0),(-1,-1),"Helvetica"),
                           ("FONTSIZE",(0,0),(-1,-1),8),("VALIGN",(0,0),(-1,-1),"MIDDLE"),
                           ("BOTTOMPADDING",(0,0),(-1,-1),6),("TOPPADDING",(0,0),(-1,-1),6)]))
    story += [t, Spacer(1, 8), Paragraph("Objective and approach", styles["SmallHead"]),
              Paragraph("Explore reported mortgage application outcomes, assess data quality, compare unadjusted group-level outcomes, and demonstrate classification and SQL analytics. The binary modeling task distinguishes recorded originated versus denied outcomes; it does not predict default or measure borrower creditworthiness.", styles["SmallBody"])]
    def add_table(title_text, frame):
        story.append(Paragraph(title_text, styles["SmallHead"]))
        if frame is None or frame.empty:
            story.append(Paragraph("No results available for this section.", styles["SmallBody"]))
            return
        frame = frame.head(12 if not detailed else 24)
        data = [[Paragraph(escape(str(c)), styles["SmallBody"]) for c in frame.columns]]
        for _, row in frame.iterrows():
            data.append([Paragraph(escape(_fmt(v, str(c))), styles["SmallBody"]) for c, v in row.items()])
        avail = letter[0] - 0.96*inch
        col_width = avail / max(1, len(frame.columns))
        table = Table(data, colWidths=[col_width]*len(frame.columns), repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#EAF0F6")),
                                   ("GRID",(0,0),(-1,-1),0.3,colors.grey),
                                   ("VALIGN",(0,0),(-1,-1),"TOP"),
                                   ("LEFTPADDING",(0,0),(-1,-1),3),("RIGHTPADDING",(0,0),(-1,-1),3),
                                   ("TOPPADDING",(0,0),(-1,-1),3),("BOTTOMPADDING",(0,0),(-1,-1),3)]))
        story.append(table)
    if detailed:
        add_table("Action outcome distribution", ctx["outcomes"])
        add_table("Remaining missingness in filtered, cleaned data", ctx["missingness"])
        for name, frame in ctx["demographics"].items():
            add_table(f"Demographic comparison: {name}", frame)
        add_table("Held-out model evaluation", ctx["models"])
        for name, frame in ctx["confusion"].items():
            add_table(f"Confusion matrix: {name}", frame)
        add_table("Cross-validation", ctx["cv"])
        add_table("SQL summary: origination share by state", ctx["sql"].get("approval_by_state"))
        add_table("SQL summary: annual trend", ctx["sql"].get("annual_trend"))
        story += [Paragraph("Methods and limitations", styles["SmallHead"]),
                  Paragraph("Data cleaning validates required columns, removes duplicate application IDs, converts numeric fields, replaces missing categorical values with Unknown, and treats impossible negative amounts/rates as missing. Models compare Logistic Regression and Random Forest with a majority-class baseline. Cross-validation is included only if run in the dashboard. No inferential hypothesis test is calculated by this report; demographic comparisons are unadjusted descriptive statistics. The target is recorded originated versus denied, not default or creditworthiness. Group differences do not establish discrimination, causation, or legal compliance. Public HMDA data may cover a limited institution, geography, and period, which limits generalizability. DTI category midpoints, where used, are approximate. Results reflect current filters.", styles["SmallBody"])]
    else:
        story += [Paragraph("Methods", styles["SmallHead"]),
                  Paragraph("Python, pandas, scikit-learn, Plotly and DuckDB; data validation and cleaning; exploratory summaries; unadjusted demographic comparisons; Logistic Regression and Random Forest on a held-out test set. Applicant race, ethnicity, sex and age are excluded from predictive features.", styles["SmallBody"])]
        add_table("Held-out model results", ctx["models"])
        story += [Paragraph("Interpretation and limitations", styles["SmallHead"]),
                  Paragraph("*Origination share = originated / (originated + denied); other action outcomes are excluded. Group differences are descriptive, not proof of discrimination or causation. The target is not default or creditworthiness. Results depend on current dashboard filters and are not for lending decisions.", styles["SmallBody"])]
    story.append(Spacer(1, 6))
    story.append(Paragraph("MortgageLens · Educational portfolio project · Not for lending decisions", styles["SmallBody"]))
    doc.build(story)
    return output.getvalue()
