# MortgageLens: Mortgage Credit Risk & Fair Lending Analytics

A portfolio project demonstrating Python, Pandas, NumPy, SQL, statistics, machine learning, model evaluation, Plotly, Streamlit, and automated testing.

## Important data and interpretation note

MortgageLens supports a bundled synthetic demonstration dataset, a prepared public HMDA dataset when supplied in the expected `data/processed/` location, and analysis-ready CSV uploads. Synthetic records are not real borrower data or evidence about any actual lender or demographic group. Public HMDA records describe reported application outcomes and should not be interpreted as creditworthiness or loan-performance labels.

HMDA public data: https://www.consumerfinance.gov/data-research/hmda/
HMDA data field definitions: https://ffiec.cfpb.gov/documentation/publications/loan-level-datasets/lar-data-fields/

This project is educational, not a credit underwriting tool, legal opinion, or finding of discrimination. Group-level differences are descriptive signals for further review, not proof of discrimination. The predictive model is for demonstrating workflow only and must not be used to make lending decisions.

## Skills demonstrated

- Data validation, missing-value handling, duplicates, and feature engineering
- EDA, descriptive statistics, confidence intervals, and a two-proportion z-test
- SQL aggregations and window-function examples using DuckDB
- Logistic regression and random forest comparison
- Stratified train/test split, precision, recall, F1, ROC-AUC, confusion matrix
- Plotly visualizations and a multi-page-style Streamlit dashboard
- Unit tests with pytest and reproducible local setup

## 1. Create an environment

Python 3.11 or 3.12 is recommended.

### Windows PowerShell
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### macOS / Linux
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Run tests

From the project root:
```bash
pytest -q
```

## 3. Run the dashboard

```bash
streamlit run app.py
```
The app opens in your browser. Use the sidebar to filter the sample data.

## 4. Project structure

```text
MortgageLens/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── mortgage_applications_sample.csv
├── src/
│   ├── __init__.py
│   ├── data_prep.py
│   ├── analysis.py
│   ├── modeling.py
│   └── sql_analysis.py
├── sql/
│   └── analysis_queries.sql
└── tests/
    ├── test_data_prep.py
    ├── test_analysis.py
    └── test_modeling.py
```

## 5. Dataset schema

The bundled synthetic sample uses analysis-friendly column names:
- `application_id`: synthetic row identifier
- `application_year`: application year
- `state`: state abbreviation
- `loan_purpose`: Home purchase, Refinance, Home improvement
- `loan_type`: Conventional, FHA, VA, USDA
- `applicant_race`: illustrative group label
- `applicant_ethnicity`: illustrative group label
- `applicant_sex`: illustrative group label
- `applicant_age_band`: age band
- `income_thousands`: annual income in thousands of dollars
- `loan_amount_thousands`: requested loan amount in thousands of dollars
- `property_value_thousands`: property value in thousands of dollars
- `debt_to_income_ratio`: debt-to-income ratio percentage
- `loan_to_value_ratio`: loan-to-value ratio percentage
- `interest_rate`: rate for originated loans; missing for other outcomes
- `action_taken`: Originated, Approved not accepted, Denied, Withdrawn, Incomplete
- `denial_reason`: simplified illustrative reason
- `approved`: binary target for a narrowly defined modeling exercise

## 6. Using public HMDA data

Download a manageable dataset from the official HMDA Data Browser or a specific institution/state/year. Nationwide files can be very large and may not fit comfortably in memory. The current synthetic sample is **not a substitute for actual HMDA analysis**.

HMDA's original `action_taken` codes commonly include:
1 originated, 2 approved but not accepted, 3 denied, 4 withdrawn, 5 file closed for incompleteness, 6 purchased, 7 preapproval denied, 8 preapproval approved but not accepted. See the official data field documentation.

For a fair comparison, define the population before calculating rates. This project defines its demo approval rate as `Originated / (Originated + Denied)` and excludes other dispositions from that denominator. Do not silently treat withdrawn/incomplete applications as denials.

For actual HMDA data, first map the downloaded field names and codes to this project's schema, document the filters, and inspect year-specific definitions. Do not combine incompatible reporting years without checking schema changes.

## 7. Portfolio extensions

1. Add year-over-year trend analysis and confidence intervals.
2. Add lender-level filtering and small-sample warnings.
3. Compare group differences after stratifying by loan purpose, loan type, income, and geography.
4. Add model calibration and threshold analysis.
5. Add an explicit model card and a data dictionary for every imported field.
6. Deploy on Streamlit Community Cloud after pushing the repository to GitHub.

## Responsible use

- Do not use this model for actual credit decisions.
- Do not infer individual creditworthiness from demographic attributes.
- Do not claim a group difference proves discrimination.
- Check sample sizes, missingness, confounding, data quality, and legal context.
- For regulated analysis, consult appropriate fair-lending and legal experts.


## 5. Automatic portfolio reports

Open the **Portfolio Report** tab in the Streamlit dashboard to download four report variants based on the currently selected data source and filters:

- One-page portfolio summary: PDF or standalone HTML
- Detailed technical report: PDF or standalone HTML

The detailed report includes action-outcome counts, remaining missingness, demographic group comparisons, available model metrics and confusion matrices, cross-validation results if run, and DuckDB SQL summaries. To include model results, first run **Train and evaluate models** in the Model Evaluation tab; run **Run cross-validation** to include cross-validation. If filters change, rerun the analyses before generating reports. Reports explicitly state that the model predicts recorded originated-versus-denied outcomes, not creditworthiness or default, and that descriptive group differences do not establish discrimination.

The report files are generated from the current dashboard state; metrics are not hard-coded. Install the dependencies in `requirements.txt` after applying this update.
