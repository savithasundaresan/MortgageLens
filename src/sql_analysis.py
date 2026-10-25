from __future__ import annotations
import duckdb
import pandas as pd


def run_sql_summaries(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Run reproducible SQL summaries against an in-memory DuckDB relation."""
    con = duckdb.connect(database=":memory:")
    con.register("applications", df)
    approval_by_state = con.execute("""
        SELECT state,
               COUNT(*) FILTER (WHERE action_taken IN ('Originated','Denied')) AS eligible_applications,
               COUNT(*) FILTER (WHERE action_taken = 'Originated') AS originated,
               COUNT(*) FILTER (WHERE action_taken = 'Denied') AS denied,
               COUNT(*) FILTER (WHERE action_taken = 'Originated') * 1.0 /
                   NULLIF(COUNT(*) FILTER (
                       WHERE action_taken IN ('Originated','Denied')), 0) AS origination_share
        FROM applications
        GROUP BY state
        ORDER BY eligible_applications DESC
    """).df()
    annual = con.execute("""
        WITH yearly AS (
            SELECT application_year,
                   COUNT(*) FILTER (WHERE action_taken IN ('Originated','Denied')) AS eligible,
                   COUNT(*) FILTER (WHERE action_taken = 'Originated') AS originated
            FROM applications
            GROUP BY application_year
        )
        SELECT application_year, eligible, originated,
               originated * 1.0 / NULLIF(eligible, 0) AS origination_share,
               LAG(originated * 1.0 / NULLIF(eligible, 0))
                   OVER (ORDER BY application_year) AS prior_year_rate
        FROM yearly ORDER BY application_year
    """).df()
    con.close()
    return {"approval_by_state": approval_by_state, "annual_trend": annual}
