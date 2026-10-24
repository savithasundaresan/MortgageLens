from __future__ import annotations
import pandas as pd
from scipy.stats import norm

def approval_summary(df: pd.DataFrame, group_col: str | None = None) -> pd.DataFrame:
    """Approval rate among originated or denied applications only."""
    eligible = df[df["action_taken"].isin(["Originated", "Denied"])].copy()
    if group_col is None:
        groups = [("All applications", eligible)]
    else:
        groups = list(eligible.groupby(group_col, dropna=False))
    rows = []
    for label, group in groups:
        n = len(group)
        originated = int((group["action_taken"] == "Originated").sum())
        rate = originated / n if n else float("nan")
        rows.append({"group": label, "applications": n, "originated": originated,
                     "denied": int((group["action_taken"] == "Denied").sum()),
                     "approval_rate": rate})
    return pd.DataFrame(rows)

def two_proportion_z_test(success_a: int, total_a: int,
                          success_b: int, total_b: int) -> dict:
    """Two-sided pooled two-proportion z-test. Input counts must be valid."""
    if min(success_a, success_b) < 0 or min(total_a, total_b) <= 0:
        raise ValueError("Counts must be non-negative and totals must be positive.")
    if success_a > total_a or success_b > total_b:
        raise ValueError("Success counts cannot exceed totals.")
    p1, p2 = success_a / total_a, success_b / total_b
    pooled = (success_a + success_b) / (total_a + total_b)
    se = (pooled * (1 - pooled) * (1 / total_a + 1 / total_b)) ** 0.5
    if se == 0:
        return {"rate_a": p1, "rate_b": p2, "difference": p1-p2, "z_stat": 0.0, "p_value": 1.0}
    z = (p1 - p2) / se
    p = 2 * (1 - norm.cdf(abs(z)))
    return {"rate_a": p1, "rate_b": p2, "difference": p1-p2, "z_stat": z, "p_value": p}

def denial_reason_summary(df: pd.DataFrame) -> pd.DataFrame:
    denied = df[df["action_taken"] == "Denied"]
    if denied.empty:
        return pd.DataFrame(columns=["denial_reason", "count"])
    return (denied["denial_reason"].value_counts()
            .rename_axis("denial_reason").reset_index(name="count"))
