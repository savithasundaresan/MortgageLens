-- Run these queries in DuckDB after registering the cleaned DataFrame as `applications`.

-- 1. Approval and denial summary by state
SELECT state,
       COUNT(*) FILTER (WHERE action_taken IN ('Originated', 'Denied')) AS eligible,
       COUNT(*) FILTER (WHERE action_taken = 'Originated') AS originated,
       COUNT(*) FILTER (WHERE action_taken = 'Denied') AS denied,
       originated * 1.0 / NULLIF(eligible, 0) AS approval_rate
FROM (
    SELECT state, action_taken,
           COUNT(*) OVER (PARTITION BY state) AS state_rows
    FROM applications
) a
GROUP BY state
ORDER BY eligible DESC;

-- 2. Year-over-year rate using a CTE and window function
WITH yearly AS (
    SELECT application_year,
           COUNT(*) FILTER (WHERE action_taken IN ('Originated', 'Denied')) AS eligible,
           COUNT(*) FILTER (WHERE action_taken = 'Originated') AS originated
    FROM applications
    GROUP BY application_year
)
SELECT application_year, eligible,
       originated * 1.0 / NULLIF(eligible, 0) AS approval_rate,
       LAG(originated * 1.0 / NULLIF(eligible, 0))
           OVER (ORDER BY application_year) AS prior_year_rate
FROM yearly
ORDER BY application_year;
