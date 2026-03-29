-- =============================================================================
-- Exploration Queries: Data Profiling and Quality Assessment
-- =============================================================================
--
-- Purpose: validate the loaded data before doing real analysis. These queries
-- answer "what do we actually have?" and surface any quality issues that
-- could silently break downstream work.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- 1. Table-level row counts
-- ---------------------------------------------------------------------------
-- Quick sanity check after every load. If any table shows 0 rows something
-- went wrong in ingestion.

SELECT 'series_metadata' AS table_name, COUNT(*) AS row_count FROM series_metadata
UNION ALL
SELECT 'observations',                  COUNT(*)              FROM observations
UNION ALL
SELECT 'ai_insights',                   COUNT(*)              FROM ai_insights;


-- ---------------------------------------------------------------------------
-- 2. Observations per series with date coverage
-- ---------------------------------------------------------------------------
-- JOIN observations to series_metadata so we see human-readable names instead
-- of raw IDs. This is the first cross-table query and immediately tells us
-- whether any series loaded short.

SELECT
    sm.series_id,
    sm.name,
    sm.category,
    sm.frequency,
    COUNT(o.id)   AS obs_count,
    MIN(o.date)   AS earliest,
    MAX(o.date)   AS latest
FROM series_metadata sm
LEFT JOIN observations o ON sm.series_id = o.series_id
GROUP BY sm.series_id
ORDER BY sm.category, sm.series_id;


-- ---------------------------------------------------------------------------
-- 3. Monthly observation density (CTE + window function)
-- ---------------------------------------------------------------------------
-- For monthly series, we expect exactly one observation per month. This CTE
-- computes the month-over-month gap in days and flags anything outside the
-- 25-35 day range (normal monthly cadence). A gap larger than 35 days means
-- a missing month; smaller than 25 means a duplicate or sub-monthly artifact.
--
-- Uses: CTE, window function (LAG), JOIN, CASE expression.

WITH observation_gaps AS (
    SELECT
        o.series_id,
        o.date,
        o.value,
        LAG(o.date) OVER (PARTITION BY o.series_id ORDER BY o.date) AS prev_date,
        CAST(
            julianday(o.date) - julianday(
                LAG(o.date) OVER (PARTITION BY o.series_id ORDER BY o.date)
            ) AS INTEGER
        ) AS days_since_prev
    FROM observations o
    JOIN series_metadata sm ON o.series_id = sm.series_id
    WHERE sm.frequency = 'monthly'
)
SELECT
    series_id,
    COUNT(*) AS total_obs,
    SUM(CASE WHEN days_since_prev > 35  THEN 1 ELSE 0 END) AS gaps_over_35d,
    SUM(CASE WHEN days_since_prev < 25
              AND days_since_prev IS NOT NULL THEN 1 ELSE 0 END) AS gaps_under_25d,
    MIN(days_since_prev) AS min_gap_days,
    MAX(days_since_prev) AS max_gap_days,
    ROUND(AVG(days_since_prev), 1) AS avg_gap_days
FROM observation_gaps
GROUP BY series_id
ORDER BY series_id;

-- DATA QUALITY NOTE: T10Y2Y is daily with weekday-only observations, so it is
-- excluded above. GDPC1 is quarterly (~90 day gaps) and also excluded.
-- For monthly series the expected avg gap is ~30-31 days.


-- ---------------------------------------------------------------------------
-- 4. Value distribution summary per series
-- ---------------------------------------------------------------------------
-- Min/max/mean/median gives a quick sense of scale and catches obvious
-- outliers. A negative CPI or unemployment over 50% would warrant
-- investigation.

SELECT
    sm.series_id,
    sm.name,
    sm.units,
    COUNT(o.value)          AS n,
    ROUND(MIN(o.value), 2)  AS min_val,
    ROUND(MAX(o.value), 2)  AS max_val,
    ROUND(AVG(o.value), 2)  AS mean_val
FROM series_metadata sm
JOIN observations o ON sm.series_id = o.series_id
GROUP BY sm.series_id
ORDER BY sm.series_id;


-- ---------------------------------------------------------------------------
-- 5. Year-over-year observation counts (data completeness heatmap)
-- ---------------------------------------------------------------------------
-- Pivoting by year shows whether coverage is consistent or if certain years
-- are sparse. Useful for deciding safe date ranges for analysis.

SELECT
    series_id,
    SUBSTR(date, 1, 4) AS year,
    COUNT(*)            AS obs_count
FROM observations
GROUP BY series_id, SUBSTR(date, 1, 4)
ORDER BY series_id, year;


-- ---------------------------------------------------------------------------
-- 6. Duplicate check (should return zero rows)
-- ---------------------------------------------------------------------------
-- The schema enforces UNIQUE(series_id, date) but belt-and-suspenders
-- verification catches silent constraint failures.

SELECT
    series_id,
    date,
    COUNT(*) AS dupes
FROM observations
GROUP BY series_id, date
HAVING COUNT(*) > 1;


-- ---------------------------------------------------------------------------
-- 7. Recent values per series (latest 3 observations each)
-- ---------------------------------------------------------------------------
-- Quick visual check: do the latest numbers match what FRED shows on its
-- website? Uses ROW_NUMBER window function to pick top-N per group.

WITH ranked AS (
    SELECT
        o.series_id,
        sm.name,
        o.date,
        o.value,
        ROW_NUMBER() OVER (
            PARTITION BY o.series_id
            ORDER BY o.date DESC
        ) AS rn
    FROM observations o
    JOIN series_metadata sm ON o.series_id = sm.series_id
)
SELECT series_id, name, date, value
FROM ranked
WHERE rn <= 3
ORDER BY series_id, date DESC;


-- ---------------------------------------------------------------------------
-- 8. Category-level summary
-- ---------------------------------------------------------------------------
-- How many series and observations do we have per analytical category?
-- Useful for confirming the data supports each section of the dashboard.

SELECT
    sm.category,
    COUNT(DISTINCT sm.series_id) AS series_count,
    COUNT(o.id)                  AS total_obs,
    MIN(o.date)                  AS earliest,
    MAX(o.date)                  AS latest
FROM series_metadata sm
LEFT JOIN observations o ON sm.series_id = o.series_id
GROUP BY sm.category
ORDER BY sm.category;
