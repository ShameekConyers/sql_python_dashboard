-- =============================================================================
-- Analysis Queries: Recession Signals and AI's Labor Market Impact
-- =============================================================================
--
-- Core business question: as AI reshapes white-collar work, how do traditional
-- recession signals interact with diverging employment trends between
-- information-sector workers and skilled trades?
--
-- These queries use value_covid_adjusted (ARIMA counterfactual for the COVID
-- window, raw values everywhere else) for trend analysis. Q5 is the exception:
-- it uses raw values because it explicitly examines the COVID shock.
--
-- Run covid_adjustment.py BEFORE exporting these queries.
-- =============================================================================


-- ---------------------------------------------------------------------------
-- Q1: Yield Curve Inversions as a Recession Leading Indicator
-- ---------------------------------------------------------------------------
-- The 10Y-2Y treasury spread (T10Y2Y) turning negative has preceded every
-- U.S. recession since the 1970s. This query converts daily yield curve
-- data to monthly averages and pairs each month with the unemployment rate,
-- letting us see how inversions precede unemployment spikes.
--
-- Why monthly average instead of daily? UNRATE is monthly, so we need a
-- common grain. Averaging also smooths out single-day noise in the spread.
--
-- Uses: CTE, JOIN, window function (LAG for month-over-month change).

WITH monthly_spread AS (
    SELECT
        SUBSTR(date, 1, 7) AS month,
        ROUND(AVG(value_covid_adjusted), 3) AS avg_spread
    FROM observations
    WHERE series_id = 'T10Y2Y'
    GROUP BY SUBSTR(date, 1, 7)
),
unemployment AS (
    SELECT
        SUBSTR(date, 1, 7) AS month,
        value_covid_adjusted AS unemployment_rate
    FROM observations
    WHERE series_id = 'UNRATE'
)
SELECT
    ms.month,
    ms.avg_spread,
    CASE WHEN ms.avg_spread < 0 THEN 1 ELSE 0 END AS inverted,
    u.unemployment_rate,
    ROUND(
        u.unemployment_rate - LAG(u.unemployment_rate, 12)
            OVER (ORDER BY ms.month),
        2
    ) AS unemployment_yoy_change
FROM monthly_spread ms
LEFT JOIN unemployment u ON ms.month = u.month
ORDER BY ms.month;


-- ---------------------------------------------------------------------------
-- Q2: Information Sector vs Specialty Trades Employment Divergence
-- ---------------------------------------------------------------------------
-- The AI thesis: information sector jobs (tech, media, data) face disruption
-- while specialty trades (plumbing, electrical, HVAC) remain insulated. If
-- true, we should see a growing gap or at least divergent trajectories.
--
-- Raw employment numbers grow partly because the population grows. To compare
-- sectors fairly over a decade, we normalize each by CNP16OV (working-age
-- population) to get "employees per 1000 working-age persons," then index
-- both to 100 at the start date.
--
-- Uses: CTE, multi-table JOIN, FIRST_VALUE window function, per-capita math.

WITH population AS (
    SELECT date, value_covid_adjusted AS pop
    FROM observations
    WHERE series_id = 'CNP16OV'
),
per_capita AS (
    SELECT
        o.date,
        o.series_id,
        -- employment per 1000 working-age persons
        o.value_covid_adjusted / p.pop * 1000 AS per_1k_pop,
        o.value_covid_adjusted AS employment
    FROM observations o
    JOIN population p ON o.date = p.date
    WHERE o.series_id IN ('USINFO', 'CES2023800001')
),
indexed AS (
    SELECT
        date,
        series_id,
        employment,
        per_1k_pop,
        ROUND(
            100.0 * per_1k_pop / FIRST_VALUE(per_1k_pop)
                OVER (PARTITION BY series_id ORDER BY date
                      ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
            2
        ) AS pc_index
    FROM per_capita
)
SELECT
    i.date,
    i.employment      AS info_employment,
    i.pc_index        AS info_pc_index,
    t.employment      AS trades_employment,
    t.pc_index        AS trades_pc_index,
    ROUND(t.pc_index - i.pc_index, 2) AS divergence_gap
FROM indexed i
JOIN indexed t ON i.date = t.date
WHERE i.series_id = 'USINFO'
  AND t.series_id = 'CES2023800001'
ORDER BY i.date;


-- ---------------------------------------------------------------------------
-- Q3: GDP Growth Rate with Recession Context
-- ---------------------------------------------------------------------------
-- Real GDP (GDPC1) is quarterly. This query computes quarter-over-quarter
-- annualized growth rates, which is how economists conventionally report GDP.
-- Two consecutive negative quarters is the informal "recession" definition.
--
-- USREC provides ground-truth NBER recession shading. The two-negative-
-- quarters heuristic serves as an independent cross-check.
--
-- Uses: CTE, LAG window function, LEFT JOIN, mathematical transformation.

WITH gdp_growth AS (
    SELECT
        date,
        value_covid_adjusted AS real_gdp,
        LAG(value_covid_adjusted) OVER (ORDER BY date) AS prev_gdp,
        ROUND(
            (value_covid_adjusted / LAG(value_covid_adjusted) OVER (ORDER BY date) - 1) * 400,
            2
        ) AS annualized_growth_pct
    FROM observations
    WHERE series_id = 'GDPC1'
),
recession_flag AS (
    SELECT
        SUBSTR(date, 1, 7) AS month,
        MAX(CAST(value_covid_adjusted AS INTEGER)) AS in_recession
    FROM observations
    WHERE series_id = 'USREC'
    GROUP BY SUBSTR(date, 1, 7)
)
SELECT
    g.date,
    g.real_gdp,
    g.annualized_growth_pct,
    CASE
        WHEN g.annualized_growth_pct < 0
         AND LAG(g.annualized_growth_pct) OVER (ORDER BY g.date) < 0
        THEN 1
        ELSE 0
    END AS two_neg_quarters,
    COALESCE(r.in_recession, 0) AS nber_recession
FROM gdp_growth g
LEFT JOIN recession_flag r ON SUBSTR(g.date, 1, 7) = r.month
WHERE g.annualized_growth_pct IS NOT NULL
ORDER BY g.date;


-- ---------------------------------------------------------------------------
-- Q4: Rolling 12-Month Per-Capita Employment Growth by Sector
-- ---------------------------------------------------------------------------
-- Instead of raw levels, growth rates reveal momentum. A sector adding 2%
-- per year is healthy; one at -1% is contracting. This query computes a
-- rolling 12-month percent change for each employment series, normalized
-- by working-age population (CNP16OV).
--
-- Why per-capita? Raw YoY growth conflates real sector expansion with
-- population growth (~0.5%/yr). Dividing by CNP16OV isolates the sector-
-- specific trend.
--
-- Why 12-month instead of month-over-month? Monthly data is noisy due to
-- seasonal hiring patterns. The 12-month window cancels out seasonality
-- and shows the underlying trend.
--
-- UNRATE is already a rate, so it is not population-adjusted here.
--
-- Uses: CTE, LAG with offset 12, JOIN, CASE for labeling.

WITH per_capita AS (
    SELECT
        o.series_id,
        o.date,
        CASE
            -- UNRATE is already a rate; pass through as-is
            WHEN o.series_id = 'UNRATE' THEN o.value_covid_adjusted
            -- Employment series: normalize to per 1k working-age population
            ELSE o.value_covid_adjusted / p.value_covid_adjusted * 1000
        END AS adj_value,
        o.value_covid_adjusted AS raw_value
    FROM observations o
    LEFT JOIN observations p
        ON o.date = p.date AND p.series_id = 'CNP16OV'
    WHERE o.series_id IN ('UNRATE', 'USINFO', 'CES2023800001')
),
lagged AS (
    SELECT
        series_id,
        date,
        raw_value,
        adj_value,
        LAG(adj_value, 12) OVER (PARTITION BY series_id ORDER BY date) AS adj_12m_ago
    FROM per_capita
)
SELECT
    series_id,
    CASE series_id
        WHEN 'UNRATE'         THEN 'Unemployment Rate'
        WHEN 'USINFO'         THEN 'Information Sector'
        WHEN 'CES2023800001'  THEN 'Specialty Trades'
    END AS series_name,
    date,
    raw_value,
    ROUND(adj_value, 4) AS adj_value,
    ROUND(adj_12m_ago, 4) AS adj_12m_ago,
    ROUND(
        (adj_value - adj_12m_ago) / adj_12m_ago * 100,
        2
    ) AS yoy_pct_change_pc
FROM lagged
WHERE adj_12m_ago IS NOT NULL
ORDER BY series_id, date;


-- ---------------------------------------------------------------------------
-- Q5: COVID Recession Recovery — Information vs Trades Timeline
-- ---------------------------------------------------------------------------
-- The COVID shock (2020-03 to 2020-04) hit both sectors hard, but recovery
-- speed and completeness diverged. This query computes each sector's
-- employment as a percent of its pre-COVID peak (Feb 2020), showing how
-- long each took to recover.
--
-- NOTE: This query intentionally uses raw `value` (not covid-adjusted) since
-- it explicitly examines the real COVID impact and recovery trajectory.
--
-- Uses: subquery for baseline, CASE for pivot.

WITH pre_covid_peak AS (
    SELECT
        series_id,
        value AS peak_value
    FROM observations
    WHERE series_id IN ('USINFO', 'CES2023800001')
      AND date = '2020-02-01'
),
recovery AS (
    SELECT
        o.series_id,
        o.date,
        o.value,
        p.peak_value,
        ROUND(o.value / p.peak_value * 100, 2) AS pct_of_peak
    FROM observations o
    JOIN pre_covid_peak p ON o.series_id = p.series_id
    WHERE o.date >= '2020-01-01'
      AND o.series_id IN ('USINFO', 'CES2023800001')
)
SELECT
    date,
    MAX(CASE WHEN series_id = 'USINFO'         THEN value END)       AS info_employment,
    MAX(CASE WHEN series_id = 'USINFO'         THEN pct_of_peak END) AS info_pct_of_peak,
    MAX(CASE WHEN series_id = 'CES2023800001'  THEN value END)       AS trades_employment,
    MAX(CASE WHEN series_id = 'CES2023800001'  THEN pct_of_peak END) AS trades_pct_of_peak
FROM recovery
GROUP BY date
ORDER BY date;


-- ---------------------------------------------------------------------------
-- Q6: U6 vs U3 Unemployment Gap (Hidden Labor Market Slack)
-- ---------------------------------------------------------------------------
-- U3 (UNRATE) counts people actively looking for work. U6 adds discouraged
-- workers and part-timers who want full-time hours. A widening U6-U3 gap
-- means more hidden slack in the labor market, exactly what you would expect
-- if AI is pushing workers into underemployment rather than outright layoffs.
--
-- Uses: JOIN, window function (LAG for YoY comparison), arithmetic.

SELECT
    u3.date,
    u3.value_covid_adjusted  AS u3_rate,
    u6.value_covid_adjusted  AS u6_rate,
    ROUND(u6.value_covid_adjusted - u3.value_covid_adjusted, 2) AS u6_u3_gap,
    ROUND(
        (u3.value_covid_adjusted - LAG(u3.value_covid_adjusted, 12)
            OVER (ORDER BY u3.date)),
        2
    ) AS u3_yoy_change,
    ROUND(
        (u6.value_covid_adjusted - LAG(u6.value_covid_adjusted, 12)
            OVER (ORDER BY u3.date)),
        2
    ) AS u6_yoy_change
FROM observations u3
JOIN observations u6 ON u3.date = u6.date AND u6.series_id = 'U6RATE'
WHERE u3.series_id = 'UNRATE'
ORDER BY u3.date;


-- ---------------------------------------------------------------------------
-- Q7: Electric Power Output vs Information Sector Employment
-- ---------------------------------------------------------------------------
-- AI's paradox: data centers are driving electricity demand up while
-- displacing the workers who build the software running on them. This query
-- indexes both series to 100 at the start date and tracks the divergence.
--
-- Uses: CTE, FIRST_VALUE window function, JOIN.

WITH power AS (
    SELECT
        date,
        value_covid_adjusted AS power_index_raw,
        ROUND(
            100.0 * value_covid_adjusted / FIRST_VALUE(value_covid_adjusted)
                OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
            2
        ) AS power_index
    FROM observations
    WHERE series_id = 'IPG2211S'
),
info AS (
    SELECT
        date,
        value_covid_adjusted AS info_employment,
        ROUND(
            100.0 * value_covid_adjusted / FIRST_VALUE(value_covid_adjusted)
                OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING),
            2
        ) AS info_index
    FROM observations
    WHERE series_id = 'USINFO'
)
SELECT
    p.date,
    p.power_index_raw,
    p.power_index,
    i.info_employment,
    i.info_index,
    ROUND(p.power_index - i.info_index, 2) AS power_vs_info_gap
FROM power p
JOIN info i ON p.date = i.date
ORDER BY p.date;


-- ---------------------------------------------------------------------------
-- Q8: CPI Inflation — Month-over-Month and Year-over-Year
-- ---------------------------------------------------------------------------
-- Exploratory view of inflation trends. CPI is reported as an index level,
-- not a rate, so we compute both MoM and YoY percent changes. This gives
-- context for the recession and employment queries above: is the Fed
-- fighting inflation while the labor market cools?
--
-- Uses: LAG with offsets 1 and 12, arithmetic.

SELECT
    date,
    value_covid_adjusted AS cpi_level,
    ROUND(
        (value_covid_adjusted / LAG(value_covid_adjusted, 1)
            OVER (ORDER BY date) - 1) * 100,
        3
    ) AS mom_pct_change,
    ROUND(
        (value_covid_adjusted / LAG(value_covid_adjusted, 12)
            OVER (ORDER BY date) - 1) * 100,
        2
    ) AS yoy_pct_change
FROM observations
WHERE series_id = 'CPIAUCSL'
ORDER BY date;
