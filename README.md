# Macro Economic Dashboard: Recession Risk and AI's Labor Market Impact

An end-to-end analytics pipeline that pulls macroeconomic data from the FRED API, models it in a SQLite database, analyzes it in Python, and surfaces AI-generated narrative insights — each one programmatically verified against the source data — through an interactive Streamlit dashboard.

---

## Live Dashboard

<!-- Replace YOUR_APP_URL with the deployed Streamlit Community Cloud URL -->
<iframe src="YOUR_APP_URL/?embed=true" width="100%" height="800" frameborder="0" allowfullscreen></iframe>

---

## Business Question

How do traditional recession indicators (yield curve, GDP, inflation) interact with the diverging employment trends between AI-disrupted white-collar sectors and AI-resistant skilled trades? As data center energy demand surges and information-sector employment shifts, what does the macro picture look like for urban knowledge workers versus tradespeople?

This project tracks 10 FRED series across recession risk, sector employment, and energy production to translate raw Federal Reserve data into plain-English, evidence-backed narratives a business stakeholder can act on.

---

## Dataset

All data comes from the [FRED API](https://fred.stlouisfed.org/) (Federal Reserve Economic Data), the gold-standard public source for U.S. macroeconomic time series.

| Series ID | Name | Category | Frequency |
|-----------|------|----------|-----------|
| UNRATE | Unemployment Rate | Labor Market | Monthly |
| U6RATE | Total Underemployment Rate (U-6) | Labor Market | Monthly |
| GDPC1 | Real Gross Domestic Product | Output/Growth | Quarterly |
| T10Y2Y | 10Y Minus 2Y Treasury Spread | Yield Curve | Daily |
| CPIAUCSL | Consumer Price Index (All Urban) | Prices | Monthly |
| CNP16OV | Civilian Noninstitutional Pop. (16+) | Population | Monthly |
| USINFO | Information Sector Employment | AI Labor Impact | Monthly |
| CES2023800001 | Specialty Trade Contractors | AI Labor Impact | Monthly |
| IPG2211S | Electric Power Gen. & Distribution | AI Energy | Monthly |
| USREC | NBER Recession Indicator | Recession | Monthly |

- **Coverage:** March 2016 through March 2026 (~10 years)
- **Volume:** 3,488 observations across 10 series, plus 9 AI insight records
- **Seed database:** 0.4 MB, committed to Git, works out of the box with zero API calls

---

## Key Findings

1. **Information sector employment declined 7.2% per capita** over the dataset period while specialty trades grew 13.5% — a 20+ point divergence that supports the AI disruption thesis.
2. **The yield curve inverted in 26 of the months tracked**, historically a signal that has preceded every U.S. recession since the 1970s.
3. **Cumulative inflation reached 37.0%** across the dataset window, compressing real wages even as headline unemployment stays near 4.4%.
4. **Electric power output rose 8.5% since ChatGPT's launch** (November 2022), consistent with surging data center demand.
5. **The U6-U3 unemployment gap averaged 3.3 percentage points since 2020**, suggesting persistent hidden labor slack from discouraged workers and involuntary part-timers.

---

## Architecture

```
FRED API → data_pull.py → db_setup.py → covid_adjustment.py → export_csv.py
                                              ↓
                                     ai_insights.py → verify_insights.py
                                              ↓
                                     Streamlit dashboard (app.py)
```

**Database:** SQLite star schema with three tables:
- `series_metadata` (dimension) — display names, categories, units for each FRED series
- `observations` (fact) — raw values plus ARIMA COVID-adjusted values side by side
- `ai_insights` (cached narratives) — batch-generated prose with pre-computed claims and verification results

**Two operating modes:**
- **Seed** (default) — ships with `data/seed.db` committed to Git. Clone and run, no API key needed.
- **Full** — pulls live data from FRED, builds `data/full.db`. Requires a free API key.

**COVID handling:** The COVID window (March 2020 through June 2021) is treated as a statistical outlier that breaks rolling-window calculations. For each series, an ARIMA model fitted on pre-COVID data forecasts a counterfactual through the disruption, with a 3-month linear taper blending back to actual values. Raw data stays in the `value` column; the adjusted series goes in `value_covid_adjusted`. All trend queries use the adjusted column except the COVID recovery chart (Q5), which intentionally shows the real shock.

**Per-capita normalization:** Employment series are divided by CNP16OV (working-age population) before indexing or computing growth rates, isolating real sector expansion from population growth (~0.5%/yr).

---

## AI Insights

Every AI-generated narrative in the dashboard is **batch-generated during data processing, not called live.** The dashboard reads from the database only.

**Hybrid architecture:**
- Verifiable claims are pre-computed from DB queries in Python, guaranteeing reproducibility
- An LLM (Ollama llama3.1:8b, running locally) writes narrative prose from the data context
- `verify_insights.py` independently re-queries the DB for each claim and checks it within tolerance (5% relative or 0.5 absolute for values, +/- 2 for counts, sign-match for trends)

**Three-tier verification badges on the dashboard:**
- **Verified** (green) — all claims confirmed against the database
- **Partially Verified** (orange) — some claims confirmed (shows "X of Y")
- **Unverified** (red) — no claims pass verification

Each insight block has a "Show sources" expander with a per-claim table showing the expected value, actual value, and color-coded status. The seed database ships with 9 pre-computed, fully verified insights so the demo works immediately.

---

## Tools Used

| Tool | Purpose |
|------|---------|
| Python | Pipeline orchestration, data processing, ARIMA modeling |
| SQL / SQLite | Star schema, 8 analysis queries (CTEs, window functions, joins) |
| pandas | Data manipulation, per-capita normalization, CSV export |
| Plotly | Interactive charts with dual axes, recession shading, annotations |
| pmdarima / statsmodels | Auto ARIMA for COVID counterfactual modeling |
| Streamlit | Dashboard framework with caching and interactive filters |
| FRED API | Data source for all 10 macroeconomic time series |
| Ollama (llama3.1:8b) | Local LLM for batch narrative generation |

---

## How to Run

### Quick start (seed mode — no API key needed)

```bash
git clone <repo-url> && cd sql_python_dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
.venv/bin/streamlit run dashboard/app.py
```

The seed database ships with all 10 series, COVID-adjusted values, and 9 verified AI insights. Everything works out of the box.

### Full pipeline (live FRED data)

```bash
# 1. Get a free API key at https://fredaccount.stlouisfed.org
cp .env.example .env   # paste your FRED_API_KEY

# 2. Pull data and build the database
.venv/bin/python src/data_pull.py
.venv/bin/python src/db_setup.py --full
.venv/bin/python src/covid_adjustment.py --db full

# 3. Export analysis results to CSV
.venv/bin/python src/export_csv.py --db full

# 4. Launch the dashboard
.venv/bin/streamlit run dashboard/app.py
```

### Re-generating AI insights (requires Ollama)

```bash
brew install ollama
ollama pull llama3.1:8b
ollama serve  # keep running in a separate terminal

.venv/bin/python src/ai_insights.py --db seed
.venv/bin/python src/verify_insights.py --db seed
```

---

## What I'd Do Next

- **Postgres migration** — move from SQLite to PostgreSQL for concurrent access and production-grade querying
- **Scheduled ingestion** — automate monthly FRED pulls with a cron job or Airflow DAG so the dashboard stays current
- **More series** — add housing starts (HOUST), initial jobless claims (ICSA), and ISM manufacturing index to widen the recession signal net
- **Forecasting models** — extend the ARIMA work to produce forward-looking recession probability estimates
- **Sector granularity** — break the information sector down by subsector (software, telecom, media) to isolate where AI displacement concentrates
