# Macro Economic Dashboard: Leading Indicators and Recession Risk

An end-to-end analytics pipeline that pulls macroeconomic data from the FRED API, models it in a SQLite database, analyzes it in Python, and surfaces AI-generated narrative insights — each one programmatically verified against the source data — through an interactive Streamlit dashboard.

---

## Business Question

Which combinations of leading economic indicators have historically preceded U.S. recessions, and what do current readings suggest about near-term recession risk? This project translates raw Federal Reserve data into plain-English, evidence-backed narratives that a business stakeholder can act on — without needing to read a chart.

---

## Pipeline

```
FRED API → data_pull.py → SQLite (star schema) → SQL analysis → Python EDA → AI insight generation → programmatic verification → Streamlit dashboard
```

---

## Dataset

**Source:** [FRED API](https://fred.stlouisfed.org/docs/api/fred/) — Federal Reserve Bank of St. Louis
**Series pulled:** GDP, unemployment rate (UNRATE), CPI, federal funds rate, yield curve spread (T10Y2Y), initial jobless claims (ICSA), housing starts, consumer sentiment
**Volume:** ~30–50 time series, decades of monthly/quarterly observations
**Update frequency:** Daily to monthly depending on series
**Auth:** Free API key at [fredaccount.stlouisfed.org](https://fredaccount.stlouisfed.org)

---

## Key Findings

> *To be filled in after data pull and analysis.*

---

## AI-Verified Narrative Insights

The dashboard surfaces LLM-generated narratives alongside each chart. The approach:

1. **Batch generation** — After the SQL analysis runs, key metric aggregations (trends, correlations, anomalies) are packaged into structured prompts and sent to Claude Haiku in a single batch
2. **Structured output** — The LLM returns JSON with a `narrative` field and a `claims` array, where each claim specifies the exact metric, value, and time period it references
3. **Programmatic verification** — Every claim is checked against the database. A claim asserting "unemployment at 4.2%" is verified against the actual stored value. Failures are flagged
4. **Cached in database** — Insights are stored in an `ai_insights` table and served statically. Zero live LLM calls at dashboard load time

Each insight in the dashboard shows a verification badge: ✓ all claims verified, or ⚠ one or more claims flagged. A "Show sources" toggle expands to display the underlying data points from the database.

This is the capability that Tableau and Power BI are racing to ship. The difference here: every factual claim is machine-checked before it ever reaches the UI.

---

## Architecture

```
project_a_sql_python_dashboard/
├── src/
│   ├── data_pull.py            # FRED API ingestion — idempotent, caches raw JSON locally
│   ├── db_setup.py             # Creates star schema, loads DB (--seed or --full)
│   ├── ai_insights.py          # Batch generates AI narratives, stores in ai_insights table
│   └── verify_insights.py      # Fact-checks every AI claim against DB values
├── sql/
│   ├── 01_schema.sql           # Table definitions
│   ├── 02_exploration_queries.sql
│   └── 03_analysis_queries.sql
├── notebooks/
│   └── analysis.ipynb          # EDA, statistical analysis, findings
├── dashboard/
│   └── app.py                  # Streamlit dashboard
├── data/
│   ├── seed.db                 # Pre-built demo DB with cached AI insights (<25MB)
│   └── raw/                    # Cached API responses (gitignored)
├── .env.example
├── requirements.txt
└── README.md
```

**Database schema (star schema):**
- `series_metadata` — series ID, name, frequency, units, source (dimension)
- `categories` — FRED category hierarchy (dimension)
- `observations` — date, series ID, value (fact)
- `ai_insights` — cached narratives, structured claims, verification results

---

## How to Run

### Quick start — uses seed data, no API key needed

```bash
git clone <repo>
cd project_a_sql_python_dashboard
pip install -r requirements.txt
python src/db_setup.py          # loads seed.db
streamlit run dashboard/app.py
```

### Full pipeline — pulls live data from FRED

```bash
cp .env.example .env
# Add your FRED API key to .env (free at fredaccount.stlouisfed.org)

python src/data_pull.py         # pulls series, caches raw JSON
python src/db_setup.py --full   # builds database from API data
python src/ai_insights.py       # batch generates narrative insights
python src/verify_insights.py   # fact-checks all claims, flags failures
streamlit run dashboard/app.py
```

---

## Dashboard

> *Screenshots to be added.*

**Views:**
- **Overview** — KPI cards (current unemployment, CPI, GDP growth, yield curve spread) + trend summary
- **Indicator Explorer** — filterable time series charts with date range selector and AI insight blocks
- **Recession Risk** — cross-indicator analysis, historical comparison, AI-verified narrative summary

---

## Tools

| Category | Tools |
|----------|-------|
| Data ingestion | Python, `fredapi`, `requests` |
| Database | SQLite, SQL (CTEs, window functions, star schema) |
| Analysis | pandas, numpy, scipy, seaborn, matplotlib |
| AI insights | Claude Haiku (`claude-haiku-4-5-20251001`) |
| Dashboard | Streamlit |
| Environment | `python-dotenv` |

---

## What I'd Do Next

- Add a predictive model for recession probability (logistic regression or gradient boosting on lagged indicator values)
- Schedule automated data refresh via GitHub Actions
- Deploy to Streamlit Community Cloud for a live public URL
- Expand to regional economic data for state-level breakdowns
