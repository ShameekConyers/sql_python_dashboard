# Macro Economic Dashboard

An analytics pipeline that pulls U.S. macroeconomic data from the FRED API, fits a recession probability model, runs NLP on Hacker News tech-labor discussions, and generates narrative insights that are each fact-checked against the source database. The output is a Streamlit dashboard with four sections and 21 verified insight blocks.

## The question

The information sector has been shedding jobs per capita since 2022. Specialty trades have been growing. Power output is climbing. The yield curve was inverted for over two years. What does this actually mean?

This project pulls 10 FRED series and 1,500+ Hacker News stories to look at that question from two angles: the numbers (recession indicators, employment divergence, inflation) and the text (what are tech workers actually talking about, and does their sentiment track the employment data).

## What's in the data

All economic data comes from the [FRED API](https://fred.stlouisfed.org/). The Hacker News corpus comes from the [Algolia HN Search API](https://hn.algolia.com/api/v1) (public, no auth).

**FRED series (10):**

| Series | What it is | Frequency |
|--------|-----------|-----------|
| UNRATE | Unemployment rate (U-3) | Monthly |
| U6RATE | Underemployment rate (U-6) | Monthly |
| GDPC1 | Real GDP | Quarterly |
| T10Y2Y | 10Y minus 2Y Treasury spread | Daily |
| CPIAUCSL | CPI, all urban consumers | Monthly |
| CNP16OV | Working-age population (16+) | Monthly |
| USINFO | Information sector employment | Monthly |
| CES2023800001 | Specialty trade contractors | Monthly |
| IPG2211S | Electric power output | Monthly |
| USREC | NBER recession indicator | Monthly |

**Hacker News corpus:** 1,547 stories (2022 onward) across layoff, AI jobs, and career themes. Scored with a transformer sentiment model. Grouped into 8 topics via NMF.

**Coverage:** ~10 years of FRED data (March 2016 to March 2026). The seed database ships committed to the repo at 3.1 MB. Clone and run, no API key needed.

## Findings

The per-capita normalization matters. Raw info-sector employment looks flat. After dividing by working-age population, it fell 7.2%. Specialty trades grew 13.5% on the same basis. That's a 20+ point gap.

The yield curve was inverted in 26 of the months tracked. Every U.S. recession since the 1970s was preceded by an inversion.

Inflation hit 37% cumulative over the dataset window. Unemployment sits near 4.4%, but U6 (which counts discouraged and part-time workers) runs 3.3 pp higher, a gap that has stayed wide since 2020.

Electric power output is up 8.5% since ChatGPT launched in November 2022.

On the NLP side, the dominant Hacker News topic is "Software Engineering Careers" (585 stories). The most negative topic by sentiment is "Executive Firings & Restructuring." Layoff story volume and the U6-U3 unemployment gap move in the same direction across the 2022-2026 window.

## How it works

```
FRED API + HN API
       |
  data_pull.py + hackernews_pull.py
       |
  sentiment_score.py
       |
  db_setup.py  ->  covid_adjustment.py  ->  topic_model.py
       |
  export_csv.py  ->  embed_references.py  ->  recession_model.py
       |
  ai_insights.py  ->  verify_insights.py
       |
  dashboard/app.py
```

The database is SQLite with a star schema: `series_metadata` (dimension), `observations` (fact), `ai_insights` (narratives + verification results), `recession_predictions` (model output), `hn_stories`, `hn_topics`, and a few more.

Two modes: **seed** (default, everything pre-computed, no API calls) and **full** (live pull from FRED and HN, requires a FRED API key).

### COVID adjustment

COVID broke every rolling-window calculation in the dataset. Unemployment went from 3.5% to 14.8% in a month. An ARIMA model fitted on pre-COVID data fills in a counterfactual for the March 2020 to January 2022 window, with a 3-month taper back to actual values. The raw data stays in the `value` column. The adjusted version goes in `value_covid_adjusted`. All queries use the adjusted column except the COVID recovery chart, which shows the real shock on purpose.

### Per-capita normalization

Employment numbers grow partly because the population grows. USINFO and CES2023800001 are divided by CNP16OV (working-age population) to get employees per 1,000 people before any indexing or growth rate calculation. UNRATE and U6RATE are already rates, so they pass through unchanged.

### Recession model

A logistic regression and random forest trained on 11 FRED-derived features plus 3 Hacker News features (sentiment, story volume, layoff frequency). Outputs monthly recession probability scores stored in `recession_predictions`. The dashboard has a Recession Risk tab with a probability timeline, feature snapshot, and a What If scenario explorer with sliders.

### NLP section

NMF topic model (sklearn, not BERTopic) over 1,547 HN story titles and excerpts, producing 8 topics. The dashboard's NLP Analysis section has four charts: topic distribution over time, sentiment by topic, layoff story volume vs the U6-U3 gap, and topic sentiment vs USINFO per-capita employment. Monthly bigram frequencies are pre-computed and shown as a heatmap.

### Insight verification

The LLM (llama3.1:8b via Ollama, running locally) never runs live. During the build step, Python pre-computes verifiable claims from database queries, then the LLM writes prose from that context. A separate script re-queries the database for every claim and checks values within tolerance (5% relative or 0.5 absolute). The dashboard shows a verification badge on each insight block and a "Show sources" panel with per-claim results. RAG retrieval pulls relevant chunks from FRED metadata and scholarly references (federal public-domain publications) into the prompt. 21 insights ship in the seed database, all verified.

## Tools

| Tool | What it does here |
|------|------------------|
| Python 3.12+ | Pipeline scripts, model training, insight generation |
| SQL / SQLite | Star schema, 8 analysis queries with CTEs and window functions |
| pandas | Data manipulation, per-capita normalization |
| scikit-learn | Recession classifier (LR + RF), NMF topic model, TF-IDF |
| pmdarima | Automatic ARIMA order selection for COVID adjustment |
| Plotly | Interactive charts, dual axes, recession shading |
| Streamlit | Dashboard with caching and sidebar filters |
| sentence-transformers | Embedding FRED metadata and scholarly docs for RAG |
| ChromaDB | Vector store for citation retrieval |
| Ollama (llama3.1:8b) | Local LLM for narrative generation |
| FRED API | 10 macroeconomic time series |
| Algolia HN API | Hacker News story search (public, no auth) |

## Running it

### Quick start (no API key needed)

```bash
git clone <repo-url> && cd sql_python_dashboard
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
.venv/bin/streamlit run dashboard/app.py
```

The seed database has everything pre-computed: 10 FRED series, COVID adjustments, recession predictions, topic assignments, and 21 verified insights.

`requirements.txt` is for Streamlit Cloud deployment (dashboard deps only). `requirements-dev.txt` is the full set for local development, including sklearn, pmdarima, and the embedding/sentiment models.

### Full pipeline (live data)

```bash
# Get a free API key at https://fredaccount.stlouisfed.org
cp .env.example .env   # paste your FRED_API_KEY

.venv/bin/python src/data_pull.py
.venv/bin/python src/hackernews_pull.py
.venv/bin/python src/sentiment_score.py
.venv/bin/python src/db_setup.py --full
.venv/bin/python src/covid_adjustment.py --db full
.venv/bin/python src/topic_model.py --db full
.venv/bin/python src/export_csv.py --db full
.venv/bin/python src/embed_references.py --db full --rebuild
.venv/bin/python src/recession_model.py --db full
.venv/bin/python src/ai_insights.py --db full   # requires Ollama running
.venv/bin/python src/verify_insights.py --db full
.venv/bin/streamlit run dashboard/app.py
```

### Tests

```bash
.venv/bin/pytest tests/ -v   # 430 tests, ~80 seconds
```

