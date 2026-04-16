# Contributing

## Setup

1. Clone the repo
2. Create a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
3. Install dev dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```
   Dev install pulls in `torch` (~700 MB on macOS arm64) via
   `sentence-transformers`, plus `chromadb`. These are generation-time-only
   and never deployed — the dashboard reads pre-computed citations from
   `data/seed.db` and the vector store is only rebuilt by
   `src/embed_references.py`. First run of `embed_references.py` also
   downloads the ~80 MB `all-MiniLM-L6-v2` model into
   `~/.cache/huggingface/`.
4. Copy `.env.example` to `.env` and add your FRED API key (only needed for
   `--full` mode)

## Running the Dashboard

The project ships with `data/seed.db`, so no API key is needed for the default
experience:

```bash
streamlit run dashboard/app.py
```

## Full Pipeline

To pull fresh data from the FRED API and rebuild everything:

```bash
python src/data_pull.py
python src/hackernews_pull.py
python src/sentiment_score.py
python src/db_setup.py --full
python src/covid_adjustment.py --full
python src/export_csv.py --full
python src/embed_references.py --db full --rebuild
python src/recession_model.py --db full
python src/ai_insights.py --db full
python src/verify_insights.py --db full
streamlit run dashboard/app.py
```

`embed_references.py` reads `reference_docs`, chunks each row, embeds it
with `sentence-transformers/all-MiniLM-L6-v2`, and persists the collection
to `data/.chroma/`. `ai_insights.py` then retrieves from that store and
writes `[ref:N]` citations into `ai_insights.citations_json`.

## Hacker News Data

Phase 13 adds a thinned Hacker News corpus as a tech-labor-sentiment
signal. `src/hackernews_pull.py` uses the public Algolia HN search API
(no auth, no API key) and caches up to the top 30 stories per month to
`data/raw/hn_stories.json`. The window starts at 2022-01-01 to match
the default x-axis start of the dashboard's time-series charts. Story
text is truncated to 500 characters at ingestion.
`src/sentiment_score.py` scores each story with
`cardiffnlp/twitter-roberta-base-sentiment-latest` and writes a
compound score back to the same JSON. `db_setup.py` loads the scored
JSON into `hn_stories` and computes `hn_sentiment_monthly` via SQL
GROUP BY.

Phase 14 wires HN data into three surfaces: the recession classifier
gains three HN-derived features (`hn_sentiment_3m_avg`,
`hn_story_volume_yoy`, `layoff_story_freq`) imputed with
training-period medians for pre-2022 months; the top 5 stories per
month are loaded into `reference_docs` under
`doc_type='social:hn:<story_id>'` so the RAG pipeline embeds and
retrieves them alongside FRED and scholarly refs; and the Deep Dive tab
shows a new dual-axis chart overlaying 3-month rolling HN sentiment on
per-capita info-sector employment with an `hn_labor_sentiment` AI
insight slice below it. After any HN data refresh,
`embed_references.py --rebuild` is mandatory to re-index the social
refs into ChromaDB.

The raw JSON is gitignored under the existing `data/raw/` rule.
`hn_stories` ships inside `seed.db` (~1,500 rows, thinned excerpts
only). No API credentials or `.env` entries are required.

## Reference Content

`reference_docs` is populated from two sources. The first is per-series
FRED metadata (series notes, release info, category path) pulled by
`data_pull.py` and upserted by `db_setup._load_reference_docs`. The
second is a set of curated JSON fixtures under
`data/reference_sources/scholarly/`, loaded by
`db_setup._load_scholarly_docs`. Each scholarly fixture has a stable
`id`, a FRED `series_id` that must exist in `series_metadata`, a short
attribution-ready excerpt in `content`, and a `source_url` pointing at
a US federal-government publication (BEA, BLS, EIA, CRS, CBO, CEA,
Treasury, GAO). The corpus is intentionally limited to public-domain
federal-government content so the repo can quote sources verbatim
without fair-use gymnastics. To add a new source, drop in a new JSON
file, then rebuild the DB (`python src/db_setup.py`) and the vector
index (`python src/embed_references.py --db seed --rebuild`). Scholarly
rows coexist with the FRED triple because their `doc_type` is
`scholarly:<slug>` rather than one of the three FRED types, which
sidesteps the UNIQUE `(series_id, doc_type)` constraint.

Phase 14 adds a third source: Hacker News social references under
`doc_type='social:hn:<story_id>'` with `series_id='USINFO'`. The top 5
stories per month by score are selected by `db_setup._load_hn_reference_docs`
and fed into the same embed/retrieve/verify pipeline as FRED and scholarly
refs. The selection is capped at ~260 rows to keep ChromaDB growth and
retrieval signal-to-noise within budget. HN sentiment is a self-selected
tech-practitioner signal, not a representative labor-market measure.

## Running Tests

```bash
pytest tests/ -v
```

Tests use in-memory SQLite databases and never modify `seed.db`.

## Code Style

- Python: type annotations on all functions and classes. Google-style docstrings.
- SQL: uppercase keywords, lowercase identifiers, CTEs over subqueries.
- Naming: `snake_case` for files, functions, variables. `UPPER_SNAKE_CASE` for
  constants. `PascalCase` for classes.

## Commit Messages

Use semantic format: `<type>: <subject>` in present tense.

| Type | Purpose |
|------|---------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `style:` | Formatting, no logic change |
| `refactor:` | Code restructuring |
| `test:` | Test additions or changes |
| `chore:` | Build, config, tooling |

## Dependencies

- `requirements.txt`: minimal deps for Streamlit Cloud deployment (no sklearn)
- `requirements-dev.txt`: full pinned deps for local development

When adding a dependency, put it in `requirements-dev.txt` with a pinned
version. Only add it to `requirements.txt` if the dashboard needs it at
runtime.

## Database

- `data/seed.db`: committed to git, contains 10 FRED series with ~10 years of
  history and pre-computed AI insights. Keep under 25 MB.
- `data/full.db`: gitignored, built from live API data via `--full` flag.
- Schema defined in `sql/01_schema.sql`. Prediction schema in
  `sql/04_prediction_schema.sql`.
