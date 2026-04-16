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
python src/db_setup.py --full
python src/covid_adjustment.py --full
python src/export_csv.py --full
python src/ai_insights.py --db full
python src/verify_insights.py --db full
python src/recession_model.py --db full
streamlit run dashboard/app.py
```

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
