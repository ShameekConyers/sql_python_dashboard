"""End-to-end integration tests for the LangGraph ReAct agent.

Tests the full pipeline: question -> agent -> tool calls -> response ->
verification -> UI-ready output. Uses mocked LLM but real SQLite (in-memory
with project schema) and mocked RAG retrieval.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, ToolMessage

from src.agent.config import AgentConfig
from src.agent.graph import AgentResponse, run_agent


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SQL_DIR: Path = Path(__file__).resolve().parent.parent / "sql"


def _run_schema_file(conn: sqlite3.Connection, filename: str) -> None:
    """Execute a SQL schema file against *conn*.

    Args:
        conn: Open SQLite connection.
        filename: Schema filename inside the ``sql/`` directory.
    """
    path = _SQL_DIR / filename
    if path.exists():
        conn.executescript(path.read_text())


@pytest.fixture()
def integration_db(tmp_path: Path) -> str:
    """Create a SQLite DB with project schema and minimal representative data.

    Returns:
        Filesystem path to the test database.
    """
    db_path = str(tmp_path / "integration_test.db")
    conn = sqlite3.connect(db_path)

    # Apply all schema files in order.
    for schema_file in [
        "01_schema.sql",
        "04_prediction_schema.sql",
        "05_scenario_schema.sql",
        "06_reference_schema.sql",
        "07_hackernews_schema.sql",
        "08_topic_schema.sql",
    ]:
        _run_schema_file(conn, schema_file)

    # -- series_metadata --
    conn.executemany(
        "INSERT INTO series_metadata (series_id, name, category, units, frequency) "
        "VALUES (?, ?, ?, ?, ?)",
        [
            ("UNRATE", "Unemployment Rate", "labor_market", "Percent", "Monthly"),
            ("T10Y2Y", "10-Year Treasury Minus 2-Year", "yield_curve", "Percent", "Daily"),
            ("USINFO", "Information Sector Employment", "employment", "Thousands", "Monthly"),
            ("CNP16OV", "Working-Age Population", "population", "Thousands", "Monthly"),
        ],
    )

    # -- observations (UNRATE) --
    conn.executemany(
        "INSERT INTO observations (series_id, date, value, value_covid_adjusted) "
        "VALUES (?, ?, ?, ?)",
        [
            ("UNRATE", "2025-12-01", 4.0, 4.0),
            ("UNRATE", "2026-01-01", 4.1, 4.1),
            ("UNRATE", "2026-02-01", 4.2, 4.2),
        ],
    )

    # -- observations (T10Y2Y) --
    conn.executemany(
        "INSERT INTO observations (series_id, date, value, value_covid_adjusted) "
        "VALUES (?, ?, ?, ?)",
        [
            ("T10Y2Y", "2025-12-01", -0.10, -0.10),
            ("T10Y2Y", "2026-01-01", 0.05, 0.05),
            ("T10Y2Y", "2026-02-01", 0.15, 0.15),
        ],
    )

    # -- observations (USINFO + CNP16OV for per-capita) --
    conn.executemany(
        "INSERT INTO observations (series_id, date, value, value_covid_adjusted) "
        "VALUES (?, ?, ?, ?)",
        [
            ("USINFO", "2026-01-01", 3000.0, 3000.0),
            ("USINFO", "2026-02-01", 2980.0, 2980.0),
            ("CNP16OV", "2026-01-01", 265000.0, 265000.0),
            ("CNP16OV", "2026-02-01", 265500.0, 265500.0),
        ],
    )

    # -- recession_predictions --
    conn.execute(
        "INSERT INTO recession_predictions "
        "(date, probability, prediction, actual, model_name, features_json, generated_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        ("2026-03-01", 0.35, 0, None, "random_forest", "{}", "2026-03-15T00:00:00"),
    )

    # -- hn_stories (1 row for combined tests) --
    conn.execute(
        "INSERT INTO hn_stories "
        "(story_id, created_utc, month, title, text_excerpt, score, num_comments, "
        "url, hn_permalink, sentiment_score, sentiment_label) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            99999, "2026-02-01T12:00:00", "2026-02-01",
            "Tech Layoffs Continue", "More layoffs announced...",
            150, 200, "https://example.com", "https://news.ycombinator.com/item?id=99999",
            -0.3, "negative",
        ),
    )

    # -- reference_docs (for conceptual questions) --
    conn.executemany(
        "INSERT INTO reference_docs "
        "(series_id, doc_type, title, content, source_url, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        [
            (
                "_CROSS_SERIES", "concept:yield_curve",
                "The Yield Curve Explained",
                "The yield curve plots U.S. Treasury bond yields across maturities. "
                "An inverted yield curve has preceded every U.S. recession since 1970.",
                None, "2026-03-01T00:00:00",
            ),
            (
                "UNRATE", "series_notes",
                "Unemployment Rate (UNRATE)",
                "The unemployment rate represents the number of unemployed as a "
                "percentage of the labor force.",
                "https://fred.stlouisfed.org/series/UNRATE",
                "2026-03-01T00:00:00",
            ),
            (
                "_CROSS_SERIES", "concept:recession_indicators",
                "Recession Indicators",
                "Key indicators include the yield curve spread, unemployment trends, "
                "and GDP growth. The Sahm Rule triggers when 3-month average "
                "unemployment rises 0.5pp above its 12-month low.",
                None, "2026-03-01T00:00:00",
            ),
        ],
    )

    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def integration_config(integration_db: str) -> AgentConfig:
    """Agent config pointing at the integration test DB.

    Args:
        integration_db: Path to the test database.

    Returns:
        An ``AgentConfig`` configured for testing.
    """
    return AgentConfig(
        provider="ollama",
        model_name="llama3.1:8b",
        temperature=0.1,
        recursion_limit=10,
        db_path=integration_db,
        max_history_turns=5,
    )


def _make_tool_call_msg(
    tool_name: str,
    args: dict[str, Any],
    tool_call_id: str = "tc_1",
) -> AIMessage:
    """Build an AIMessage with a single tool call.

    Args:
        tool_name: Name of the tool to call.
        args: Arguments dict for the tool.
        tool_call_id: Unique tool call identifier.

    Returns:
        An ``AIMessage`` with the tool call attached.
    """
    return AIMessage(
        content="",
        tool_calls=[{"id": tool_call_id, "name": tool_name, "args": args}],
    )


def _make_final_msg(narrative: str, claims: list[dict] | None = None) -> AIMessage:
    """Build an AIMessage with structured JSON content.

    Args:
        narrative: Prose answer text.
        claims: Optional list of claim dicts. Defaults to empty.

    Returns:
        An ``AIMessage`` with JSON-serialized content.
    """
    payload: dict[str, Any] = {
        "narrative": narrative,
        "claims": claims or [],
    }
    return AIMessage(content=json.dumps(payload))


# ---------------------------------------------------------------------------
# 1. Pure data question
# ---------------------------------------------------------------------------


class TestPureDataQuestion:
    """Agent answers a data question using the SQL tool."""

    def test_sql_tool_called_and_claim_verified(
        self, integration_config: AgentConfig
    ) -> None:
        """SQL tool is called, claim is parsed, verification passes."""
        tool_call = _make_tool_call_msg(
            "execute_sql",
            {"sql": "SELECT date, value_covid_adjusted FROM observations WHERE series_id = 'UNRATE' ORDER BY date DESC LIMIT 1"},
        )
        tool_result = ToolMessage(
            content=json.dumps({
                "columns": ["date", "value_covid_adjusted"],
                "rows": [["2026-02-01", 4.2]],
                "row_count": 1,
                "truncated": False,
            }),
            name="execute_sql",
            tool_call_id="tc_1",
        )
        final = _make_final_msg(
            "The latest unemployment rate is 4.2% as of February 2026.",
            [
                {
                    "statement": "unemployment at 4.2%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.2,
                    "date_range": ["2026-02", "2026-02"],
                }
            ],
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "messages": [tool_call, tool_result, final]
        }

        result = run_agent(
            "What is the latest unemployment rate?",
            integration_config,
            compiled_graph=mock_compiled,
        )

        assert isinstance(result, AgentResponse)
        assert "4.2%" in result.answer
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "execute_sql"
        assert len(result.claims) == 1
        assert result.verification["status"] == "Verified"
        assert result.verification["all_verified"] is True
        assert result.verification["passed_count"] == 1

    def test_wrong_claim_fails_verification(
        self, integration_config: AgentConfig
    ) -> None:
        """A claim with incorrect expected_value fails verification."""
        final = _make_final_msg(
            "The latest unemployment rate is 5.0%.",
            [
                {
                    "statement": "unemployment at 5.0%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 5.0,
                    "date_range": ["2026-02", "2026-02"],
                }
            ],
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent(
            "What is unemployment?",
            integration_config,
            compiled_graph=mock_compiled,
        )

        assert result.verification["status"] == "Unverified"
        assert result.verification["all_verified"] is False
        assert result.verification["passed_count"] == 0

    def test_multiple_claims_partial_verification(
        self, integration_config: AgentConfig
    ) -> None:
        """One correct and one wrong claim produces Partially Verified."""
        final = _make_final_msg(
            "UNRATE is 4.2% and T10Y2Y is 1.5%.",
            [
                {
                    "statement": "unemployment at 4.2%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.2,
                    "date_range": ["2026-02", "2026-02"],
                },
                {
                    "statement": "yield spread at 1.5%",
                    "metric_type": "latest",
                    "series_id": "T10Y2Y",
                    "expected_value": 1.5,
                    "date_range": ["2026-02", "2026-02"],
                },
            ],
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent(
            "Tell me about UNRATE and T10Y2Y",
            integration_config,
            compiled_graph=mock_compiled,
        )

        assert result.verification["status"] == "Partially Verified"
        assert result.verification["total"] == 2
        assert result.verification["passed_count"] == 1


# ---------------------------------------------------------------------------
# 2. Conceptual question
# ---------------------------------------------------------------------------


class TestConceptualQuestion:
    """Agent answers a conceptual question using the RAG tool."""

    def test_rag_tool_called_no_claims(
        self, integration_config: AgentConfig
    ) -> None:
        """RAG tool is called, no claims, badge is Verified."""
        rag_call = _make_tool_call_msg(
            "retrieve_context",
            {"query": "What is the yield curve?"},
            tool_call_id="tc_rag_1",
        )
        rag_result = ToolMessage(
            content=json.dumps({
                "chunks": [
                    {
                        "ref_id": 1,
                        "doc_id": 1,
                        "title": "The Yield Curve Explained",
                        "doc_type": "concept:yield_curve",
                        "content": "The yield curve plots Treasury yields...",
                        "source_url": "",
                        "score": 0.2,
                    }
                ],
                "reference_block": 'REFERENCE CONTEXT:\n[ref:1] The Yield Curve Explained — "The yield curve plots..."',
            }),
            name="retrieve_context",
            tool_call_id="tc_rag_1",
        )
        final = _make_final_msg(
            "The yield curve plots U.S. Treasury bond yields across "
            "different maturities [ref:1]. An inverted curve has preceded "
            "every U.S. recession since 1970.",
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "messages": [rag_call, rag_result, final]
        }

        result = run_agent(
            "What is the yield curve?",
            integration_config,
            compiled_graph=mock_compiled,
        )

        assert "yield curve" in result.answer.lower()
        assert result.claims == []
        assert result.verification["status"] == "Verified"
        assert result.verification["total"] == 0
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0]["name"] == "retrieve_context"

    def test_rag_references_captured_in_tool_results(
        self, integration_config: AgentConfig
    ) -> None:
        """RAG tool results are captured for UI display."""
        rag_call = _make_tool_call_msg(
            "retrieve_context",
            {"query": "CPI methodology"},
            tool_call_id="tc_rag_2",
        )
        rag_result = ToolMessage(
            content=json.dumps({
                "chunks": [{"ref_id": 1, "title": "CPI Docs", "content": "..."}],
                "reference_block": "REFERENCE CONTEXT:\n[ref:1] CPI Docs",
            }),
            name="retrieve_context",
            tool_call_id="tc_rag_2",
        )
        final = _make_final_msg("CPI measures consumer inflation [ref:1].")

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "messages": [rag_call, rag_result, final]
        }

        result = run_agent(
            "How is CPI measured?",
            integration_config,
            compiled_graph=mock_compiled,
        )

        assert len(result.tool_results) == 1
        assert result.tool_results[0]["name"] == "retrieve_context"
        assert "CPI" in result.tool_results[0]["content"]


# ---------------------------------------------------------------------------
# 3. Combined question (SQL + RAG)
# ---------------------------------------------------------------------------


class TestCombinedQuestion:
    """Agent uses both SQL and RAG tools for a combined question."""

    def test_multiple_tool_calls_verified(
        self, integration_config: AgentConfig
    ) -> None:
        """SQL + RAG called, multiple claims all verified."""
        sql_call = _make_tool_call_msg(
            "execute_sql",
            {"sql": "SELECT date, value_covid_adjusted FROM observations WHERE series_id = 'T10Y2Y' AND value_covid_adjusted < 0 ORDER BY date DESC LIMIT 1"},
            tool_call_id="tc_sql_1",
        )
        sql_result = ToolMessage(
            content=json.dumps({
                "columns": ["date", "value_covid_adjusted"],
                "rows": [["2025-12-01", -0.10]],
                "row_count": 1,
                "truncated": False,
            }),
            name="execute_sql",
            tool_call_id="tc_sql_1",
        )
        sql_call_2 = _make_tool_call_msg(
            "execute_sql",
            {"sql": "SELECT date, value_covid_adjusted FROM observations WHERE series_id = 'UNRATE' ORDER BY date DESC LIMIT 1"},
            tool_call_id="tc_sql_2",
        )
        sql_result_2 = ToolMessage(
            content=json.dumps({
                "columns": ["date", "value_covid_adjusted"],
                "rows": [["2026-02-01", 4.2]],
                "row_count": 1,
                "truncated": False,
            }),
            name="execute_sql",
            tool_call_id="tc_sql_2",
        )
        rag_call = _make_tool_call_msg(
            "retrieve_context",
            {"query": "yield curve recession indicator"},
            tool_call_id="tc_rag_1",
        )
        rag_result = ToolMessage(
            content=json.dumps({
                "chunks": [{"ref_id": 1, "title": "Yield Curve", "content": "..."}],
                "reference_block": "REFERENCE CONTEXT:\n[ref:1] Yield Curve",
            }),
            name="retrieve_context",
            tool_call_id="tc_rag_1",
        )
        final = _make_final_msg(
            "The yield curve was last inverted in December 2025 [ref:1]. "
            "The latest unemployment rate is 4.2%.",
            [
                {
                    "statement": "unemployment at 4.2%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.2,
                    "date_range": ["2026-02", "2026-02"],
                },
            ],
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "messages": [
                sql_call, sql_result,
                sql_call_2, sql_result_2,
                rag_call, rag_result,
                final,
            ]
        }

        result = run_agent(
            "When did the yield curve invert and what happened to unemployment?",
            integration_config,
            compiled_graph=mock_compiled,
        )

        assert len(result.tool_calls) == 3
        tool_names = [tc["name"] for tc in result.tool_calls]
        assert "execute_sql" in tool_names
        assert "retrieve_context" in tool_names
        assert result.verification["status"] == "Verified"
        assert result.verification["passed_count"] == 1

    def test_tool_results_include_both_types(
        self, integration_config: AgentConfig
    ) -> None:
        """Both SQL and RAG tool results appear in tool_results."""
        sql_call = _make_tool_call_msg(
            "execute_sql", {"sql": "SELECT 1"}, tool_call_id="tc_s",
        )
        sql_result = ToolMessage(
            content=json.dumps({"rows": [[1]], "columns": ["1"]}),
            name="execute_sql",
            tool_call_id="tc_s",
        )
        rag_call = _make_tool_call_msg(
            "retrieve_context", {"query": "test"}, tool_call_id="tc_r",
        )
        rag_result = ToolMessage(
            content=json.dumps({"chunks": [], "reference_block": "No sources"}),
            name="retrieve_context",
            tool_call_id="tc_r",
        )
        final = _make_final_msg("Combined answer.")

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "messages": [sql_call, sql_result, rag_call, rag_result, final]
        }

        result = run_agent("combined", integration_config, compiled_graph=mock_compiled)

        result_names = [tr["name"] for tr in result.tool_results]
        assert "execute_sql" in result_names
        assert "retrieve_context" in result_names


# ---------------------------------------------------------------------------
# 4. Off-topic question
# ---------------------------------------------------------------------------


class TestOffTopicQuestion:
    """Agent refuses off-topic questions with plain text."""

    def test_plain_text_refusal_no_tools(
        self, integration_config: AgentConfig
    ) -> None:
        """Off-topic returns plain text, no tool calls, no claims."""
        refusal = "I can only answer questions about the economic data in this dashboard."
        final = AIMessage(content=refusal)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent(
            "What's the weather in Tampa?",
            integration_config,
            compiled_graph=mock_compiled,
        )

        assert "only answer questions" in result.answer
        assert result.tool_calls == []
        assert result.claims == []
        assert result.verification["status"] == "Verified"
        assert result.verification["total"] == 0

    def test_no_verification_badge_for_refusal(
        self, integration_config: AgentConfig
    ) -> None:
        """Refusal gets Verified with zero claims (no badge needed)."""
        final = AIMessage(content="That's outside my scope.")

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("Who won the Super Bowl?", integration_config, compiled_graph=mock_compiled)

        assert result.verification["all_verified"] is True
        assert result.verification["results"] == []


# ---------------------------------------------------------------------------
# 5. Follow-up question (conversation history)
# ---------------------------------------------------------------------------


class TestFollowUpQuestion:
    """Agent uses conversation history for follow-up questions."""

    def test_history_passed_to_graph(
        self, integration_config: AgentConfig
    ) -> None:
        """Conversation history is included in graph invocation messages."""
        final = AIMessage(content="The rate increased by 0.1pp.")

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        history = [
            {"role": "user", "content": "What is the unemployment rate?"},
            {"role": "assistant", "content": "The unemployment rate is 4.2%."},
        ]

        result = run_agent(
            "How has that changed?",
            integration_config,
            history=history,
            compiled_graph=mock_compiled,
        )

        # Verify the graph received messages including history.
        call_kwargs = mock_compiled.invoke.call_args
        messages = call_kwargs[0][0]["messages"]
        contents = [m.content for m in messages]
        assert "What is the unemployment rate?" in contents
        assert "The unemployment rate is 4.2%." in contents
        assert "How has that changed?" in contents
        assert isinstance(result, AgentResponse)

    def test_sliding_window_limits_history(
        self, integration_config: AgentConfig
    ) -> None:
        """Only the last max_history_turns are sent to the graph."""
        integration_config.max_history_turns = 2
        final = AIMessage(content="Answer.")

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        history = [
            {"role": "user", "content": "old q1"},
            {"role": "assistant", "content": "old a1"},
            {"role": "user", "content": "old q2"},
            {"role": "assistant", "content": "old a2"},
            {"role": "user", "content": "recent q"},
            {"role": "assistant", "content": "recent a"},
        ]

        run_agent("new q", integration_config, history=history, compiled_graph=mock_compiled)

        messages = mock_compiled.invoke.call_args[0][0]["messages"]
        # System + 2 history turns + new question = 4
        assert len(messages) == 4
        assert messages[1].content == "recent q"
        assert messages[2].content == "recent a"
        assert messages[3].content == "new q"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    """Edge cases: empty DB results, SQL errors, malformed LLM output."""

    def test_empty_claims_array_verified(
        self, integration_config: AgentConfig
    ) -> None:
        """Empty claims array results in Verified status."""
        final = _make_final_msg("The economy looks stable.", [])

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("How's the economy?", integration_config, compiled_graph=mock_compiled)

        assert result.verification["status"] == "Verified"
        assert result.verification["total"] == 0

    def test_malformed_json_preserves_raw_text(
        self, integration_config: AgentConfig
    ) -> None:
        """Malformed JSON in LLM output is treated as raw text."""
        malformed = '{"narrative": "partial json...'
        final = AIMessage(content=malformed)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("test", integration_config, compiled_graph=mock_compiled)

        assert result.answer == malformed
        assert result.claims == []

    def test_claims_missing_required_fields_skipped(
        self, integration_config: AgentConfig
    ) -> None:
        """Claims missing required fields are silently skipped."""
        final = _make_final_msg(
            "Something about the data.",
            [
                {"statement": "incomplete claim"},  # missing metric_type, series_id, expected_value
                {
                    "statement": "unemployment at 4.2%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.2,
                    "date_range": ["2026-02", "2026-02"],
                },
            ],
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("test", integration_config, compiled_graph=mock_compiled)

        assert len(result.claims) == 1
        assert result.claims[0]["series_id"] == "UNRATE"

    def test_direction_claim_none_expected_value(
        self, integration_config: AgentConfig
    ) -> None:
        """Direction claims work and produce verification results."""
        final = _make_final_msg(
            "Unemployment has been rising.",
            [
                {
                    "statement": "unemployment trending upward",
                    "metric_type": "direction",
                    "series_id": "UNRATE",
                    "expected_value": 1,
                    "date_range": ["2025-12", "2026-02"],
                },
            ],
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("test", integration_config, compiled_graph=mock_compiled)

        assert result.verification["total"] == 1
        # The direction claim should be processed (pass or fail depending on data).
        assert len(result.verification["results"]) == 1

    def test_recursion_limit_graceful_message(
        self, integration_config: AgentConfig
    ) -> None:
        """Hitting recursion limit returns a graceful error message."""
        integration_config.recursion_limit = 2

        with patch("src.agent.graph._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = AIMessage(
                content="",
                tool_calls=[{
                    "id": "tc_loop",
                    "name": "execute_sql",
                    "args": {"sql": "SELECT 1"},
                }],
            )
            mock_build.return_value = mock_llm

            result = run_agent("endless loop", integration_config)

        assert "couldn't answer" in result.answer.lower()
        assert result.claims == []

    def test_json_in_markdown_fence_parsed(
        self, integration_config: AgentConfig
    ) -> None:
        """JSON wrapped in markdown code fences is still parsed."""
        wrapped = '```json\n{"narrative": "Rate is 4.2%.", "claims": [{"statement": "rate", "metric_type": "latest", "series_id": "UNRATE", "expected_value": 4.2, "date_range": ["2026-02", "2026-02"]}]}\n```'
        final = AIMessage(content=wrapped)

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("test", integration_config, compiled_graph=mock_compiled)

        assert result.answer == "Rate is 4.2%."
        assert len(result.claims) == 1
        assert result.verification["status"] == "Verified"

    def test_claim_outside_tolerance_fails(
        self, integration_config: AgentConfig
    ) -> None:
        """A claim with value far outside tolerance fails verification."""
        final = _make_final_msg(
            "Unemployment is 10%.",
            [
                {
                    "statement": "unemployment at 10%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 10.0,
                    "date_range": ["2026-02", "2026-02"],
                },
            ],
        )

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("test", integration_config, compiled_graph=mock_compiled)

        assert result.verification["status"] == "Unverified"
        assert result.verification["passed_count"] == 0

    def test_no_tool_calls_no_tool_results(
        self, integration_config: AgentConfig
    ) -> None:
        """A direct answer without tools has empty tool logs."""
        final = _make_final_msg("The economy is complex.")

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("test", integration_config, compiled_graph=mock_compiled)

        assert result.tool_calls == []
        assert result.tool_results == []

    def test_response_has_elapsed_seconds(
        self, integration_config: AgentConfig
    ) -> None:
        """AgentResponse includes elapsed_seconds field."""
        final = _make_final_msg("Answer.")

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {"messages": [final]}

        result = run_agent("test", integration_config, compiled_graph=mock_compiled)

        assert hasattr(result, "elapsed_seconds")
        assert isinstance(result.elapsed_seconds, float)
        assert result.elapsed_seconds >= 0.0
