"""Tests for src/agent/graph.py.

All tests mock the LLM so they run without Ollama or API keys.
"""

from __future__ import annotations

import json
import sqlite3
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent.config import AgentConfig
from src.agent.graph import (
    AgentResponse,
    AgentState,
    _build_llm,
    _should_continue,
    build_graph,
    run_agent,
)
from src.agent.tools.sql_tool import make_sql_tool


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_db(tmp_path: pytest.TempPathFactory) -> str:
    """Create a tiny economic-style DB for agent tests."""
    db_path = str(tmp_path / "agent_test.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE series_metadata "
        "(series_id TEXT PRIMARY KEY, name TEXT, category TEXT, "
        "units TEXT, frequency TEXT)"
    )
    conn.execute(
        "INSERT INTO series_metadata VALUES "
        "('UNRATE', 'Unemployment Rate', 'Labor', 'Percent', 'Monthly')"
    )
    conn.execute(
        "CREATE TABLE observations "
        "(id INTEGER PRIMARY KEY, series_id TEXT, date TEXT, "
        "value REAL, value_covid_adjusted REAL)"
    )
    conn.executemany(
        "INSERT INTO observations (series_id, date, value, "
        "value_covid_adjusted) VALUES (?, ?, ?, ?)",
        [
            ("UNRATE", "2025-12-01", 4.2, 4.2),
            ("UNRATE", "2026-01-01", 4.3, 4.3),
            ("UNRATE", "2026-02-01", 4.4, 4.4),
        ],
    )
    conn.commit()
    conn.close()
    return db_path


@pytest.fixture()
def test_config(test_db: str) -> AgentConfig:
    """Config pointing at the test DB with ollama provider."""
    return AgentConfig(
        provider="ollama",
        model_name="llama3.1:8b",
        temperature=0.1,
        recursion_limit=6,
        db_path=test_db,
        max_history_turns=5,
    )


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


class TestShouldContinue:
    """Tests for the conditional routing function."""

    def test_routes_to_tools_when_tool_calls_present(self) -> None:
        """Agent message with tool_calls routes to 'tools'."""
        msg = AIMessage(
            content="",
            tool_calls=[
                {
                    "id": "tc1",
                    "name": "execute_sql",
                    "args": {"sql": "SELECT 1"},
                }
            ],
        )
        state: AgentState = {"messages": [msg]}
        assert _should_continue(state) == "tools"

    def test_routes_to_end_when_no_tool_calls(self) -> None:
        """Agent message without tool_calls routes to 'end'."""
        msg = AIMessage(content="The unemployment rate is 4.4%.")
        state: AgentState = {"messages": [msg]}
        assert _should_continue(state) == "end"


class TestBuildLlm:
    """Tests for the LLM factory function."""

    def test_ollama_provider(self) -> None:
        """Ollama provider creates a ChatOllama instance."""
        config = AgentConfig(provider="ollama", model_name="llama3.1:8b")
        llm = _build_llm(config)
        assert type(llm).__name__ == "ChatOllama"

    def test_anthropic_provider(self) -> None:
        """Anthropic provider creates a ChatAnthropic instance."""
        config = AgentConfig(
            provider="anthropic", model_name="claude-sonnet-4-20250514"
        )
        llm = _build_llm(config)
        assert type(llm).__name__ == "ChatAnthropic"

    def test_invalid_provider_raises(self) -> None:
        """An unknown provider raises ValueError."""
        config = AgentConfig(provider="unknown", model_name="foo")
        with pytest.raises(ValueError, match="Unsupported provider"):
            _build_llm(config)


class TestBuildGraph:
    """Tests for graph construction."""

    def test_graph_compiles(self, test_config: AgentConfig) -> None:
        """build_graph returns a compiled graph without errors."""
        with patch(
            "src.agent.graph._build_llm"
        ) as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_build.return_value = mock_llm
            graph = build_graph(test_config)
            assert graph is not None

    def test_sql_tool_bound(self, test_config: AgentConfig) -> None:
        """The SQL tool is bound to the LLM during graph construction."""
        with patch(
            "src.agent.graph._build_llm"
        ) as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_build.return_value = mock_llm
            build_graph(test_config)
            mock_llm.bind_tools.assert_called_once()
            tools = mock_llm.bind_tools.call_args[0][0]
            assert len(tools) == 1
            assert tools[0].name == "execute_sql"


class TestAgentResponse:
    """Tests for the AgentResponse dataclass."""

    def test_defaults(self) -> None:
        """AgentResponse has sensible defaults for Phase 16."""
        resp = AgentResponse(answer="test")
        assert resp.answer == "test"
        assert resp.tool_calls == []
        assert resp.claims == []
        assert resp.verification == {}


class TestRunAgent:
    """Tests for the run_agent entry point with mocked LLM."""

    def test_direct_answer(self, test_config: AgentConfig) -> None:
        """Agent returns a prose answer when the LLM doesn't call tools."""
        answer_text = "The latest unemployment rate is 4.4% as of Feb 2026."
        mock_response = AIMessage(content=answer_text)

        with patch("src.agent.graph._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            mock_build.return_value = mock_llm

            result = run_agent("What is the unemployment rate?", test_config)

        assert isinstance(result, AgentResponse)
        assert result.answer == answer_text
        assert result.tool_calls == []

    def test_off_topic_refusal(self, test_config: AgentConfig) -> None:
        """Agent returns a refusal for off-topic questions."""
        refusal = (
            "I can only answer questions about the economic data "
            "in this dashboard."
        )
        mock_response = AIMessage(content=refusal)

        with patch("src.agent.graph._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            mock_build.return_value = mock_llm

            result = run_agent("What's the weather in Tampa?", test_config)

        assert "only answer questions" in result.answer

    def test_history_truncation(self, test_config: AgentConfig) -> None:
        """Only the last max_history_turns are included in messages."""
        test_config.max_history_turns = 2
        history = [
            {"role": "user", "content": "old question 1"},
            {"role": "assistant", "content": "old answer 1"},
            {"role": "user", "content": "old question 2"},
            {"role": "assistant", "content": "old answer 2"},
            {"role": "user", "content": "recent question"},
            {"role": "assistant", "content": "recent answer"},
        ]
        mock_response = AIMessage(content="Answer with history.")

        with patch("src.agent.graph._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            mock_build.return_value = mock_llm

            run_agent("new question", test_config, history=history)

            # Check the messages passed to invoke.
            call_args = mock_llm.invoke.call_args[0][0]
            # System + 2 history turns + user question = 1 + 2 + 1 = 4
            assert len(call_args) == 4
            assert isinstance(call_args[0], SystemMessage)
            assert call_args[1].content == "recent question"
            assert call_args[2].content == "recent answer"
            assert call_args[3].content == "new question"

    def test_recursion_limit_graceful(
        self, test_config: AgentConfig
    ) -> None:
        """Hitting the recursion limit returns a graceful message."""
        test_config.recursion_limit = 2

        with patch("src.agent.graph._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            # Always return a tool call so the loop never ends.
            mock_llm.invoke.return_value = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "execute_sql",
                        "args": {"sql": "SELECT 1"},
                    }
                ],
            )
            mock_build.return_value = mock_llm

            result = run_agent("endless loop", test_config)

        assert "couldn't answer" in result.answer.lower()

    def test_structured_json_populates_claims(
        self, test_config: AgentConfig
    ) -> None:
        """LLM returning structured JSON populates claims and verification."""
        llm_output = json.dumps({
            "narrative": "The latest unemployment rate is 4.4%.",
            "claims": [
                {
                    "statement": "unemployment at 4.4%",
                    "metric_type": "latest",
                    "series_id": "UNRATE",
                    "expected_value": 4.4,
                    "date_range": ["2026-02", "2026-02"],
                }
            ],
        })
        mock_response = AIMessage(content=llm_output)

        with patch("src.agent.graph._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            mock_build.return_value = mock_llm

            result = run_agent("What is unemployment?", test_config)

        assert result.answer == "The latest unemployment rate is 4.4%."
        assert len(result.claims) == 1
        assert result.claims[0]["series_id"] == "UNRATE"
        assert result.verification["status"] == "Verified"
        assert result.verification["all_verified"] is True
        assert result.verification["total"] == 1
        assert result.verification["passed_count"] == 1

    def test_plain_text_empty_verification(
        self, test_config: AgentConfig
    ) -> None:
        """LLM returning plain text gets empty claims and Verified status."""
        refusal = (
            "I can only answer questions about the economic data "
            "in this dashboard."
        )
        mock_response = AIMessage(content=refusal)

        with patch("src.agent.graph._build_llm") as mock_build:
            mock_llm = MagicMock()
            mock_llm.bind_tools.return_value = mock_llm
            mock_llm.invoke.return_value = mock_response
            mock_build.return_value = mock_llm

            result = run_agent("What's the weather?", test_config)

        assert result.claims == []
        assert result.verification["status"] == "Verified"
        assert result.verification["all_verified"] is True
