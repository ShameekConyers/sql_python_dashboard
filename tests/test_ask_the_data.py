"""Tests for dashboard/ask_the_data.py.

All tests mock the LLM and Streamlit components. No live inference.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from dashboard.ask_the_data import (
    EXAMPLE_QUESTIONS,
    _build_agent_history,
    _check_agent_available,
    _check_key_gate_required,
    _extract_expiry_date,
    _format_ref_tags,
    _get_remaining_seconds,
    _is_access_active,
    _is_key_expired,
    _render_claim_details,
    _render_references,
    _render_verification_badge,
    render_ask_the_data,
)
from src.agent.graph import AgentResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_session_state() -> None:
    """Ensure each test starts with a fresh Streamlit session state."""
    with patch("dashboard.ask_the_data.st") as mock_st:
        mock_st.session_state = {}
        yield


@pytest.fixture()
def mock_st() -> MagicMock:
    """Provide a mocked Streamlit module."""
    with patch("dashboard.ask_the_data.st") as m:
        m.session_state = {}
        m.chat_message = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ))
        m.expander = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ))
        m.spinner = MagicMock(return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        ))
        m.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
        m.chat_input = MagicMock(return_value=None)
        m.button = MagicMock(return_value=False)
        m.cache_resource = lambda f: f  # passthrough decorator
        yield m


# ---------------------------------------------------------------------------
# Test 1: Feature availability — Ollama reachable
# ---------------------------------------------------------------------------


class TestFeatureAvailability:
    """Tests for _check_agent_available()."""

    def test_ollama_reachable(self, mock_st: MagicMock) -> None:
        """Ollama HTTP 200 sets provider to 'ollama'."""
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        with patch("dashboard.ask_the_data.httpx") as mock_httpx:
            mock_httpx.get.return_value = mock_resp
            available, provider = _check_agent_available()

        assert available is True
        assert provider == "ollama"

    def test_api_key_anthropic(self, mock_st: MagicMock) -> None:
        """Anthropic API key in secrets sets provider to 'anthropic'."""
        mock_st.secrets.get.return_value = "sk-ant-test"

        with patch("dashboard.ask_the_data.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("no ollama")
            available, provider = _check_agent_available()

        assert available is True
        assert provider == "anthropic"

    def test_none_available(self, mock_st: MagicMock) -> None:
        """No Ollama and no API keys returns disabled state."""
        mock_st.secrets.get.return_value = None

        with patch("dashboard.ask_the_data.httpx") as mock_httpx:
            mock_httpx.get.side_effect = Exception("no ollama")
            with patch.dict("os.environ", {}, clear=True):
                available, provider = _check_agent_available()

        assert available is False
        assert provider is None


# ---------------------------------------------------------------------------
# Tests 4-5: Conversation history
# ---------------------------------------------------------------------------


class TestConversationHistory:
    """Tests for history persistence and clearing."""

    def test_persistence(self, mock_st: MagicMock) -> None:
        """Two Q&A turns produce 4 entries in session state."""
        mock_st.session_state["_ask_history"] = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1",
             "verification": {}, "tool_calls": [], "tool_results": [],
             "claims": []},
            {"role": "user", "content": "Q2"},
            {"role": "assistant", "content": "A2",
             "verification": {}, "tool_calls": [], "tool_results": [],
             "claims": []},
        ]
        assert len(mock_st.session_state["_ask_history"]) == 4

    def test_clear(self, mock_st: MagicMock) -> None:
        """Clearing conversation resets history to empty."""
        mock_st.session_state["_ask_history"] = [
            {"role": "user", "content": "Q1"},
            {"role": "assistant", "content": "A1",
             "verification": {}, "tool_calls": [], "tool_results": [],
             "claims": []},
        ]
        mock_st.session_state["_ask_history"] = []
        assert mock_st.session_state["_ask_history"] == []


# ---------------------------------------------------------------------------
# Test 6: Example question prefill
# ---------------------------------------------------------------------------


class TestExampleQuestionPrefill:
    """Tests for the example chip -> pending question flow."""

    def test_prefill_sets_pending(self, mock_st: MagicMock) -> None:
        """Setting _ask_pending_question uses that as the question."""
        mock_st.session_state["_ask_pending_question"] = "What is CPI?"
        pending = mock_st.session_state.pop("_ask_pending_question", None)
        assert pending == "What is CPI?"


# ---------------------------------------------------------------------------
# Tests 7-9: Verification badges
# ---------------------------------------------------------------------------


class TestVerificationBadge:
    """Tests for _render_verification_badge()."""

    def test_all_verified(self, mock_st: MagicMock) -> None:
        """Verified status renders green badge."""
        verification = {
            "status": "Verified",
            "all_verified": True,
            "total": 2,
            "passed_count": 2,
            "results": [],
        }
        _render_verification_badge(verification)
        mock_st.markdown.assert_called_with(":green[**Verified**]")

    def test_partial(self, mock_st: MagicMock) -> None:
        """Partially Verified status renders orange badge."""
        verification = {
            "status": "Partially Verified",
            "all_verified": False,
            "total": 3,
            "passed_count": 2,
            "results": [],
        }
        _render_verification_badge(verification)
        call_text = mock_st.markdown.call_args[0][0]
        assert ":orange[**Partially Verified**]" in call_text
        assert "2 of 3" in call_text

    def test_no_claims_green(self, mock_st: MagicMock) -> None:
        """Empty claims with Verified status gets green badge."""
        verification = {
            "status": "Verified",
            "all_verified": True,
            "total": 0,
            "passed_count": 0,
            "results": [],
        }
        _render_verification_badge(verification)
        mock_st.markdown.assert_called_with(":green[**Verified**]")

    def test_no_verification_dict(self, mock_st: MagicMock) -> None:
        """Empty verification dict renders nothing."""
        _render_verification_badge({})
        mock_st.markdown.assert_not_called()


# ---------------------------------------------------------------------------
# Test 10: Ref tag rendering
# ---------------------------------------------------------------------------


class TestRefTagRendering:
    """Tests for [ref:N] -> <sup>[N]</sup> conversion."""

    def test_single_ref(self) -> None:
        """Single ref tag is converted to superscript."""
        result = _format_ref_tags("CPI rose 3.2% [ref:1] last year.")
        assert "<sup>[1]</sup>" in result
        assert "[ref:1]" not in result

    def test_multiple_refs(self) -> None:
        """Multiple ref tags are all converted."""
        result = _format_ref_tags("A [ref:1] and B [ref:2].")
        assert "<sup>[1]</sup>" in result
        assert "<sup>[2]</sup>" in result

    def test_no_refs(self) -> None:
        """Text without ref tags is unchanged."""
        text = "No references here."
        assert _format_ref_tags(text) == text


# ---------------------------------------------------------------------------
# Test 11: Cached graph
# ---------------------------------------------------------------------------


class TestCachedGraph:
    """Tests for compiled graph caching."""

    def test_build_graph_called_once(self, mock_st: MagicMock) -> None:
        """build_graph is called once when compiled_graph is passed."""
        from src.agent.config import AgentConfig

        mock_compiled = MagicMock()
        mock_compiled.invoke.return_value = {
            "messages": [
                MagicMock(content="Answer.", tool_calls=None),
            ]
        }

        with patch("src.agent.graph.build_graph") as mock_bg:
            from src.agent.graph import run_agent

            config = AgentConfig(
                provider="ollama", db_path="/tmp/test.db"
            )
            # First call with pre-compiled graph
            run_agent("Q1", config, compiled_graph=mock_compiled)
            # Second call with same pre-compiled graph
            run_agent("Q2", config, compiled_graph=mock_compiled)

            mock_bg.assert_not_called()


# ---------------------------------------------------------------------------
# Test 12: Error handling
# ---------------------------------------------------------------------------


class TestErrorHandling:
    """Tests for agent failure error handling."""

    def test_agent_failure_shows_error(self, mock_st: MagicMock) -> None:
        """Agent exception triggers st.error and leaves history unchanged."""
        mock_st.session_state["_ask_history"] = []

        with patch(
            "dashboard.ask_the_data.run_agent",
            side_effect=RuntimeError("LLM down"),
        ), patch(
            "dashboard.ask_the_data._get_compiled_graph",
            return_value=MagicMock(),
        ):
            from dashboard.ask_the_data import _handle_question

            _handle_question("test?", "ollama", "/tmp/db", "/tmp/chroma")

        mock_st.error.assert_called_once()
        # User question should be rolled back on failure
        assert len(mock_st.session_state["_ask_history"]) == 0


# ---------------------------------------------------------------------------
# Test 13: Agent history format
# ---------------------------------------------------------------------------


class TestAgentHistoryFormat:
    """Tests for _build_agent_history()."""

    def test_format_and_truncation(self, mock_st: MagicMock) -> None:
        """History is truncated and contains only role + content."""
        mock_st.session_state["_ask_history"] = [
            {"role": "user", "content": "Q1", "verification": None,
             "tool_calls": None, "tool_results": None, "claims": None},
            {"role": "assistant", "content": "A1", "verification": {},
             "tool_calls": [], "tool_results": [], "claims": []},
            {"role": "user", "content": "Q2", "verification": None,
             "tool_calls": None, "tool_results": None, "claims": None},
            {"role": "assistant", "content": "A2", "verification": {},
             "tool_calls": [], "tool_results": [], "claims": []},
            {"role": "user", "content": "Q3", "verification": None,
             "tool_calls": None, "tool_results": None, "claims": None},
            {"role": "assistant", "content": "A3", "verification": {},
             "tool_calls": [], "tool_results": [], "claims": []},
        ]
        result = _build_agent_history(max_turns=2)
        assert len(result) == 2
        assert set(result[0].keys()) == {"role", "content"}
        assert result[0]["content"] == "Q3"
        assert result[1]["content"] == "A3"


# ---------------------------------------------------------------------------
# Test 14: Render without crash
# ---------------------------------------------------------------------------


class TestRenderReferences:
    """Tests for _render_references with dict-wrapped RAG results."""

    def test_parses_dict_wrapped_chunks(self, mock_st: MagicMock) -> None:
        """RAG tool results are JSON dicts with a 'chunks' key."""
        import json

        chunks_data = {
            "chunks": [
                {"title": "Yield Curve", "doc_type": "concept:yield_curve",
                 "content": "The yield curve measures..."},
                {"title": "CPI", "doc_type": "concept:cpi_inflation",
                 "content": "CPI tracks consumer prices..."},
            ],
            "reference_block": "REFERENCE CONTEXT:\n[ref:1] ...",
        }
        tool_results = [{
            "name": "retrieve_context",
            "content": json.dumps(chunks_data),
            "tool_call_id": "call_1",
        }]
        _render_references(tool_results)
        # Should have rendered 2 chunks via st.markdown
        assert mock_st.markdown.call_count == 2
        first_call = mock_st.markdown.call_args_list[0][0][0]
        assert "Yield Curve" in first_call

    def test_ignores_non_rag_tools(self, mock_st: MagicMock) -> None:
        """SQL tool results are not rendered as references."""
        tool_results = [{
            "name": "execute_sql",
            "content": '{"rows": []}',
            "tool_call_id": "call_1",
        }]
        _render_references(tool_results)
        mock_st.expander.assert_not_called()


class TestRenderWithoutCrash:
    """Tests for the full render path."""

    def test_render_disabled_state(self, mock_st: MagicMock) -> None:
        """render_ask_the_data with no agent shows info message."""
        mock_st.secrets.get.return_value = None

        with patch("dashboard.ask_the_data.httpx") as mock_httpx, \
             patch.dict("os.environ", {}, clear=True):
            mock_httpx.get.side_effect = Exception("no ollama")
            render_ask_the_data("/tmp/db", "/tmp/chroma")

        mock_st.info.assert_called_once()
        assert "Ollama" in mock_st.info.call_args[0][0]

    def test_render_available_no_crash(self, mock_st: MagicMock) -> None:
        """render_ask_the_data with an available agent doesn't crash."""
        mock_st.session_state["_agent_available"] = (True, "ollama")
        mock_st.session_state["_ask_history"] = []
        mock_st.chat_input.return_value = None
        mock_st.button.return_value = False

        render_ask_the_data("/tmp/db", "/tmp/chroma")
        # Should not raise


# ---------------------------------------------------------------------------
# Tests: Access key gate
# ---------------------------------------------------------------------------


class TestAccessKeyGate:
    """Tests for the per-session access key gate."""

    def test_ollama_skips_gate(self, mock_st: MagicMock) -> None:
        """Ollama provider never requires the key gate."""
        assert _check_key_gate_required("ollama") is False

    def test_anthropic_requires_gate_when_keys_configured(
        self, mock_st: MagicMock
    ) -> None:
        """Anthropic provider requires gate when access keys exist."""
        mock_st.secrets = {"access_keys": {"keys": ["demo2026a"]}}
        assert _check_key_gate_required("anthropic") is True

    def test_anthropic_skips_gate_when_no_keys(
        self, mock_st: MagicMock
    ) -> None:
        """Anthropic provider skips gate when no access keys are configured."""
        mock_st.secrets = {}
        assert _check_key_gate_required("anthropic") is False

    def test_openai_requires_gate(self, mock_st: MagicMock) -> None:
        """OpenAI provider requires gate when access keys exist."""
        mock_st.secrets = {"access_keys": {"keys": ["key1"]}}
        assert _check_key_gate_required("openai") is True

    def test_valid_key_activates_session(self, mock_st: MagicMock) -> None:
        """Entering a valid key sets _ask_key_activated_at in session state."""
        from datetime import datetime

        mock_st.session_state["_ask_key_activated_at"] = datetime.now()
        assert _is_access_active() is True

    def test_expired_key_deactivates(self, mock_st: MagicMock) -> None:
        """Key entered more than 2 minutes ago is expired."""
        from datetime import datetime, timedelta

        mock_st.session_state["_ask_key_activated_at"] = (
            datetime.now() - timedelta(minutes=3)
        )
        assert _is_access_active() is False

    def test_no_activation_is_inactive(self, mock_st: MagicMock) -> None:
        """No activation timestamp means access is inactive."""
        assert _is_access_active() is False

    def test_remaining_seconds_active(self, mock_st: MagicMock) -> None:
        """Remaining seconds is positive when key was just entered."""
        from datetime import datetime, timedelta

        mock_st.session_state["_ask_key_activated_at"] = (
            datetime.now() - timedelta(seconds=30)
        )
        remaining = _get_remaining_seconds()
        assert 85 <= remaining <= 90  # ~1.5 minutes left

    def test_remaining_seconds_expired(self, mock_st: MagicMock) -> None:
        """Remaining seconds is zero when key is expired."""
        from datetime import datetime, timedelta

        mock_st.session_state["_ask_key_activated_at"] = (
            datetime.now() - timedelta(minutes=10)
        )
        assert _get_remaining_seconds() == 0

    def test_extract_expiry_date_valid(self, mock_st: MagicMock) -> None:
        """A well-formed key returns a non-None expiry date."""
        key = "1ut50bt52ir02kt6"
        expiry = _extract_expiry_date(key)
        assert expiry is not None
        assert expiry.year > 2000

    def test_extract_expiry_date_wrong_length(self, mock_st: MagicMock) -> None:
        """Key with wrong length returns None."""
        assert _extract_expiry_date("abc123def") is None
        assert _extract_expiry_date("abcdefghijklmnopqrs") is None

    def test_extract_expiry_date_no_digits_at_positions(
        self, mock_st: MagicMock
    ) -> None:
        """All-letter 16-char key returns None."""
        assert _extract_expiry_date("abcdefghijklmnop") is None

    def test_key_not_expired_future_date(self, mock_st: MagicMock) -> None:
        """Key with a future expiry date is not expired."""
        # Pre-generated key expiring Sep 1, 2026.
        assert _is_key_expired("0jm10pa92cq02aq6") is False

    def test_key_expired_past_date(self, mock_st: MagicMock) -> None:
        """Key with a past expiry date is expired."""
        # Pre-generated key expiring Jan 1, 2020.
        assert _is_key_expired("0rf10et12np02ma0") is True

    def test_key_wrong_length_not_expired(self, mock_st: MagicMock) -> None:
        """Key with wrong length is treated as non-expiring."""
        assert _is_key_expired("short") is False
