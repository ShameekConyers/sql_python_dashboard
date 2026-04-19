"""Ask the Data — conversational AI interface for the economic dashboard.

Provides a Streamlit chat UI where users type natural-language questions
and get verified, cited answers from the LangGraph ReAct agent.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

import httpx
import streamlit as st

from src.agent.config import AgentConfig
from src.agent.graph import AgentResponse, build_graph, run_agent

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXAMPLE_QUESTIONS: list[str] = [
    "What is the latest unemployment rate?",
    "When was the last yield curve inversion?",
    "What is the current recession risk score?",
    "How does U6 compare to U3 right now?",
]

_REF_TAG_RE = re.compile(r"\[ref:(\d+)\]")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _check_agent_available() -> tuple[bool, str | None]:
    """Detect whether an LLM backend is reachable.

    Checks Ollama (HTTP ping), then Anthropic/OpenAI API keys in
    ``st.secrets`` or environment variables.

    Returns:
        A ``(available, provider)`` tuple. ``provider`` is ``None``
        when no backend is found.
    """
    if "_agent_available" in st.session_state:
        return st.session_state["_agent_available"]

    # 1. Ollama local server
    try:
        resp = httpx.get("http://localhost:11434/api/tags", timeout=2)
        if resp.status_code == 200:
            result: tuple[bool, str | None] = (True, "ollama")
            st.session_state["_agent_available"] = result
            return result
    except Exception:  # noqa: BLE001
        pass

    # 2. Anthropic API key
    api_key = _get_secret("ANTHROPIC_API_KEY")
    if api_key:
        result = (True, "anthropic")
        st.session_state["_agent_available"] = result
        return result

    # 3. OpenAI API key
    api_key = _get_secret("OPENAI_API_KEY")
    if api_key:
        result = (True, "openai")
        st.session_state["_agent_available"] = result
        return result

    result = (False, None)
    st.session_state["_agent_available"] = result
    return result


def _get_secret(key: str) -> str | None:
    """Read a secret from Streamlit secrets or the environment.

    Args:
        key: The secret name to look up.

    Returns:
        The secret value, or ``None`` if not found.
    """
    try:
        val = st.secrets.get(key)
        if val:
            return str(val)
    except Exception:  # noqa: BLE001
        pass
    return os.getenv(key)


def _build_agent_history(
    max_turns: int = 5,
) -> list[dict[str, str]]:
    """Convert session-state history to the format ``run_agent`` expects.

    Args:
        max_turns: Maximum number of recent turns to include.

    Returns:
        A list of dicts with ``"role"`` and ``"content"`` keys, trimmed
        to the last *max_turns* entries.
    """
    history: list[dict[str, Any]] = st.session_state.get("_ask_history", [])
    simple: list[dict[str, str]] = [
        {"role": h["role"], "content": h["content"]} for h in history
    ]
    return simple[-max_turns:]


@st.cache_resource
def _get_compiled_graph(
    provider: str,
    model_name: str,
    temperature: float,
    db_path: str,
) -> object:
    """Build and cache a compiled LangGraph graph.

    Args:
        provider: LLM provider name.
        model_name: Model identifier.
        temperature: Sampling temperature.
        db_path: Path to the SQLite database.

    Returns:
        A compiled ``StateGraph`` ready for invocation.
    """
    config = AgentConfig(
        provider=provider,
        model_name=model_name,
        temperature=temperature,
        db_path=db_path,
    )
    return build_graph(config)


def _render_verification_badge(verification: dict[str, Any]) -> None:
    """Render a colored verification badge based on claim results.

    Args:
        verification: Verification dict from ``AgentResponse``.
    """
    if not verification:
        return

    status = verification.get("status", "")
    total = verification.get("total", 0)
    passed = verification.get("passed_count", 0)

    if status == "Verified":
        st.markdown(":green[**Verified**]")
    elif status == "Partially Verified":
        st.markdown(
            f":orange[**Partially Verified**] — {passed} of {total} "
            f"claims confirmed."
        )
    elif status == "Unverified":
        st.markdown(":red[**Unverified**]")


def _render_claim_details(verification: dict[str, Any]) -> None:
    """Render a claim-level breakdown table inside an expander.

    Args:
        verification: Verification dict from ``AgentResponse``.
    """
    results = verification.get("results", [])
    if not results:
        return

    with st.expander("Show verification details"):
        rows: list[dict[str, str]] = []
        for r in results:
            rows.append({
                "Claim": r.get("statement", ""),
                "Expected": str(r.get("expected", "")),
                "Actual": str(r.get("actual_value", "")),
                "Status": "Passed" if r.get("passed") else "Failed",
            })
        st.table(rows)


def _render_references(tool_results: list[dict[str, Any]]) -> None:
    """Render RAG reference sources in an expander.

    Filters tool results for ``retrieve_context`` calls and displays
    the chunk metadata (title, doc_type, content snippet).

    Args:
        tool_results: List of tool result dicts from ``AgentResponse``.
    """
    rag_results = [
        tr for tr in tool_results if tr.get("name") == "retrieve_context"
    ]
    if not rag_results:
        return

    with st.expander("Show references"):
        for tr in rag_results:
            content = tr.get("content", "")
            try:
                parsed = json.loads(content)
                chunks = (
                    parsed.get("chunks", parsed)
                    if isinstance(parsed, dict)
                    else parsed
                )
                if isinstance(chunks, list):
                    for i, chunk in enumerate(chunks, 1):
                        title = chunk.get("title", "Untitled")
                        doc_type = chunk.get("doc_type", "")
                        snippet = chunk.get("content", "")[:200]
                        st.markdown(
                            f"**{i}. {title}** ({doc_type})\n\n"
                            f"> {snippet}..."
                        )
            except (json.JSONDecodeError, TypeError):
                st.text(content[:300])


def _format_ref_tags(text: str) -> str:
    """Convert ``[ref:N]`` tags to HTML superscript.

    Args:
        text: Answer text that may contain ``[ref:N]`` tags.

    Returns:
        Text with tags replaced by ``<sup>[N]</sup>``.
    """
    return _REF_TAG_RE.sub(r"<sup>[\1]</sup>", text)


def _render_history() -> None:
    """Render all past Q&A turns from session-state history."""
    history: list[dict[str, Any]] = st.session_state.get("_ask_history", [])

    for entry in history:
        role = entry["role"]
        content = entry["content"]

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        elif role == "assistant":
            with st.chat_message("assistant"):
                formatted = _format_ref_tags(content)
                st.markdown(formatted, unsafe_allow_html=True)

                verification = entry.get("verification")
                if verification:
                    _render_verification_badge(verification)
                    _render_claim_details(verification)

                tool_results = entry.get("tool_results")
                if tool_results:
                    _render_references(tool_results)


def _handle_question(
    question: str,
    provider: str,
    db_path: str,
    chroma_path: str,
) -> None:
    """Process a user question through the agent and update history.

    Args:
        question: The user's natural-language question.
        provider: LLM provider name.
        db_path: Path to the SQLite database.
        chroma_path: Path to the ChromaDB directory.
    """
    history: list[dict[str, Any]] = st.session_state.get("_ask_history", [])
    history.append({
        "role": "user",
        "content": question,
        "verification": None,
        "tool_calls": None,
        "tool_results": None,
        "claims": None,
    })
    st.session_state["_ask_history"] = history

    config = AgentConfig(
        provider=provider,
        db_path=db_path,
        chroma_path=chroma_path,
    )

    try:
        with st.spinner("Analyzing..."):
            cached_graph = _get_compiled_graph(
                provider=config.provider,
                model_name=config.model_name,
                temperature=config.temperature,
                db_path=db_path,
            )
            agent_history = _build_agent_history(
                max_turns=config.max_history_turns,
            )
            result: AgentResponse = run_agent(
                question,
                config,
                history=agent_history,
                compiled_graph=cached_graph,
            )

        history.append({
            "role": "assistant",
            "content": result.answer,
            "verification": result.verification,
            "tool_calls": result.tool_calls,
            "tool_results": result.tool_results,
            "claims": result.claims,
        })
        st.session_state["_ask_history"] = history
    except Exception:
        logger.exception("Agent invocation failed")
        st.error("Something went wrong. Try rephrasing your question.")
        # Remove the user question we just appended since the answer failed.
        if history and history[-1]["role"] == "user":
            history.pop()
            st.session_state["_ask_history"] = history


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def render_ask_the_data(db_path: str, chroma_path: str) -> None:
    """Render the Ask the Data chat UI section.

    Args:
        db_path: Absolute path to ``seed.db`` or ``full.db``.
        chroma_path: Absolute path to the ``.chroma/`` directory.
    """
    # --- Session state initialization ---
    if "_ask_history" not in st.session_state:
        st.session_state["_ask_history"] = []

    # --- Feature availability ---
    available, provider = _check_agent_available()

    if not available:
        st.info(
            "Ask the Data requires a local Ollama instance or an API key. "
            "See the README for setup instructions."
        )
        reconnect_col, _ = st.columns([1, 5])
        with reconnect_col:
            if st.button("Reconnect", key="_ask_reconnect"):
                st.session_state.pop("_agent_available", None)
                st.rerun()
        return

    # --- Reconnect button (when available too, for provider switching) ---
    # --- Example question chips ---
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLE_QUESTIONS):
        with cols[i % 2]:
            if st.button(q, key=f"_ask_example_{i}"):
                st.session_state["_ask_pending_question"] = q

    # --- Clear conversation ---
    if st.session_state.get("_ask_history"):
        if st.button("Clear conversation", key="_ask_clear"):
            st.session_state["_ask_history"] = []
            st.rerun()

    # --- History display ---
    _render_history()

    # --- Pending question from example chip ---
    pending = st.session_state.pop("_ask_pending_question", None)
    if pending and provider:
        _handle_question(pending, provider, db_path, chroma_path)
        st.rerun()

    # --- Chat input ---
    question = st.chat_input("Ask a question about the economic data...")
    if question and provider:
        _handle_question(question, provider, db_path, chroma_path)
        st.rerun()
