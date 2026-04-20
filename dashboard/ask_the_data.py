"""Ask the Data — conversational AI interface for the economic dashboard.

Provides a Streamlit chat UI where users type natural-language questions
and get verified, cited answers from the LangGraph ReAct agent.
"""

from __future__ import annotations

import hmac
import json
import logging
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# Ensure the project root is on sys.path so ``src.*`` imports resolve
# regardless of the working directory Streamlit uses.
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

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

_ACCESS_DURATION: timedelta = timedelta(minutes=2)
"""How long an access key stays active after entry."""

_DEFAULT_CONTACT_MSG: str = "See the project README for contact info."
"""Fallback text when no contact_email is configured in secrets."""


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


def _check_cloud_provider() -> tuple[bool, str | None]:
    """Detect whether a cloud API key is available (skip Ollama).

    Checks Anthropic then OpenAI API keys in ``st.secrets`` or
    environment variables.

    Returns:
        A ``(available, provider)`` tuple. ``provider`` is ``None``
        when no API key is found.
    """
    if "_cloud_provider_available" in st.session_state:
        return st.session_state["_cloud_provider_available"]

    api_key = _get_secret("ANTHROPIC_API_KEY")
    if api_key:
        result: tuple[bool, str | None] = (True, "anthropic")
        st.session_state["_cloud_provider_available"] = result
        return result

    api_key = _get_secret("OPENAI_API_KEY")
    if api_key:
        result = (True, "openai")
        st.session_state["_cloud_provider_available"] = result
        return result

    result = (False, None)
    st.session_state["_cloud_provider_available"] = result
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


def _check_key_gate_required(provider: str | None) -> bool:
    """Return whether the access-key gate should be shown.

    The gate is required when using a cloud API provider (anthropic or
    openai) to prevent unauthenticated API cost. Local Ollama usage is
    always ungated.

    Args:
        provider: The detected LLM provider name.

    Returns:
        ``True`` if the user must enter an access key before using the
        feature.
    """
    if provider == "ollama":
        return False
    # Gate is only meaningful if access keys are configured in secrets.
    valid_keys = _get_access_keys()
    return len(valid_keys) > 0


def _get_access_keys() -> list[str]:
    """Read the list of valid access keys from Streamlit secrets.

    Returns:
        A list of key strings, or an empty list if no keys are
        configured.
    """
    try:
        keys = st.secrets["access_keys"]["keys"]
        if isinstance(keys, (list, tuple)):
            return [str(k) for k in keys]
    except (KeyError, FileNotFoundError, AttributeError):
        pass
    return []


_KP = (0, 3, 4, 7, 8, 11, 12, 15)
_KL = 16
_KF = "%d%m%Y"


def _extract_expiry_date(key: str) -> datetime | None:
    """Extract the expiry date embedded in an access key.

    Args:
        key: An access key.

    Returns:
        The expiry ``datetime``, or ``None`` if not parseable.
    """
    if len(key) != _KL:
        return None
    digits = "".join(key[p] for p in _KP)
    if not digits.isdigit():
        return None
    try:
        return datetime.strptime(digits, _KF).replace(
            hour=23, minute=59, second=59
        )
    except ValueError:
        return None


def _is_key_expired(key: str) -> bool:
    """Check whether an access key has expired.

    Args:
        key: An access key.

    Returns:
        ``True`` if expired, ``False`` otherwise.
    """
    expiry = _extract_expiry_date(key)
    if expiry is None:
        return False
    return datetime.now() > expiry


def _get_contact_email() -> str:
    """Read the contact email from Streamlit secrets.

    Returns:
        The configured email address, or a fallback message.
    """
    try:
        email = st.secrets.get("contact_email")
        if email:
            return str(email)
    except (FileNotFoundError, AttributeError):
        pass
    return _DEFAULT_CONTACT_MSG


def _is_access_active() -> bool:
    """Check whether a valid access key session is currently active.

    Returns:
        ``True`` if the key was entered less than ``_ACCESS_DURATION``
        ago in this session.
    """
    activated_at: datetime | None = st.session_state.get(
        "_ask_key_activated_at"
    )
    if activated_at is None:
        return False
    return datetime.now() - activated_at < _ACCESS_DURATION


def _get_remaining_seconds() -> int:
    """Return the number of seconds left in the current access window.

    Returns:
        Seconds remaining, or ``0`` if expired or not activated.
    """
    activated_at: datetime | None = st.session_state.get(
        "_ask_key_activated_at"
    )
    if activated_at is None:
        return 0
    remaining = _ACCESS_DURATION - (datetime.now() - activated_at)
    return max(0, int(remaining.total_seconds()))


def _render_key_gate() -> None:
    """Render the access-key entry UI with a feature preview.

    Shows a description of what the feature does, example questions
    as static text, a contact prompt, and a password input for the
    access key.
    """
    st.markdown(
        "Ask natural-language questions about the economic data and get "
        "verified, cited answers. The AI agent queries the database, "
        "retrieves reference sources, and fact-checks every claim."
    )

    # Show example questions as static preview text.
    st.markdown("**Example questions:**")
    for q in EXAMPLE_QUESTIONS:
        st.markdown(f"- *{q}*")

    # Check for expired session.
    activated_at = st.session_state.get("_ask_key_activated_at")
    if activated_at is not None and not _is_access_active():
        st.warning("Access key expired. Enter a key to continue.")
        st.session_state.pop("_ask_key_activated_at", None)

    # Contact and key input.
    contact = _get_contact_email()
    st.info(f"Email **{contact}** for a 5-minute access key to try this feature.")

    key_input = st.text_input(
        "Access key",
        type="password",
        key="_ask_key_input",
        placeholder="Paste your access key here",
    )
    if key_input:
        valid_keys = _get_access_keys()
        matched = any(
            hmac.compare_digest(key_input, k) for k in valid_keys
        )
        if matched and _is_key_expired(key_input):
            st.error("This access key has expired. Request a new one.")
        elif matched:
            st.session_state["_ask_key_activated_at"] = datetime.now()
            st.rerun()
        else:
            st.error("Invalid access key.")


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
                "Expected": str(r.get("expected", "")) if r.get("expected") is not None else "N/A",
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


def _render_countdown(remaining_seconds: int) -> None:
    """Render a live JS countdown timer.

    Args:
        remaining_seconds: Seconds left in the access window.
    """
    import streamlit.components.v1 as components

    components.html(
        f"""
        <div id="countdown" style="
            font-size: 14px;
            color: #888;
            font-family: 'Source Sans Pro', sans-serif;
        "></div>
        <script>
            let remaining = {remaining_seconds};
            const el = document.getElementById('countdown');
            function tick() {{
                if (remaining <= 0) {{
                    el.textContent = 'Access expired. Enter a key to continue.';
                    el.style.color = '#e74c3c';
                    return;
                }}
                const m = Math.floor(remaining / 60);
                const s = remaining % 60;
                el.textContent = 'Access expires in ' + m + ':' + String(s).padStart(2, '0');
                remaining--;
                setTimeout(tick, 1000);
            }}
            tick();
        </script>
        """,
        height=30,
    )


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

                elapsed = entry.get("elapsed_seconds")
                if elapsed is not None:
                    st.caption(f"Answered in {elapsed:.1f}s")

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

    # Ensure the API key is in the environment so the LLM client can find it.
    # st.secrets doesn't set env vars automatically.
    for env_key in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY"):
        val = _get_secret(env_key)
        if val and env_key not in os.environ:
            os.environ[env_key] = val

    model_name = _get_secret("AGENT_MODEL") or AgentConfig().model_name
    config = AgentConfig(
        provider=provider,
        model_name=model_name,
        db_path=db_path,
        chroma_path=chroma_path,
    )

    try:
        spinner_msg = (
            "Querying the database and verifying claims "
            "(this can take 10-15 seconds with Ollama)..."
            if provider == "ollama"
            else "Querying the database and verifying claims..."
        )
        with st.spinner(spinner_msg):
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
            "elapsed_seconds": result.elapsed_seconds,
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


def render_ask_the_data(
    db_path: str,
    chroma_path: str,
    force_local: bool = True,
) -> None:
    """Render the Ask the Data chat UI section.

    Args:
        db_path: Absolute path to ``seed.db`` or ``full.db``.
        chroma_path: Absolute path to the ``.chroma/`` directory.
        force_local: When ``True``, use Ollama (no key gate). When
            ``False``, use the first available cloud API provider
            and require an access key.
    """
    # --- Session state initialization ---
    if "_ask_history" not in st.session_state:
        st.session_state["_ask_history"] = []

    # --- Feature availability ---
    if force_local:
        available, provider = _check_agent_available()
    else:
        # Cloud mode: skip Ollama, look for an API key.
        available, provider = _check_cloud_provider()

    if not available:
        if force_local:
            st.info(
                "Ask the Data requires a local Ollama instance. "
                "See the README for setup instructions."
            )
        else:
            st.info(
                "Cloud mode requires an ANTHROPIC_API_KEY or OPENAI_API_KEY "
                "in .streamlit/secrets.toml or environment variables."
            )
        reconnect_col, _ = st.columns([1, 5])
        with reconnect_col:
            if st.button("Reconnect", key="_ask_reconnect"):
                st.session_state.pop("_agent_available", None)
                st.session_state.pop("_cloud_provider_available", None)
                st.rerun()
        return

    # --- Access key gate (cloud providers only) ---
    if _check_key_gate_required(provider):
        if not _is_access_active():
            _render_key_gate()
            return
        # Show live countdown via JS.
        remaining = _get_remaining_seconds()
        _render_countdown(remaining)

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
