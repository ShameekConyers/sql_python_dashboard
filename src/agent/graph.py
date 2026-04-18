"""LangGraph ReAct agent for natural-language SQL queries.

Builds a two-node state graph (agent + tools) that loops until the LLM
produces a final answer or the recursion limit is hit.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from src.agent.config import AgentConfig
from src.agent.prompts import AGENT_SYSTEM_PROMPT, RAG_TOOL_CONTEXT, SQL_TOOL_CONTEXT
from src.agent.tools.rag_tool import make_rag_tool
from src.agent.tools.sql_tool import make_sql_tool
from src.agent.tools.verify_tool import (
    parse_agent_response,
    verify_all_claims,
)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """LangGraph state carrying the full message list."""

    messages: Annotated[list[BaseMessage], add_messages]


# ---------------------------------------------------------------------------
# Response dataclass
# ---------------------------------------------------------------------------


@dataclass
class AgentResponse:
    """Structured output returned by :func:`run_agent`.

    Attributes:
        answer: Prose answer (extracted narrative) or off-topic refusal.
        tool_calls: Log of tool invocations (SQL queries and RAG
            retrievals) the agent executed during the ReAct loop.
        claims: Parsed structured claims from the agent's JSON output.
            Each claim is a dict with keys like ``statement``,
            ``metric_type``, ``series_id``, ``expected_value``, and
            ``date_range``.
        verification: Aggregate verification result dict with keys
            ``status``, ``all_verified``, ``results``, ``total``, and
            ``passed_count``. Each entry in ``results`` contains the
            claim statement, pass/fail, actual value, and reason.
    """

    answer: str
    tool_calls: list[dict] = field(default_factory=list)
    claims: list = field(default_factory=list)
    verification: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------


def _build_llm(config: AgentConfig) -> BaseChatModel:
    """Instantiate an LLM backend from *config*.

    Args:
        config: Agent configuration with provider and model details.

    Returns:
        A LangChain chat model ready for tool binding.

    Raises:
        ValueError: If the provider is not supported.
    """
    if config.provider == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(
            model=config.model_name, temperature=config.temperature
        )
    if config.provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(
            model=config.model_name, temperature=config.temperature
        )
    if config.provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=config.model_name, temperature=config.temperature
        )
    raise ValueError(f"Unsupported provider: {config.provider!r}")


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------


def _should_continue(state: AgentState) -> str:
    """Route after the agent node: tool calls go to tools, else END.

    Args:
        state: Current graph state with the message list.

    Returns:
        ``"tools"`` if the last message has tool calls, ``"end"``
        otherwise.
    """
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "end"


def build_graph(
    config: AgentConfig,
) -> StateGraph:
    """Construct and compile the ReAct agent graph.

    Args:
        config: Agent configuration (provider, model, db_path, etc.).

    Returns:
        A compiled LangGraph ``StateGraph`` ready for invocation.
    """
    sql_tool = make_sql_tool(config.db_path)
    rag_tool = make_rag_tool()
    tools = [sql_tool, rag_tool]
    llm = _build_llm(config)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: AgentState) -> dict:
        """Invoke the LLM with the current message history.

        Args:
            state: Current graph state.

        Returns:
            Dict with the new messages list containing the LLM response.
        """
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.set_entry_point("agent")
    graph.add_conditional_edges(
        "agent",
        _should_continue,
        {"tools": "tools", "end": END},
    )
    graph.add_edge("tools", "agent")
    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_agent(
    question: str,
    config: AgentConfig,
    history: list[dict] | None = None,
) -> AgentResponse:
    """Run the ReAct agent on a user question.

    Args:
        question: Natural-language question about the economic data.
        config: Agent configuration.
        history: Optional prior conversation turns. Each dict should
            have ``"role"`` (``"user"`` or ``"assistant"``) and
            ``"content"`` keys. Only the last
            ``config.max_history_turns`` turns are used.

    Returns:
        An :class:`AgentResponse` with the prose answer and a log of
        any SQL tool calls made during the ReAct loop.
    """
    system_content = (
        AGENT_SYSTEM_PROMPT
        + "\n\n"
        + SQL_TOOL_CONTEXT
        + "\n\n"
        + RAG_TOOL_CONTEXT
    )
    messages: list[BaseMessage] = [SystemMessage(content=system_content)]

    if history:
        recent = history[-config.max_history_turns :]
        for turn in recent:
            role = turn.get("role", "")
            content = turn.get("content", "")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))

    messages.append(HumanMessage(content=question))

    compiled = build_graph(config)

    try:
        result = compiled.invoke(
            {"messages": messages},
            config={"recursion_limit": config.recursion_limit},
        )
    except Exception as exc:
        # Catch GraphRecursionError or any other runtime failure.
        if "recursion" in str(exc).lower():
            return AgentResponse(
                answer=(
                    "I couldn't answer within the allowed steps. "
                    "Try rephrasing your question or asking something "
                    "more specific."
                ),
            )
        raise

    # Extract final answer and tool call log.
    all_messages = result["messages"]
    answer = ""
    tool_calls_log: list[dict] = []

    for msg in all_messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_log.append(
                    {"name": tc["name"], "args": tc["args"]}
                )

    # The final message should be the agent's prose answer.
    final_msg = all_messages[-1]
    if isinstance(final_msg, AIMessage):
        answer = final_msg.content or ""

    # Phase 17: parse structured claims and verify against DB.
    narrative, claims = parse_agent_response(answer)
    verification = verify_all_claims(claims, config.db_path)

    return AgentResponse(
        answer=narrative,
        tool_calls=tool_calls_log,
        claims=[dataclasses.asdict(c) for c in claims],
        verification={
            "status": verification.status,
            "all_verified": verification.all_verified,
            "results": [
                {
                    "statement": r.claim.statement,
                    "passed": r.passed,
                    "actual_value": r.actual_value,
                    "reason": r.reason,
                }
                for r in verification.results
            ],
            "total": verification.total,
            "passed_count": verification.passed_count,
        },
    )
