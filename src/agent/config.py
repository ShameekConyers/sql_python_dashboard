"""Agent configuration with environment-variable loading and provider validation."""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class AgentConfig:
    """Configuration for the LangGraph ReAct agent.

    Attributes:
        provider: LLM backend — ``"ollama"``, ``"anthropic"``, or ``"openai"``.
        model_name: Model identifier passed to the provider
            (e.g. ``"llama3.1:8b"``, ``"claude-sonnet-4-20250514"``).
        temperature: Sampling temperature. Low values produce more
            deterministic, factual answers.
        recursion_limit: Maximum LangGraph graph steps. Each tool call
            costs 2 steps (agent to tool + tool to agent). A limit of 6
            allows up to 2 full SQL round-trips plus a final synthesis.
        db_path: Path to the SQLite database (``seed.db`` or ``full.db``).
        max_history_turns: Sliding-window size for conversation history.
    """

    provider: str = field(
        default_factory=lambda: os.getenv("AGENT_PROVIDER", "ollama")
    )
    model_name: str = field(
        default_factory=lambda: os.getenv("AGENT_MODEL", "llama3.1:8b")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("AGENT_TEMPERATURE", "0.1"))
    )
    recursion_limit: int = field(
        default_factory=lambda: int(os.getenv("AGENT_RECURSION_LIMIT", "6"))
    )
    db_path: str = field(
        default_factory=lambda: os.getenv("AGENT_DB_PATH", "data/seed.db")
    )
    max_history_turns: int = field(
        default_factory=lambda: int(os.getenv("AGENT_MAX_HISTORY", "5"))
    )

    def validate(self) -> None:
        """Check that the selected provider's package is importable.

        Raises:
            ValueError: If ``provider`` is not one of the supported values.
            ImportError: If the required package for the chosen provider
                is not installed.
        """
        provider_packages: dict[str, tuple[str, str]] = {
            "ollama": ("langchain_ollama", "langchain-ollama"),
            "anthropic": ("langchain_anthropic", "langchain-anthropic"),
            "openai": ("langchain_openai", "langchain-openai"),
        }
        if self.provider not in provider_packages:
            raise ValueError(
                f"Unknown provider {self.provider!r}. "
                f"Choose from: {', '.join(sorted(provider_packages))}"
            )
        module_name, pip_name = provider_packages[self.provider]
        try:
            __import__(module_name)
        except ImportError as exc:
            raise ImportError(
                f"Provider {self.provider!r} requires the {pip_name} package. "
                f"Install it with: pip install {pip_name}"
            ) from exc
