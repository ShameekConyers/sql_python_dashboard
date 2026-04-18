"""Read-only SQL tool for the economic dashboard agent.

Validates queries (no mutations, no multi-statement), executes against
a SQLite database in read-only mode, and caps results at 100 rows.
"""

from __future__ import annotations

import re
import sqlite3

from langchain_core.tools import tool

_MUTATION_PATTERN: re.Pattern[str] = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA)\b",
    re.IGNORECASE,
)

_MAX_ROWS: int = 100


def _validate_sql(sql: str) -> str | None:
    """Check *sql* for disallowed patterns.

    Returns:
        ``None`` if the query passes validation, or a human-readable
        error string describing the violation.
    """
    if not sql or not sql.strip():
        return "SQL query is empty or whitespace-only."

    match = _MUTATION_PATTERN.search(sql)
    if match:
        return (
            f"Mutation keyword {match.group().upper()!r} is not allowed. "
            "Only SELECT queries are permitted."
        )

    # Reject chained statements: semicolon followed by non-whitespace.
    stripped = sql.strip().rstrip(";")
    if ";" in stripped:
        return (
            "Multiple SQL statements are not allowed. "
            "Send one SELECT at a time."
        )

    return None


def make_sql_tool(db_path: str):
    """Create a LangGraph-compatible SQL tool bound to *db_path*.

    Args:
        db_path: Filesystem path to the SQLite database. Injected at
            graph construction time so the LLM never chooses which
            database to query.

    Returns:
        A ``@tool``-decorated function suitable for binding to a
        LangGraph agent.
    """

    @tool
    def execute_sql(sql: str) -> dict:
        """Execute a read-only SQL query against the economic dashboard database.

        Use this to answer questions about FRED economic data, recession
        predictions, Hacker News sentiment, and other tables in the
        schema. Only SELECT queries are allowed.
        """
        error = _validate_sql(sql)
        if error:
            return {"error": error, "sql": sql}

        try:
            conn = sqlite3.connect(
                f"file:{db_path}?mode=ro", uri=True
            )
            try:
                cursor = conn.cursor()
                cursor.execute("PRAGMA query_only = ON")
                cursor.execute(sql)
                columns = (
                    [desc[0] for desc in cursor.description]
                    if cursor.description
                    else []
                )
                rows = cursor.fetchmany(_MAX_ROWS + 1)
                truncated = len(rows) > _MAX_ROWS
                if truncated:
                    rows = rows[:_MAX_ROWS]
                return {
                    "sql": sql,
                    "columns": columns,
                    "rows": [list(r) for r in rows],
                    "row_count": len(rows),
                    "truncated": truncated,
                    **(
                        {
                            "note": (
                                "Results truncated to 100 rows. Add LIMIT "
                                "to your query for precise control."
                            )
                        }
                        if truncated
                        else {}
                    ),
                }
            finally:
                conn.close()
        except sqlite3.Error as exc:
            return {"error": str(exc), "sql": sql}

    return execute_sql
