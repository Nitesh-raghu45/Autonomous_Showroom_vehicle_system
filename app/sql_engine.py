"""SQL engine: in-memory SQLite + LLM text-to-SQL for analytical inventory queries.

This module handles aggregate, ranking, comparison and statistical queries about
the vehicle inventory by converting natural language to SQL, executing it against
an in-memory SQLite database seeded from the same inventory data the rest of the
app uses, and formatting the results into natural language.

Examples of queries this handles:
  - "average price by make"
  - "how many diesel cars does each brand have"
  - "which make has the most expensive car"
  - "total number of vehicles by fuel type"
  - "compare average price of new vs used cars"

The SQL model can be a dedicated text-to-SQL model (e.g. sqlcoder via Ollama)
or the same LLM used for chat, configured via SQL_LLM_PROVIDER / SQL_LLM_MODEL.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from typing import Any

from .config import settings

log = logging.getLogger("sql_engine")

_db: sqlite3.Connection | None = None
_table_schema: str = ""

# Detection patterns for queries that should be routed to SQL
# These are analytical/aggregate queries that go beyond simple filtering
SQL_PATTERNS = re.compile(
    r"\b(average|avg|mean|median)\b.*\b(price|cost|mileage|km)\b|"
    r"\b(price|cost|mileage|km)\b.*\b(average|avg|mean|median)\b|"
    r"\b(total|sum|aggregate)\b.*\b(price|cost|value|worth)\b|"
    r"\bgroup\s*(by|wise)\b|"
    r"\b(breakdown|distribution|spread|split)\b.*\b(by|across|per|of)\b|"
    r"\b(per|each|every)\s+(make|brand|manufacturer|fuel|type|transmission|body)\b|"
    r"\b(make|brand|manufacturer|fuel|type)\s*(wise|by)\b|"
    r"\bhow many\b.*\b(make|brand|does each|by)\b|"
    r"\b(count|number)\b.*\b(by|per|each|every|group)\b.*\b(make|brand|fuel|type|transmission)\b|"
    r"\b(compare|comparison)\b.*\b(average|avg|mean|price|cost|total)\b|"
    r"\bmost\s+(common|popular|frequent)\b|"
    r"\bleast\s+(common|popular|frequent)\b|"
    r"\b(min|max|minimum|maximum)\s+(price|cost|mileage)\b.*\b(by|per|each|for each)\b|"
    r"\b(percentage|percent|ratio|proportion|share)\b.*\b(of|by|per)\b|"
    r"\bstatistics\b|\bstats\b|\bsummary stats\b|\banalytics\b|"
    r"\b(rank|ranking)\b.*\b(make|brand|model|fuel)\b|"
    r"\bnew\s+vs\.?\s+used\b|\bused\s+vs\.?\s+new\b|"
    r"\bprice\s+range\b.*\b(by|per|each|for each|of)\b",
    re.IGNORECASE,
)

# Additional pattern: "how many X by Y" or "X per Y" style queries
SQL_AGGREGATE_CUE = re.compile(
    r"\bhow many\b.*\b(each|per|by|every)\b|"
    r"\b(what|which)\s+(is|are)\s+the\s+(average|total|highest|lowest|most|least)\b.*\b(by|per|for each|of each)\b|"
    r"\b(across|among)\s+(all\s+)?(makes?|brands?|fuel\s+types?|transmissions?)\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Startup: build the SQLite database from inventory
# ---------------------------------------------------------------------------

def init(items: list[dict[str, Any]]) -> bool:
    """Create an in-memory SQLite database seeded from inventory items."""
    global _db, _table_schema
    if settings.sql_routing == "off":
        log.info("SQL routing disabled")
        return False
    try:
        _db = sqlite3.connect(":memory:", check_same_thread=False)
        _db.row_factory = sqlite3.Row
        _db.execute("""
            CREATE TABLE vehicles (
                stock_id TEXT PRIMARY KEY,
                make TEXT NOT NULL,
                model TEXT NOT NULL,
                variant TEXT,
                year INTEGER,
                type TEXT,
                color TEXT,
                fuel TEXT,
                transmission TEXT,
                price INTEGER,
                mileage_km INTEGER,
                body TEXT,
                seats INTEGER
            )
        """)
        for item in items:
            _db.execute(
                "INSERT OR IGNORE INTO vehicles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    item.get("stock_id"),
                    item.get("make"),
                    item.get("model"),
                    item.get("variant"),
                    item.get("year"),
                    item.get("type"),
                    item.get("color"),
                    item.get("fuel"),
                    item.get("transmission"),
                    item.get("price"),
                    item.get("mileage_km"),
                    item.get("body"),
                    item.get("seats"),
                ),
            )
        _db.commit()
        count = _db.execute("SELECT COUNT(*) FROM vehicles").fetchone()[0]
        _table_schema = _build_schema()
        log.info("SQL engine ready: %d vehicles in SQLite", count)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("SQL engine init failed: %s", exc)
        _db = None
        return False


def _build_schema() -> str:
    """Generate a human-readable schema description for the LLM prompt."""
    if _db is None:
        return ""
    # Get sample data for context
    rows = _db.execute("SELECT DISTINCT make FROM vehicles ORDER BY make").fetchall()
    makes = [r[0] for r in rows]
    rows = _db.execute("SELECT DISTINCT fuel FROM vehicles ORDER BY fuel").fetchall()
    fuels = [r[0] for r in rows]
    rows = _db.execute("SELECT DISTINCT type FROM vehicles ORDER BY type").fetchall()
    types = [r[0] for r in rows]
    rows = _db.execute("SELECT DISTINCT transmission FROM vehicles ORDER BY transmission").fetchall()
    transmissions = [r[0] for r in rows]

    return f"""Table: vehicles
Columns:
  stock_id TEXT PRIMARY KEY  -- e.g. 'S-1001'
  make TEXT                  -- vehicle manufacturer. Values: {', '.join(makes)}
  model TEXT                 -- model name
  variant TEXT               -- variant/trim name
  year INTEGER               -- manufacture year
  type TEXT                  -- Values: {', '.join(types)}
  color TEXT                 -- exterior color
  fuel TEXT                  -- Values: {', '.join(fuels)}
  transmission TEXT          -- Values: {', '.join(transmissions)}
  price INTEGER              -- price in Indian Rupees (INR)
  mileage_km INTEGER         -- odometer reading in km (0 for new cars)
  body TEXT                  -- body style (suv, sedan, hatchback, mpv) or NULL
  seats INTEGER              -- seating capacity or NULL"""


def ready() -> bool:
    return _db is not None


def should_route_to_sql(question: str) -> bool:
    """Check if a question should be routed to the SQL engine."""
    if not ready():
        return False
    low = question.lower()
    return bool(SQL_PATTERNS.search(low) or SQL_AGGREGATE_CUE.search(low))


# ---------------------------------------------------------------------------
# Text-to-SQL prompt
# ---------------------------------------------------------------------------

SQL_SYSTEM_PROMPT = """You are a SQL expert. Given a natural language question about a vehicle inventory database, generate a single SQLite SELECT query to answer it.

{schema}

Rules:
- Output ONLY the SQL query, nothing else. No markdown, no explanation, no ```sql blocks.
- Use only SELECT statements (read-only).
- price is in Indian Rupees (INR). To show in lakhs, divide by 100000.
- For "average price", use ROUND(AVG(price)/100000.0, 2) to show in lakhs.
- Use proper GROUP BY, ORDER BY, LIMIT as needed.
- For "top N" queries, use ORDER BY ... LIMIT N.
- Always alias computed columns with readable names.
- If the question is ambiguous, pick the most reasonable interpretation.
"""


def _build_sql_prompt(question: str) -> list[dict[str, str]]:
    """Build the messages for the text-to-SQL LLM call."""
    return [
        {"role": "system", "content": SQL_SYSTEM_PROMPT.format(schema=_table_schema)},
        {"role": "user", "content": question},
    ]


def _format_lakh(value: int | float) -> str:
    """Format a price value in lakhs."""
    if isinstance(value, (int, float)):
        lakhs = value / 100000
        if lakhs >= 100:
            return f"₹{lakhs/100:.2f} Cr"
        return f"₹{lakhs:.2f} L"
    return str(value)


def _format_price_value(value: int | float, column: str) -> str:
    """Format a price value, respecting SQL columns already expressed in lakhs."""
    if re.search(r"lakh|lakhs", column, re.I):
        return f"₹{value:.2f} L"
    return _format_lakh(value)


def _format_results(rows: list[dict[str, Any]], question: str) -> str:
    """Format SQL results into a natural language response."""
    if not rows:
        return "I couldn't find any matching data for that query."

    # Single value result
    if len(rows) == 1 and len(rows[0]) == 1:
        key = list(rows[0].keys())[0]
        val = rows[0][key]
        if isinstance(val, (int, float)) and re.search(r"price|cost|amount|total|value|worth", key, re.I):
            return f"**{key.replace('_', ' ').title()}**: {_format_price_value(val, key)}"
        return f"**{key.replace('_', ' ').title()}**: {val}"

    # Tabular results
    if len(rows) > 20:
        rows = rows[:20]
        truncated = True
    else:
        truncated = False

    headers = list(rows[0].keys())
    # Detect price columns for formatting
    price_cols = {h for h in headers if re.search(r"price|cost|amount|total|value|avg|average|min|max|sum", h, re.I)}

    lines = []
    # Format as a clean table
    header_line = " | ".join(h.replace("_", " ").title() for h in headers)
    sep_line = " | ".join("---" for _ in headers)
    lines.append(header_line)
    lines.append(sep_line)
    for row in rows:
        cells = []
        for h in headers:
            val = row[h]
            if h in price_cols and isinstance(val, (int, float)) and val > 10000:
                cells.append(_format_price_value(val, h))
            elif h in price_cols and isinstance(val, (int, float)) and re.search(r"lakh|lakhs", h, re.I):
                cells.append(_format_price_value(val, h))
            else:
                cells.append(str(val) if val is not None else "-")
        lines.append(" | ".join(cells))

    result = "\n".join(lines)
    if truncated:
        result += "\n\n*(showing first 20 rows)*"
    return result


# ---------------------------------------------------------------------------
# Core: execute a SQL query from natural language
# ---------------------------------------------------------------------------

def _extract_sql(text: str) -> str | None:
    """Extract a SQL query from LLM output, handling markdown code blocks."""
    text = text.strip()
    # Try to extract from markdown code block
    m = re.search(r"```(?:sql)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL | re.IGNORECASE)
    if m:
        text = m.group(1).strip()
    # Remove any leading/trailing quotes
    text = text.strip("`'\"")
    # Must start with SELECT
    if not text.upper().startswith("SELECT"):
        # Try to find a SELECT statement in the text
        m = re.search(r"(SELECT\b.*)", text, re.IGNORECASE | re.DOTALL)
        if m:
            text = m.group(1).strip().rstrip(";")
        else:
            return None
    # Safety: reject anything that's not a pure SELECT
    forbidden = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|EXEC|ATTACH|DETACH|PRAGMA)\b", re.I)
    if forbidden.search(text):
        return None
    return text.rstrip(";")


async def answer_sql(question: str) -> str:
    """Convert a natural language question to SQL, execute it, and return a formatted answer."""
    if _db is None:
        return "SQL engine is not available right now."

    t0 = time.perf_counter()

    # Step 1: Generate SQL from the question using the LLM
    try:
        sql = await _generate_sql(question)
    except Exception as exc:  # noqa: BLE001
        log.warning("SQL generation failed: %s: %s", type(exc).__name__, exc)
        return "I can answer from your customer record and stock details. Please ask the data question in a clearer form."

    if not sql:
        return "I can answer from your customer record and stock details. Please ask the data question in a clearer form."

    log.info("text-to-SQL: %s -> %s", question[:80], sql[:200])

    # Step 2: Execute the SQL
    try:
        rows = _execute_sql(sql)
    except sqlite3.Error as exc:
        log.warning("SQL execution failed: %s (query: %s)", exc, sql[:200])
        return "I had trouble running that query, but I can still answer from your customer record and stock details. Please try again in a clearer form."

    # Step 3: Format results
    result = _format_results(rows, question)
    ms = (time.perf_counter() - t0) * 1000
    log.info("SQL query completed in %.0fms, %d rows", ms, len(rows))
    return f"📊 **Query Result**\n\n{result}"


async def _generate_sql(question: str) -> str | None:
    """Use the LLM to convert natural language to SQL."""
    from .llm import ProviderError

    provider = _get_sql_provider()
    messages = _build_sql_prompt(question)

    # Collect the full response (no streaming needed for SQL generation)
    full_text = ""
    try:
        timeout = settings.sql_timeout_ms / 1000.0 if getattr(settings, 'sql_timeout_ms', None) else 30.0
        async for ev in provider.stream(messages, max_tokens=256, timeout=timeout):
            if ev.get("type") == "token":
                full_text += ev["text"]
    except ProviderError as exc:
        log.warning("SQL LLM call failed: %s", exc)
        raise

    if not full_text.strip():
        log.warning("SQL LLM returned empty response")
        return None

    sql = _extract_sql(full_text)
    if not sql:
        log.warning("Could not extract SQL from LLM output: %s", full_text[:300])
    return sql


def _execute_sql(sql: str) -> list[dict[str, Any]]:
    """Execute a read-only SQL query with safety limits."""
    assert _db is not None
    # Add a LIMIT if not present (safety)
    if not re.search(r"\bLIMIT\b", sql, re.I):
        sql = sql + " LIMIT 50"
    cursor = _db.execute(sql)
    columns = [desc[0] for desc in cursor.description] if cursor.description else []
    rows = cursor.fetchall()
    return [{col: row[i] for i, col in enumerate(columns)} for row in rows]


# ---------------------------------------------------------------------------
# Provider for SQL LLM calls
# ---------------------------------------------------------------------------

_sql_provider = None


def _get_sql_provider():
    """Get or create the LLM provider for SQL generation.

    Uses a dedicated provider if SQL_LLM_PROVIDER is set, otherwise
    falls back to the main LLM provider.
    """
    global _sql_provider
    if _sql_provider is not None:
        return _sql_provider

    sql_provider_name = settings.sql_llm_provider or settings.llm_provider
    sql_model = settings.sql_llm_model

    # If a dedicated SQL model/provider is configured, create a separate provider
    if settings.sql_llm_provider or settings.sql_llm_model:
        from .llm.openai_compat import OpenAICompatProvider
        from .llm.ollama import OllamaProvider

        if sql_provider_name == "ollama":
            model = sql_model or settings.ollama_model
            base_url = settings.sql_llm_base_url or settings.ollama_base_url
            _sql_provider = OllamaProvider(base_url=base_url, model=model)
            log.info("SQL provider: ollama/%s at %s", model, base_url)
        elif sql_provider_name in ("groq", "openai", "local"):
            if sql_provider_name == "groq":
                base_url = settings.sql_llm_base_url or settings.groq_base_url
                api_key = settings.groq_api_key
                model = sql_model or settings.groq_model
            elif sql_provider_name == "openai":
                base_url = settings.sql_llm_base_url or settings.openai_base_url
                api_key = settings.openai_api_key
                model = sql_model or settings.openai_model
            else:  # local
                base_url = settings.sql_llm_base_url or settings.local_base_url
                api_key = settings.local_api_key
                model = sql_model or settings.local_model
            _sql_provider = OpenAICompatProvider(
                name=sql_provider_name,
                base_url=base_url,
                api_key=api_key,
                model=model,
                key_required=(sql_provider_name != "local"),
            )
            log.info("SQL provider: %s/%s at %s", sql_provider_name, model, base_url)
        else:
            # Fall back to the main provider
            from .llm import get_provider
            _sql_provider = get_provider()
            log.info("SQL provider: using main LLM provider")
    else:
        # Use the main LLM provider
        from .llm import get_provider
        _sql_provider = get_provider()
        log.info("SQL provider: using main LLM provider")

    return _sql_provider


async def close() -> None:
    """Clean up resources."""
    global _db, _sql_provider
    if _db is not None:
        _db.close()
        _db = None
    _sql_provider = None
