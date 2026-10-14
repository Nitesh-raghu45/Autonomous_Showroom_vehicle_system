import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Must be set before `app` is imported (settings are read at import time).
os.environ.setdefault("REDIS_URL", "fakeredis://")
os.environ.setdefault("TODAY_OVERRIDE", "2026-09-18")     # benchmark dataset was authored on this date
os.environ.setdefault("RATE_LIMIT_ENABLED", "false")
os.environ.setdefault("LLM_PROVIDER", "local")
os.environ.setdefault("SEMANTIC_ROUTER", "off")      # no model download in CI; tests inject a HashingEmbedder
os.environ.setdefault("LOCAL_LLM_BASE_URL", "http://mock-llm/v1")
os.environ.setdefault("LLM_DEADLINE_MS", "1500")
os.environ["DATA_FILE"] = "data/customers.json"
