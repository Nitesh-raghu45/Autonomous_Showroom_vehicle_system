"""Print the compiled LangGraph workflow as a Mermaid diagram:   python -m scripts.print_graph"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("REDIS_URL", "fakeredis://")

from app.graph import get_graph  # noqa: E402

if __name__ == "__main__":
    print(get_graph().get_graph().draw_mermaid())
