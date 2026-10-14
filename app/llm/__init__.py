from __future__ import annotations

from functools import lru_cache

from ..config import settings
from .base import LLMProvider, ProviderError
from .http import close_client
from .ollama import OllamaProvider
from .openai_compat import groq, local, openai

_FACTORIES = {"local": local, "groq": groq, "openai": openai, "ollama": OllamaProvider}


@lru_cache(maxsize=1)
def get_provider() -> LLMProvider:
    name = settings.llm_provider
    if name not in _FACTORIES:
        raise ProviderError(f"LLM_PROVIDER='{name}' is not supported. Use one of: {', '.join(_FACTORIES)}")
    return _FACTORIES[name]()


__all__ = ["get_provider", "close_client", "LLMProvider", "ProviderError"]
