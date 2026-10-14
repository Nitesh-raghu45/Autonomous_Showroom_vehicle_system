"""LangChain chat-model adapter over the existing providers (``app/llm``).

Why an adapter instead of ``langchain-openai``
----------------------------------------------
The providers already do the latency-critical work: one shared keep-alive HTTP client,
JSON-schema constrained decoding (with automatic downgrade), tiny retries, usage + TTFT
accounting. Wrapping them keeps all of that, while the rest of the LLM path gets a normal
LangChain surface:

    chain = CHAT_PROMPT | ProviderChatModel(provider=..., json_schema=PLAN_SCHEMA)
    async for chunk in chain.astream(inputs): ...

Streaming contract: text chunks are ``AIMessageChunk(content=piece)``; one final chunk with empty
content carries ``usage_metadata`` and ``response_metadata["ttft_ms"]``.
"""
from __future__ import annotations

from typing import Any, AsyncIterator, Sequence

from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

_ROLE = {"system": "system", "human": "user", "ai": "assistant"}


def to_provider_messages(messages: Sequence[BaseMessage]) -> list[dict[str, str]]:
    """LangChain messages -> the ``[{"role", "content"}]`` dicts the providers take."""
    return [{"role": _ROLE.get(m.type, "user"), "content": str(m.content)} for m in messages]


class ProviderChatModel(BaseChatModel):
    """Async-only chat model that streams from one of our ``LLMProvider`` objects."""

    provider: Any
    json_schema: dict[str, Any] | None = None
    max_tokens: int | None = None

    @property
    def _llm_type(self) -> str:
        return f"teknnova-{getattr(self.provider, 'name', 'provider')}"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"provider": getattr(self.provider, "name", None), "model": getattr(self.provider, "model", None)}

    def _generate(  # pragma: no cover - the request path is async only
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError("ProviderChatModel is async-only: use ainvoke() / astream().")

    async def _astream(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        stream = self.provider.stream(
            to_provider_messages(messages), schema=self.json_schema, max_tokens=self.max_tokens,
        )
        async for ev in stream:
            if ev["type"] == "token":
                yield ChatGenerationChunk(message=AIMessageChunk(content=ev["text"]))
            elif ev["type"] == "usage":
                u = ev.get("usage") or {}
                p, c = int(u.get("prompt_tokens") or 0), int(u.get("completion_tokens") or 0)
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="",
                    usage_metadata={
                        "input_tokens": p, "output_tokens": c, "total_tokens": p + c,
                        "input_token_details": {"cache_read": int(u.get("cached_prompt_tokens") or 0)},
                    },
                    response_metadata={"ttft_ms": int(ev.get("ttft_ms") or 0)},
                ))

    async def _agenerate(
        self, messages: list[BaseMessage], stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None, **kwargs: Any,
    ) -> ChatResult:
        text: list[str] = []
        usage: dict[str, Any] | None = None
        meta: dict[str, Any] = {}
        async for gen in self._astream(messages, stop, run_manager, **kwargs):
            msg = gen.message
            text.append(str(msg.content))
            usage = getattr(msg, "usage_metadata", None) or usage
            meta.update(msg.response_metadata or {})
        return ChatResult(generations=[ChatGeneration(message=AIMessage(
            content="".join(text), usage_metadata=usage, response_metadata=meta,
        ))])
