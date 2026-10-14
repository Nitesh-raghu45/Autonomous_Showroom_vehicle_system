# Verified Customer Context Architecture

## Desired behavior

1. Customer enters/pastes their customer number or mobile number.
2. `POST /session/start` resolves and verifies the customer (**once**).
3. The server creates a session bound to that customer ID.
4. The server loads the customer's complete JSON record (L1 cache + Redis) for subsequent turns.
5. The same customer record is passed to:
   - the deterministic rule router/handlers (`rule_handler` node, **no LLM**)
   - the LLM fallback (`llm_handler` node: LangChain `prompt | llm`, **one** call)
6. The **complete** conversation history is stored in the Redis session and never truncated there.
   On an LLM turn the model receives it through LangChain messages as a token-bounded view
   (`app/history.py`): recent turns verbatim, older turns shortened, hard token cap, and a note when anything was omitted.
7. The browser only needs to send `session_id` after verification; it does not need to resend the customer number.
8. If a customer number is supplied again, the backend verifies that it matches the session binding to prevent
   cross-customer data leakage.

## Runtime flow (LangGraph, `app/graph/`)

```text
Customer number
      |
      v
/session/start  ->  resolve_number() -> customer_id  ->  bind session_id -> customer_id
      |
      v
+--------------------------------------------------------------+
| Every chat turn  (service.answer -> compiled LangGraph)       |
|                                                              |
|  load_state   verified session_id -> customer_id, ONE Redis   |
|      |        pipeline: session(+full history), JSON, appts   |
|      | (error: unknown / mismatched customer -> END)          |
|      v                                                       |
|    route      deterministic router (pure function)           |
|      |                                                       |
|   +--+------------------------+                              |
|   |                           |                              |
| rule_handler              llm_handler                        |
| JSON/template/action      customer JSON + stock rows         |
| NO LLM                    + bounded history + question       |
|   |                       -> ONE LLM call -> same actions.py |
|   +--------------+------------+                              |
|                  v                                           |
|            save_history  append turn, Redis write-back (bg)   |
+--------------------------------------------------------------+
```

## Security rule

A session is customer-bound. If a later request supplies a different customer number, the request is rejected and a new
session is required.

## LLM grounding rule

The LLM must answer customer-specific questions only from the supplied JSON context and conversation history. It must not
use outside/world knowledge for customer-record facts. If the requested fact is absent, it must say it is unavailable.

## Deliberate decisions

- No LangGraph checkpointer: session state already lives in Redis and is loaded in one round trip (a checkpointer adds a Redis
  write per node and needs Redis Stack).
- No LLM-written history summary: it would be a second (background) LLM call; deterministic shortening + token cap is used instead.
- History sent to the model is bounded by default (`LLM_HISTORY_*`); set `LLM_HISTORY_RECENT_TURNS=100000
  LLM_HISTORY_MAX_TOKENS=0` to send every turn verbatim.
