# Customer Assist - v3 (latency-first architecture)

Vehicle-dealership CRM assistant, re-architected for a **P95 < 500 ms** target.
The old path put an LLM behind every question; the new path answers factual questions with
rules + templates in single-digit milliseconds and calls a small, constrained-JSON LLM only when
the request is genuinely ambiguous.

```
Browser -> (nginx) -> FastAPI  --  POST /session/start verifies the customer ONCE and binds session_id -> customer_id
                        |
                 LangGraph workflow (app/graph/)          5 nodes, one compiled graph shared by all requests
                        |
   load_state ----------+--- ONE Redis pipeline: session (COMPLETE history) + customer JSON (L1 cache) + appointments
        |
      route -------------- deterministic intent router (pure function, ~µs): rules, vocab, date/time + slot extraction
      /    \
 rule_handler  llm_handler
 actions.py +   LangChain `prompt | llm`: verified customer JSON + stock rows + bounded history + question
 render.py      -> ONE JSON-schema-constrained call (deadline-capped, deterministic fallback)
 (NO model)         -> same actions.py
      \    /
   save_history ------------ append the turn to the full history; Redis write-back is fire-and-forget
                        |
                  SSE: meta -> token* -> [replace] -> [action] -> done
```

> **v3.1:** LangGraph + LangChain are back as the orchestration/history layer (see
> [LangGraph + LangChain layer](#langgraph--langchain-layer)). Behaviour is unchanged: with the mock LLM, the 200-query benchmark
> produces identical event streams through the old and new `service.answer()` (199/200 byte-identical after masking random
> IDs/timestamps; the 200th differs only in prompt-token count because of history compaction). Real-model behaviour was not run.

## What changed, and where

| Problem found in v2 | Fix | Files |
|---|---|---|
| Full inventory read from Redis **twice per turn** and pasted into every prompt | Indexed in-memory catalogue (make/model/type/fuel/body sets); only retrieved rows ever reach a prompt | `app/inventory.py`, `app/prompting.py::stock_block` |
| Runtime **summarisation = 2nd LLM call** | Removed. Compact session state (last N turns + pending action), no summariser | `app/cache.py` (session state), `app/service.py` |
| LangGraph on a linear 4-step path (v2) | v3 removed it; **v3.1 re-introduced it as a deliberately tiny 5-node graph** where the rule path has no LLM node (cost measured below) | `app/graph/` |
| Giant prompt for "what is my email?" | Router answers it with **no model**; LLM prompt is rules + this customer's record + a few retrieved rows | `app/router/`, `app/fast_path.py`, `app/render.py` |
| `MAX_TOKENS=800` | `LLM_MAX_TOKENS=192`, optional slots in schema so the model only emits what it needs | `app/config.py`, `app/prompting.py` |
| `<!-- ACTION:{...} -->` regex-parsed from prose | JSON-schema constrained output (`intent`, `slots`, `reply`), validated by pydantic | `app/planner.py`, `app/actions.py` |
| 6 sequential Redis commands to register | One atomic Lua script | `app/cache.py::_REGISTER_LUA` |
| Read-modify-write races on update | Optimistic CAS script + retry | `app/cache.py::update_customer_record` |
| N+1 GETs for appointments | One HASH per customer (`HGETALL`), legacy keys auto-migrated | `app/cache.py` |
| Customer/session re-resolved every message; several Redis round trips | L1 cache + single pipelined round trip; cross-worker invalidation via pub/sub | `app/cache.py`, `app/l1.py` |
| Redis HINCRBY metrics written **on every request** | In-process histograms, Prometheus text, `Server-Timing` header | `app/observability.py` |
| Ollama built a new HTTP client per request; Ollama as the production server | One shared keep-alive client; `local` provider for **vLLM / SGLang** (OpenAI-compatible) | `app/llm/` |
| Dead Redis could add its timeout to every request | 5 s circuit breaker, JSON-seed fallback for reads | `app/cache.py` |
| Cold TCP + cold prefix cache on first request | Warm-up call at startup | `app/service.py::warm_llm` |
| Hung LLM = hung user | Hard `LLM_DEADLINE_MS`, deterministic fallback message | `app/service.py` |
| No source of truth beyond Redis | Optional Postgres write-behind (**experimental**, off by default) | `app/persistence.py` |
| Chat could not cancel/register at all | Two-step cancel (propose -> "yes"), chat registration, guest mode | `app/actions.py`, `app/fast_path.py` |
| Duplicate-mobile error leaked another customer's ID | Generic message | `app/mutations.py` |
| Session could be re-pointed at another customer | Session is bound; mismatch rejected | `app/service.py` |
| "Today" = server UTC date | Dealership timezone (`APP_TIMEZONE`), `TODAY_OVERRIDE` for tests | `app/clock.py` |

Removed (in v3): the old v2 `app/graph/` (since replaced by the v3.1 graph), `tiktoken`, the old `tests/benchmark_deep_latency.py`
(it monkey-patched graph nodes; replaced by `tests/benchmark_latency.py`), `scripts/verify_offline.py`, and
**`.env` (it contained live Groq and Redis credentials - rotate them; use `.env.example`)**.
`frontend/node_modules` was dropped from the archive; run `npm install` in `frontend/`.

## LangGraph + LangChain layer

```
START -> load_state --(error)--> END
              |
            route
           /     \
   rule_handler  llm_handler
           \     /
         save_history -> END          (print it: python -m scripts.print_graph)
```

| Node | Job | LLM? |
|---|---|---|
| `load_state` | verify session/customer (once), ONE pipelined Redis round trip: session + complete history, customer JSON, appointments | no |
| `route` | deterministic router; emits the `meta` event | no |
| `rule_handler` | data lookup / validated action / template | **no** |
| `llm_handler` | LangChain `CHAT_PROMPT \| ProviderChatModel` -> exactly one constrained-JSON call, then the same `actions.py` | yes |
| `save_history` | side effects, append turn to the history, fire-and-forget Redis write-back, `done` event | no |

* **State** (`app/graph/state.py`): the verified customer record, appointments and session travel through the graph state, so no
  node re-reads Redis or the JSON store.
* **LangChain**: `app/prompting.py::CHAT_PROMPT` (a `ChatPromptTemplate` with a `MessagesPlaceholder` for history),
  `app/history.py` (messages + token-bounded view), `app/llm/langchain_model.py` (`ProviderChatModel`, a `BaseChatModel`
  adapter over the existing providers - so constrained decoding, the shared keep-alive client, retries, usage/TTFT and the
  deadline all keep working). Customer text is passed as template *variables*, never parsed as template syntax (tested).
* **History: lossless storage, bounded prompt.** Redis keeps every turn. What the model sees (`LLM_HISTORY_*`): the last 6 turn
  pairs verbatim, older messages shortened to 160 characters, hard cap 2000 tokens (`trim_messages`, oldest dropped first, and the
  model is told when something was omitted). Conversations of up to 6 exchanges are sent exactly as before. To send everything
  verbatim set `LLM_HISTORY_RECENT_TURNS=100000 LLM_HISTORY_MAX_TOKENS=0`. Rule-answered turns are also in the history, so the LLM
  can refer to them. Token counts are an estimate (~4 chars/token), not a tokenizer.
* **Python:** developed and tested on 3.12 (the Dockerfile's version). The nodes use `get_stream_writer()`, which LangGraph documents as
  not working inside async nodes on Python < 3.11 - not tested on older versions here.
* **LangSmith** tracing works through the standard `LANGSMITH_*` env vars (see `.env.example`). Not tested here; traces contain
  customer data.

**What it costs (measured, in-process, fakeredis + instant mock LLM, same machine, 2 runs each):**

| Path | before (v3) p50 / p95 | now (v3.1) p50 / p95 |
|---|---|---|
| rule path | ~1.9 / ~3.3 ms | ~5.3 / ~8.0 ms |
| LLM path, excluding the model's own time | ~4.1 / ~6.1 ms | ~11.4 / ~18.4 ms |

So the graph + LangChain add roughly **3 ms on the rule path and ~7 ms on the LLM path**. Both are small next to the 500 ms
budget and next to real model latency, but they are not zero. The overhead is LangGraph's own scheduling, spread across nodes
(profiled: no large copies of the customer record).

**Deliberately not done**
* **No LangGraph checkpointer.** The session (customer binding, complete history, pending action) already lives in Redis and is
  loaded by `load_state` in the single pipelined round trip. A Redis checkpointer would add a write per node and needs Redis Stack
  (RediSearch/RedisJSON), which the fakeredis dev mode and the stock `redis:7-alpine` image don't provide.
* **No LLM-written summary of old turns.** It would be a second (background) LLM call; the deterministic shortening above covers
  the "minimum tokens" goal without one. It can be added as a background task in `save_history` if you find shortened old turns
  lose too much.
* **Not verified here:** a live Redis, a real vLLM/Groq/Ollama call, Docker (the Dockerfile `COPY` was changed to tolerate the
  missing `requirements-semantic.txt`, but no image was built).

## Run it

```bash
cp .env.example .env                     # edit provider / redis
pip install -r requirements.txt
uvicorn app.main:app --port 8000         # add --workers 4 in production

# zero-dependency dev (no Redis, no GPU):
pip install -r requirements-dev.txt
REDIS_URL=fakeredis:// uvicorn scripts.mock_llm_server:app --port 8001 &
REDIS_URL=fakeredis:// LOCAL_LLM_BASE_URL=http://127.0.0.1:8001/v1 uvicorn app.main:app --port 8000
```

Production model server (NVIDIA L4): `deploy/serve_vllm.sh` or `deploy/serve_sglang.sh`
(or `docker compose --profile vllm up --build`). Model: `Qwen/Qwen3-4B-Instruct-2507-FP8`.

Frontend: `cd frontend && npm install && npm run dev` (dev) or `npm run build` (served by FastAPI from `frontend/dist`).

### Troubleshooting model fallbacks

The chat fallback is intentionally generic: it can mean a provider/network error, an empty or malformed model response, or the configured generation deadline expiring. Check the backend logs for `planner.provider_error` / `LLM outcome` and distinguish `source=llm_error` from `source=llm_timeout`; the provider error text identifies HTTP errors and connection failures. Never include or share API keys when collecting logs.

The OpenAI-compatible providers retry transient connection failures and HTTP 408, 425, 429, and 5xx responses up to `LLM_MAX_RETRIES` (default `1`), with bounded backoff and support for `Retry-After`. Verify `LLM_PROVIDER`, the matching API key, base URL, and model name if errors persist. Increase `LLM_DEADLINE_MS` only when logs show timeouts; it does not fix authentication, rate-limit, or malformed-response errors.

## Semantic router (MiniLM) - stage 2

```
message -> rules/regex (µs) --resolved--> deterministic handler
                |
                '-- not resolved --> MiniLM embedding vs intent exemplars (data/semantic_intents.json)
                                        |-- score >= HIGH and margin >= MARGIN (and anchor guard passes) --> same handler, no LLM
                                        |-- score >= LOW                                                   --> LLM, told the likely intent
                                        '-- else                                                           --> LLM
```
* The embedding only picks the **intent**. Dates, e-mails, prices, makes, etc. are still extracted by the
  deterministic extractors and every write still goes through `actions.py`, so a semantic hit cannot do
  anything the rules could not.
* Guards against MiniLM's known weakness (near-identical vectors for "email" vs "phone"): a required margin
  over the runner-up, an **anchor guard** (query says "phone" but the winner is GET_EMAIL -> demoted to a hint),
  and a higher bar for book / cancel / update / register. Multi-intent messages never use this stage.
* Loads in the background after start-up (the app serves immediately; `/health` -> `semantic.ready`).
  Missing package / model / network -> warning, rules + LLM continue. A crash inside the stage is caught per request.
* Install: `pip install -r requirements-semantic.txt` (ONNX Runtime, CPU, no torch). Docker:
  `docker build --build-arg WITH_SEMANTIC=1 .` bakes the model in. Each uvicorn worker holds its own copy
  (~100-250 MB), and it runs on CPU so it never competes with vLLM for the GPU.
* **Calibrate before trusting it:** `python -m scripts.calibrate_semantic --show-errors` prints coverage vs precision
  for a grid of thresholds on `data/semantic_eval.json` (92 labelled paraphrases incl. Hinglish/typos and
  must-not-execute cases; add your production queries). The 0.80 / 0.62 / 0.08 defaults are placeholders.
* **Not verified here:** the sandbox cannot download MiniLM (huggingface.co is blocked), so the real model was never
  run. Tests use a lexical `HashingEmbedder` and pinned scores to verify the logic, not MiniLM's accuracy or latency.
  Expect a few ms per uncached query on CPU (unmeasured); repeat queries hit an in-process cache.

## Tests and benchmark

```bash
pytest -q                                   # 86 tests: flows, graph/history/LangChain layer, router coverage over all 200 queries, accuracy
python -m scripts.router_report             # how each benchmark category is routed + every LLM-bound query
python -m tests.benchmark_latency --base-url http://localhost:8000 --concurrency 1,5,10,25,50 --repeat 3
```
Run the server with `RATE_LIMIT_ENABLED=false` for load tests. The runner reports P50/P90/P95/P99/max,
% under 500 ms, TTFT, per-route breakdown, tokens, error rate and rule-scored accuracy per concurrency level.
Scoring only covers rows with an objective ground truth (~100 of 200); the rest are reported as unscored.

### Dataset rows I believe are wrong (excluded from scoring, see `tests/scoring.py`)
* BQ-109 expects a primary-mobile change to be rejected; it is now allowed (valid + globally unique number). No OTP step exists - add one before production, since the mobile number is also the sign-in credential.
* BQ-073 expects 2026-09-22 for "next Monday" authored on Fri 09-18; that Monday is 09-21.
* BQ-139 expects 48 for "how many makes" (48 is the vehicle count; there are 9 makes).
* BQ-150 expects 6 Maruti/Hyundai hatchbacks <= 10 lakh; only 4 such vehicles exist at all (3 hatchbacks).
* Quotation rows store the *grand total* as ground truth regardless of what the question asks.

## What was actually measured (and what was not)

Measured here, in a sandbox, with fakeredis (zero network RTT) and an instant mock LLM, client and server
on the same CPU: router-path P50 ~2 ms / P99 ~7 ms at concurrency 1, 100 % under 500 ms; at concurrency 50
on one uvicorn worker the shared-CPU sandbox queues (P95 ~580 ms) - that is a sandbox limit, not a server
result; run with several workers on real hardware. **This measures framework overhead only.**

* **Router coverage:** 189/200 benchmark queries (94.5 %) are answered without a model; 100/100 objectively
  scored deterministic answers are correct. **Caveat - this is optimistic:** the rules were developed while
  looking at this dataset. On 40 held-out phrasings I wrote afterwards, ~58 % were routed deterministically and
  ~42 % fell through to the LLM (safe, but slower); 3 deterministic answers were wrong (fixed, and the
  rules generalised a bit further). Expect real traffic to send far more than 5 % to the LLM until you add
  production phrasings to the dataset and extend the rules - `python -m scripts.router_report` shows the gaps.
* **NOT measured:** anything on your L4 - real LLM TTFT/throughput, real Redis RTT, Postgres, Docker/compose
  files, the Groq/OpenAI/Ollama providers (same code path as the tested mock, but no live call was made).
* **LLM leg:** decode speed sets the floor. 4B FP8 on an L4 will not reliably finish a ~40-token JSON plan
  inside 500 ms for every request; that is why `LLM_DEADLINE_MS` + fallback exist. On the benchmark only ~5 % of traffic reaches it; on
  unseen phrasing it may be several times that, so P95 will be decided by this leg. If that share matters, point `LLM_PROVIDER=groq` at the LLM leg (same code) and compare
  with `tests/benchmark_latency.py`. So "everything < 500 ms" is achievable for the deterministic path
  by construction; for the LLM path it is something to measure, not assume.
* Answers in the LLM path are grounded by prompt + constrained schema, not by a post-hoc fact checker.

## Adversarial audit (what was found and fixed)
Probed ~150 hand-written queries (typos, Hinglish, terse fragments, injections, other-people lookups,
booking edge cases, races, hostile model output, Redis down). Fixed:

| Finding | Fix |
|---|---|
| Same appointment could be booked repeatedly (also under parallel requests) | Pre-check + atomic Redis slot guard (`claim_slot`); REST returns 409 |
| "what is kavita's email" / "who owns 9559407816" / "my friend X wants his status" answered with **your own** data | Possessive, third-party and number-lookup detection -> refusal |
| An email like `customer10006@example.com` was refused as a "customer reference" | Emails stripped before the cross-tenant check |
| Bad model output (`"max_price": "cheap"`) raised a TypeError -> 500 | Slot coercion + guarded execution; deterministic handlers guarded too |
| Terse / Hinglish questions (`status`, `my quotation`, `mera email kya hai`) went to the LLM | Fragment + Hinglish rules |
| "ok", "what can you do", "speak to a human" answered oddly or by the LLM | Dedicated replies (no invented contact details) |
| Unknown brands (Ferrari, Tesla, Mercedes) went to the LLM | "We don't stock X. We carry ..." |
| "reduce the price" returned a stock list | Treated as not-on-record (no discounting in chat) |
| Appointment list grew unbounded and mixed cancelled with upcoming | Upcoming first, capped, cancelled collapsed |
| Names accepted any characters (prompt-injection / XSS vector) | Letters, spaces, `.` `'` `-` only; prompt marks record text as data |
| Behind nginx every user shared one rate-limit bucket | `--proxy-headers --forwarded-allow-ips` in the Dockerfile (only safe behind your own proxy) |

Known gaps (not fixed): Devanagari/other-script queries go to the LLM; no showroom opening days/holidays are
modelled (Sundays can be booked); no OTP on mobile-number changes; session state (last turns, incl. echoed
email/ID) sits in Redis for `SESSION_TTL_SECONDS`; prompt-injection resistance of the LLM leg relies on the
prompt + schema, not on a verifier.

## Behaviour notes
* Booking hours 08:00-20:00, no past dates; bare hours 1-7 are read as PM ("at 4" -> 16:00) and the reply says so.
* Cancel asks for confirmation (state kept in the session for 3 turns). Dealership-arranged contacts
  (`scheduled_contact`) are read-only in chat.
* Inventory body-type/seat filters use `data/vehicle_attributes.json` (approximate model-level data; edit it).

## Inventory domain (structured query engine)

Inventory is structured data, so it is **not** RAG. Every inventory question follows one path and
needs no model for anything the rules can parse:

```text
question -> router (rules) -> InventoryQuery -> in-memory indexed Catalog -> InventoryResult -> template
```

| Module | Role |
| --- | --- |
| `app/inventory.py` | `Catalog`: in-RAM indexes (make, model, type, fuel, transmission, body, **year, variant**) + predicates (price, km, seats, **year, colour**) |
| `app/inventory_query.py` | `InventoryQuery` / `InventoryResult` and the pure `execute()`; also `slots_to_query()` which validates the LLM planner's slots |
| `app/inventory_renderer.py` | deterministic answer text (plain text - the UI is `pre-wrap`, not markdown) |
| `app/router/extract.py` | filter extraction: price, km, fuel, body, seats (`5 seater` = exactly 5, `7 seater` = 7+), **variant, year, colour, `5 cheapest`** |
| `app/router/rules.py` | `aggregate_route`, `compare_route`, `detail_route` + the distinct-field modes |

Operations: `list`, `count`, superlatives (`sort` + `limit`), `distinct`, `range`, `average`,
`group_by` (count / avg / min / max per make, fuel, body, transmission, year, new-vs-used, colour; optional
`pick` for "which brand has the highest ...") , `compare`, `detail`.

| Question | Handled by |
| --- | --- |
| "Do you have a 2024 Swift ZXi AMT?" | list (model + variant + year) |
| "5 cheapest cars", "newest Swift" | superlative |
| "What variants of Creta do you have?", "What fuel options for BMW?" | distinct |
| "Average price of Tata cars", "price range of BMW" | average / range |
| "How many cars from each brand?", "Which brand has the highest average price?" | group_by |
| "Compare Swift and Baleno" | compare (side-by-side from stock) |
| "Tell me about S-1005" | detail |
| "Which ones are automatic?" / "Which one is cheapest among these?" | refinement of the previous filters / shown `inv_result_ids` |

Boundaries: verdict questions ("which is better / cheaper?", "should I buy ...") and compare requests with
new/used qualifiers still go to the LLM, as does advice-style wording. When the LLM planner runs a
`SEARCH_INVENTORY` it only emits slots (`body`, `transmission`, `min_seats`, `year`, `color`, `variant`, `sort`,
`limit`, `operation`, `field`, `metric` ...); `slots_to_query()` validates/clamps them and the same engine computes
the answer. The text-to-SQL engine (`app/sql_engine.py`) remains the fallback for analytics the rules do not express
(median, percentages, totals ...). Whole-catalogue analytics and stock-id lookups carry `sticky=False`, so they do not
overwrite the browsing context used by follow-ups such as "only automatic ones".

* Multi-part requests go to the LLM, which is told to do one action per turn.
* Old `MAX_TOKENS` env var is ignored; use `LLM_MAX_TOKENS`.
