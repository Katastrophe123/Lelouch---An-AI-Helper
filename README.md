# AI Homework Helper — v2

[![CI](https://github.com/kanishksingh-arch/ai-homework-helper/actions/workflows/ci.yml/badge.svg)](https://github.com/kanishksingh-arch/ai-homework-helper/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/lint-ruff-261230.svg)](https://github.com/astral-sh/ruff)
[![Checked with mypy](https://img.shields.io/badge/types-mypy-blue.svg)](https://mypy-lang.org/)

A tutoring agent that **solves your homework question, checks its own answer,
and only then explains it** — behind an async **FastAPI** backend with
**PostgreSQL** persistence, with **Streamlit** reduced to a pure API client.

---

## Problem statement

v1 was a single Streamlit script: a LangChain `ChatHuggingFace` wrapper around
`Meta-Llama-3-8B-Instruct`, with `SystemMessage` / `HumanMessage` /
`AIMessage` history kept in `st.session_state`. It worked as a demo and failed
as a product, in four specific ways:

| v1 problem | Why it mattered | v2 answer |
|---|---|---|
| History lived in `st.session_state` | A browser refresh wiped the conversation; nothing was queryable afterwards | `users` / `conversations` / `messages` in PostgreSQL, retrievable via the API |
| The UI *was* the application | No other client could use it, and the model credentials sat in the same process as the browser-facing code | FastAPI backend; Streamlit holds no LLM client, no prompts, no DB driver |
| Whatever the model said went to the student | A confidently wrong answer was indistinguishable from a correct one | A `self_check` node must pass before the answer is explained |
| One transient HF 503 produced a stack trace | Any provider hiccup lost the turn | Retries with backoff, then an explicitly *unverified* partial answer |

The v1 script is kept at [`legacy/v1_streamlit_app.py`](legacy/v1_streamlit_app.py)
so this comparison points at real code. It is excluded from lint, types, tests
and both Docker images.

The central design claim of v2: **an unverified answer must be labelled, not
hidden.** When the self-check never passes, the student gets a response that
opens with *"I could not fully verify this answer, so please check it yourself"*
— not a rejected answer dressed up as a checked one.

---

## Architecture

```mermaid
flowchart TB
    subgraph ui["Streamlit frontend (separate container)"]
        UI["streamlit_app.py<br/>(rendering only)"]
        AC["api_client.py<br/>(httpx + httpx-sse)"]
        UI --> AC
    end

    subgraph api["FastAPI backend"]
        MW["RequestContextMiddleware<br/>(request_id, structured logs)"]
        AUTH["JWT bearer auth<br/>(all routes except /health, /auth/token)"]
        R1["POST /conversations"]
        R2["GET /conversations"]
        R3["GET /conversations/:id"]
        R4["POST /conversations/:id/messages"]
        R5["POST /conversations/:id/messages/stream<br/>(SSE)"]
        TS["TutorService<br/>(graph + persistence)"]
        CS["ConversationService<br/>(read side)"]
    end

    subgraph agent["LangGraph state graph"]
        N1["understand"]
        N2["solve"]
        N3["self_check"]
        N4["explain"]
        N5["fallback"]
    end

    LLM["LLMClient<br/>huggingface | openai | stub"]
    DB[("PostgreSQL<br/>users · conversations · messages")]

    AC -->|HTTP / SSE| MW --> AUTH
    AUTH --> R1 --> CS
    AUTH --> R2 --> CS
    AUTH --> R3 --> CS
    AUTH --> R4 --> TS
    AUTH --> R5 --> TS

    TS -->|ainvoke| N1
    TS -->|messages + trace| DB
    CS -->|query| DB

    N1 --> N2 --> N3
    N3 -->|passed| N4
    N3 -->|retry ≤ 2| N2
    N3 -->|retries spent| N5
    N1 -->|not answerable / error| N5
    N2 -->|error| N5
    N4 -->|error| N5

    N1 --- LLM
    N2 --- LLM
    N3 --- LLM
    N4 -->|stream tokens| LLM
    N4 -.->|token events| R5
```

### The graph in detail

```
                       ┌──────────── not answerable
                       │             node error after retries
                       │             step budget exhausted
                       ▼
START ─▶ understand ─▶ solve ─▶ self_check
                        ▲            │
                        └─retry ≤2───┤
                                     ├─ passed ────────▶ explain ──▶ END
                                     └─ retries spent ─▶ fallback ─▶ END
```

| Node | Does | Routing decision it emits |
|---|---|---|
| `understand` | Restates the question; infers subject, topic, difficulty, knowns/unknowns; decides whether it is answerable at all | `to_solve`, `to_fallback_not_answerable`, `to_fallback_node_error` |
| `solve` | Produces an answer plus step-by-step working. On a retry it receives the marker's objections and the rejected answer | `to_self_check`, `to_fallback_node_error` |
| `self_check` | A sceptical marker reviews the answer. Its verdict is **ANDed** with a confidence floor and deterministic guardrails | `to_explain`, `retry_solve`, `to_fallback_check_failed` |
| `explain` | Turns the *checked* answer into a step-by-step explanation, streamed token by token when a client is listening | `end`, `to_fallback_node_error` |
| `fallback` | Builds the explicit "partial answer, please verify" response | `end` |

Three design choices worth calling out:

- **Nodes decide, routers look up.** Each node computes the routing decision it
  implies, stores it on its own step record, and sets `state["last_decision"]`.
  Routers are then pure lookups. Branch logic lives next to the data it derives
  from, and *every* decision is persisted on the message.
- **Nodes are total.** A node never raises; failures become an error step plus
  `node_error`, and the router sends the turn to `fallback`. One auditable
  fallback path instead of scattered try/except.
- **The self-check cannot be talked out of a rejection.** `passed` is
  `model_says_passed AND confidence >= floor AND no deterministic issues`. A
  model that cheerfully approves `"The answer is TODO compute this"` is still
  overruled — there is a test for exactly that.

### Project layout

```
app/                     # FastAPI backend
├── agents/              # LangGraph: state, nodes, prompts, graph wiring
├── api/                 # routers, deps, error envelope, request-id middleware, SSE
├── core/                # JWT issuing/verification
├── db/                  # SQLAlchemy models, async session, repositories
├── llm/                 # provider-agnostic client (+streaming), retry, factory
├── models/              # Pydantic schemas, domain objects, enums
├── services/            # TutorService (write side), ConversationService (read side)
├── config.py            # pydantic-settings
└── logging_config.py
frontend/                # Streamlit client - no LLM, no DB, no prompts
├── streamlit_app.py
└── api_client.py
legacy/                  # v1, for reference only
alembic/                 # async migration environment + versions
scripts/                 # seed_demo.py
tests/                   # 270 tests, fully offline
```

---

## Tech stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ (type hints throughout, `mypy --disallow-untyped-defs`) |
| API | FastAPI 0.143, async/await, Pydantic v2 |
| Agent | LangGraph 1.2 `StateGraph` with a `TypedDict` state |
| Persistence | PostgreSQL 16, SQLAlchemy 2.1 async (asyncpg), Alembic |
| Auth | JWT (HS256) via PyJWT |
| LLM | Hugging Face Inference API (the v1 Llama-3 path), any OpenAI-compatible endpoint, or an offline stub |
| Streaming | `sse-starlette` server side, `httpx-sse` client side |
| Frontend | Streamlit 1.65 — API client only |
| Logging | structlog, JSON in production, `request_id` + `run_id` context |
| Tests | pytest, pytest-asyncio, pytest-cov; SQLite in-memory + scripted LLM |
| Quality | ruff (lint + format), mypy, GitHub Actions |

---

## Setup with Docker

`docker compose up` starts PostgreSQL, the API and the Streamlit UI, applies
migrations, and needs **no API key** — the default `LLM_PROVIDER=stub` is a
deterministic offline client.

```bash
git clone https://github.com/kanishksingh-arch/ai-homework-helper.git
cd ai-homework-helper
cp .env.example .env          # optional; compose has safe local defaults
docker compose up --build
```

- UI: <http://localhost:8501> (sign in with any name; password `homework`)
- API docs: <http://localhost:8000/docs>
- PostgreSQL: `localhost:5434`

To use the real v1 model (Llama-3 via Hugging Face), put this in `.env`:

```bash
LLM_PROVIDER=huggingface
LLM_API_KEY=hf_...
LLM_MODEL=meta-llama/Meta-Llama-3-8B-Instruct
```

Or any OpenAI-compatible endpoint:

```bash
LLM_PROVIDER=openai
LLM_API_KEY=sk-...
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini
```

Seed a demo learner with answered questions:

```bash
docker compose exec api python scripts/seed_demo.py
```

### Running locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,frontend]"

docker compose up -d db       # just PostgreSQL, on host port 5434
export DATABASE_URL="postgresql+asyncpg://tutor:tutor@localhost:5434/tutor"
alembic upgrade head

uvicorn app.main:app --reload                 # terminal 1
API_BASE_URL=http://localhost:8000 \
  streamlit run frontend/streamlit_app.py     # terminal 2
```

---

## API examples

Everything except `/health` and `/auth/token` needs a bearer token.

### 1. Sign in (creates the learner on first use)

```bash
TOKEN=$(curl -sS -X POST http://localhost:8000/auth/token \
  -H 'Content-Type: application/json' \
  -d '{"username":"ada","password":"homework"}' | jq -r .access_token)
```

### 2. Health (no auth)

```bash
curl -sS http://localhost:8000/health | jq
```

```json
{
  "status": "ok",
  "app": "ai-homework-helper",
  "environment": "local",
  "version": "2.0.0",
  "database": "up",
  "llm_provider": "stub",
  "llm_model": "stub-tutor-v1"
}
```

### 3. Start a conversation

```bash
CONV=$(curl -sS -X POST http://localhost:8000/conversations \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"title":"Physics revision","subject":"physics"}' | jq -r .id)
```

### 4. Ask a question — verified path

```bash
curl -sS -X POST "http://localhost:8000/conversations/$CONV/messages" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"content":"What is 17 * 23? Please show the working."}' \
  | jq '.assistant_message | {answer_status, confidence, retry_count, latency_ms,
          trace: [.trace[] | {sequence, node, routing_decision}]}'
```

```json
{
  "answer_status": "verified",
  "confidence": 0.88,
  "retry_count": 0,
  "latency_ms": 6,
  "trace": [
    { "sequence": 1, "node": "understand",  "routing_decision": "to_solve" },
    { "sequence": 2, "node": "solve",       "routing_decision": "to_self_check" },
    { "sequence": 3, "node": "self_check",  "routing_decision": "to_explain" },
    { "sequence": 4, "node": "explain",     "routing_decision": "end" }
  ]
}
```

### 5. History survives — the v1 → v2 headline

```bash
curl -sS "http://localhost:8000/conversations/$CONV" \
  -H "Authorization: Bearer $TOKEN" \
  | jq '{title, subject, message_count, roles: [.messages[].role]}'
```

```json
{
  "title": "Physics revision",
  "subject": "math",
  "message_count": 2,
  "roles": ["user", "assistant"]
}
```

In v1 this request did not exist, and a refresh of the browser lost the thread.

### 6. List your conversations (paginated)

```bash
curl -sS "http://localhost:8000/conversations?page=1&page_size=10" \
  -H "Authorization: Bearer $TOKEN" | jq '{total, total_pages, page}'
```

Supports `?status=active|archived` and `?subject=math|physics|...`.

### 7. Stream the explanation (SSE)

```bash
curl -sS -N -X POST "http://localhost:8000/conversations/$CONV/messages/stream" \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"content":"What is 15% of 240? Please show the working."}'
```

```
event: step
data: {"node": "understand", "sequence": 1, "status": "ok", "decision": "to_solve", "latency_ms": 0}

event: step
data: {"node": "solve", "sequence": 2, "status": "ok", "decision": "to_self_check", "latency_ms": 0}

event: step
data: {"node": "self_check", "sequence": 3, "status": "ok", "decision": "to_explain", "latency_ms": 0}

event: token
data: {"text": "Here is the reasoning, "}

event: token
data: {"text": "one step at a time. "}

...

event: done
data: {"conversation_id": "...", "assistant_message_id": "...", "answer_status": "verified",
       "confidence": 0.88, "retry_count": 0, "latency_ms": 7, "content": "..."}
```

Event contract:

| Event | When | Payload |
|---|---|---|
| `step` | Each graph node completes | `node`, `sequence`, `status`, `decision`, `latency_ms` |
| `token` | Each chunk of the explanation | `text` |
| `done` | The turn is persisted | message ids, `answer_status`, `fallback_reason`, `confidence`, `retry_count`, `latency_ms`, full `content` |
| `error` | The run failed | `code`, sanitised `message` |

`done.content` is authoritative: a **fallback** answer is not token-streamed
(there is nothing verified to stream), so a client must render `done.content`
rather than only the concatenated tokens. The Streamlit UI does exactly this.

### Error envelope

Every failure shares one shape, with the correlating `request_id`:

```json
{
  "error": {
    "code": "not_found",
    "message": "Conversation 0ec1... was not found",
    "details": { "conversation_id": "0ec1..." }
  },
  "request_id": "b09325f3278a4206af3acec330c63c20"
}
```

| Status | When |
|---|---|
| `200` | Reads, token issuing, SSE stream |
| `201` | Conversation created, or a turn completed (verified *or* unverified — both are successful runs) |
| `401` | Missing, malformed, expired or wrongly-signed token; unknown token subject |
| `404` | Conversation does not exist **or belongs to another learner** (deliberately indistinguishable) |
| `422` | Payload or query parameters failed validation |
| `500` | Unexpected error (internal detail is logged, never returned) |
| `503` | `/health` only — database unreachable |

---

## Running the tests

The suite is **fully offline**: SQLite in-memory for persistence and scripted
fake LLM clients for the model. No API key, no PostgreSQL, no network.

```bash
pytest                                              # 270 tests
pytest --cov=app --cov-report=term-missing          # coverage report
pytest tests/test_graph_routing.py -v               # just the graph branches
ruff check . && ruff format --check . && mypy app scripts frontend
```

Measured on the committed code (Python 3.12.13, `pytest 9.1.1`):

```
270 passed in 3.44s
TOTAL  1559 statements   56 missed   96% coverage
```

What the tests actually cover:

| Area | Examples |
|---|---|
| Every graph branch | verified path, retry→pass, retries exhausted→fallback, not-answerable, solve error, explain-after-pass error, step-limit guard |
| Self-check rigour | confidence floor overrides `passed: true`; placeholder/short/no-working answers overruled; retry prompt carries the marker's objections |
| Retry/fallback | transient error recovered, exponential delays asserted (`[0.5, 1.0, 2.0]`), permanent errors not retried, unparseable JSON retried then fallen back |
| Providers | OpenAI client incl. **SSE delta parsing and a malformed mid-stream frame**; HF client incl. the Llama-3 chat template and 503-is-transient; stub arithmetic |
| Persistence | per-conversation message sequencing, unique-sequence constraint, history window ordering, pagination boundaries, per-user scoping |
| Service | message/trace faithfulness, title derivation, subject backfill, `GraphRecursionError`→step-limit fallback, streaming event order |
| API | auth enforcement on every protected route, **cross-learner isolation on read/write/stream**, status codes, error envelopes, SSE contract |
| Regressions | structlog logger laziness (see "Bugs found while building") |

---

## Data model

```
users                    conversations                  messages
──────────────────       ──────────────────────────     ───────────────────────────
id        uuid PK        id            uuid PK          id               uuid PK
username  varchar(80) U  user_id       uuid FK → users  conversation_id  uuid FK idx
display_name varchar     title         varchar(200)     sequence         int
created_at timestamptz   subject       varchar(32) idx   role            varchar(16)
                         status        varchar(16) idx   content         text
                         message_count int              answer_status    varchar(16) idx
                         created_at    timestamptz      fallback_reason  varchar(32)
                         updated_at    timestamptz      confidence       float
                                                        retry_count      int
                                                        latency_ms       int
                                                        error            text
                                                        verified         bool
                                                        trace            jsonb
                                                        created_at       timestamptz
```

Indexes follow the queries the API issues:

- `users.username` is **unique** — the JWT subject resolves through it on every
  authenticated request.
- `(user_id, updated_at)` on `conversations` serves `GET /conversations`, which
  is always scoped to one learner and ordered by `updated_at DESC`.
- `(conversation_id, created_at)` on `messages` serves history replay.
- `(conversation_id, sequence)` is **unique**, so two messages can never occupy
  the same position in a conversation.

`JSONB` on PostgreSQL degrades to plain `JSON` on SQLite via `with_variant`,
which is what lets the test suite run with no database server.

---

## Design decisions and trade-offs

**Why a graph rather than one prompt.** The useful part is not "call the model
four times" — it is the *edges*. `self_check → solve` is a loop with a bound,
and `self_check → fallback` is a product decision about what a student sees
when the system is not confident. Expressing those as graph edges makes them
testable in isolation and persistable per turn; expressing them as prose inside
one prompt makes them unenforceable.

**The trace lives on the message, not in a fourth table.** The API exposes three
entities (users, conversations, messages) and the trace is a property of one
answer, so it is a `JSONB` column. The cost: you cannot efficiently query
"all runs where `self_check` failed" — that needs a JSONB index or a separate
`run_steps` table. At this scale, three tables that match the API beat four
that do not.

**`POST /messages` is synchronous; `POST /messages/stream` exists for the UI.**
The synchronous endpoint makes the API trivial to script and test, and returns
the full trace. The SSE endpoint is what makes the UI feel responsive. Both run
the identical graph.

**Real token streaming, via a queue rather than a second LLM call.** The
`explain` node takes an optional `emitter`; when present it uses
`llm.stream()` and pushes each chunk onto an `asyncio.Queue` that the SSE
endpoint drains. The alternative — generate fully, then re-chunk for display —
would be simpler but fake. The cost is that the streaming path builds a fresh
graph per request so the emitter can be bound to that request's queue.

**The streaming worker owns its own database session.** A streaming response
outlives FastAPI's request-scoped dependency teardown, so reusing the request
session would risk using it after close. Ownership is still checked *before*
the stream opens, so an unauthorised request is a normal JSON 404 rather than
an error event inside a 200 stream.

**Streaming is a capability, not an assumption.** `LLMClient.stream()` has a
default implementation that generates then chunks, so a provider without a
streaming API (Hugging Face Inference, the stub) still satisfies the same SSE
contract. `supports_streaming` tells you which you are getting.

**A 404, not a 403, for another learner's conversation.** Returning 403 would
confirm the id exists. Both paths return the same 404.

**The user row is looked up on every request, not trusted from the token.** A
token for a deleted account stops working immediately, at the cost of one
indexed query per request.

**No vendor SDK.** Providers are thin `httpx` clients behind one abstract
`LLMClient`. v1's `langchain-huggingface` dependency is gone; the Llama-3 chat
template it applied is now explicit in `HuggingFaceInferenceClient.build_prompt`
and unit-tested. Swapping providers is a config change, and tests mock at the
`LLMClient` seam rather than monkey-patching a library.

**An offline stub provider is a first-class citizen.** `LLM_PROVIDER=stub` makes
`docker compose up` and the whole test suite work with no credentials. It even
evaluates simple arithmetic (pattern-matched, never `eval`) so the demo
produces real answers. It is a control-flow demonstrator, explicitly not a
quality baseline.

**No jitter on the backoff.** Delays are exactly `base * 2^(n-1)` so the retry
tests can assert them. A high-concurrency deployment should add jitter.

**Streamlit keeps almost nothing in `session_state`** — only the current
session token and selected conversation id. History is re-fetched from
`GET /conversations/{id}` on every rerun. That is slightly more chatty and
entirely the point: the server is the source of truth.

---

## Bugs found while building this

Recorded because the fixes are instructive, and both now have regression tests:

1. **`LOG_JSON=true` silently did nothing.** `get_logger()` called
   `structlog.get_logger().bind(logger=name)`. `.bind()` on structlog's *lazy
   proxy* materialises the logger against whatever configuration is active —
   which, at module-import time, is structlog's default. Every module-level
   logger was frozen before `configure_logging()` ran. Fixed by passing initial
   values to `get_logger()` so the proxy stays lazy. (`logger` also collides
   with `wrap_logger`'s own parameter, hence `logger_name`.)
   See `tests/test_logging_config.py::TestLoggerLaziness`.
2. **`MissingGreenlet` when serialising a fresh turn.** `messages` rows are
   inserted by `conversation_id` rather than through the relationship, so
   `conversation.messages` was still unloaded; touching it during response
   serialisation triggered a lazy load inside asyncio. Fixed with an explicit
   `session.refresh(conversation, ["messages"])`. See
   `tests/test_tutor_service.py::TestSuccessfulTurn::test_messages_are_eager_loaded_for_serialisation`.

---

## Limitations

- **Auth is demo-grade.** One shared application password from the environment;
  the username is simply claimed, not proven. There is no password per user, no
  registration flow, no refresh tokens, no revocation. The JWT verification and
  per-user data isolation are real; the identity *proof* is a placeholder. Do
  not deploy this as-is.
- **The stub provider is not a quality baseline.** It is keyword rules plus a
  regex arithmetic evaluator. It demonstrates control flow, nothing about
  Llama-3's actual tutoring quality.
- **The self-check shares the solver's blind spots.** Asking the same model to
  mark its own work catches format errors, missing steps and low-confidence
  guesses — not systematic misconceptions it holds confidently. A second,
  different model as marker would be a real improvement.
- **No answer quality evaluation.** There is no labelled dataset here measuring
  how often `verified` answers are actually correct. The tests prove the
  *machinery* works, not that the tutoring is good.
- **Hugging Face streaming is not native** — it falls back to generate-then-chunk,
  so with `LLM_PROVIDER=huggingface` the first token arrives only after the full
  explanation is generated.
- **No rate limiting, quotas or cost accounting** on LLM calls.
- **The trace is lost if the process dies mid-turn** (one transaction per turn).
- **No conversation archiving endpoint** — the `archived` status exists in the
  model and filter, but nothing sets it.
- **Single-tenant.** No organisation/tenant column.
- **No file or image input**, so no diagram-based questions.

---

## Future work

- **A second model as marker**: run `self_check` against a different provider so
  the reviewer does not share the solver's blind spots.
- **A labelled evaluation set** of homework questions with known answers, plus a
  nightly CI job scoring verified-answer accuracy and failing on regression.
- **Real auth**: OAuth2/OIDC or per-user passwords with Argon2, refresh tokens,
  revocation.
- **LangGraph checkpointing**: a Postgres checkpointer so an interrupted turn
  resumes from its last node instead of restarting.
- **Native HF streaming** via the text-generation-inference streaming endpoint.
- **Retrieval**: ground answers in the student's own textbook or syllabus via
  pgvector, so "as your course defines it" becomes possible.
- **Student-facing feedback loop**: let a learner mark an answer wrong and store
  that against the trace, turning the audit trail into training data.
- **Observability**: OpenTelemetry spans per node, plus token/cost counters per
  message.
- **Image input** for photographed problems and diagrams.

---

## License

MIT — see [LICENSE](LICENSE).
