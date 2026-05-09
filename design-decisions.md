# Design Decisions — SHL Assessment Recommender

This document captures every architectural and implementation decision locked during the design interview. Each decision lists what was chosen, alternatives considered, and the reasoning. The order roughly follows the dependency tree of the design.

---

## 1. LLM Provider & Model Family

**Decision:** Google Gemini via Vertex AI.

**Alternatives considered:** Groq (Llama 3.3 / Llama 4), OpenRouter free models, OpenAI GPT-4o-mini, Anthropic Haiku.

**Reasoning:**
- User has Google AI credits, removing free-tier rate-limit anxiety.
- Vertex AI's `response_schema` is the cleanest mechanism for guaranteeing schema compliance on every turn — directly attacks the "schema compliance" hard eval.
- Gemini's structured output and function calling are both production-grade.
- Same SDK serves both the chat models and the embedding model.

---

## 2. Specific Models per Call Site

| Call site | Model | Temp | Notes |
|---|---|---|---|
| Router | `gemini-2.5-flash` | 0.0 | Deterministic intent classification; schema reliability is non-negotiable. |
| Recommend / refine selection | `gemini-2.5-flash` | 0.1 | Tool calling + closed-set ID emission. |
| Compare | `gemini-2.5-flash` | 0.2 | Grounded explanation; minor variation in phrasing. |
| Clarify | `gemini-2.5-flash` | 0.3 | Single coherent question; natural variation across turns. |
| Refuse / End | None (canned text) | — | No model call. |
| Embeddings | `gemini-embedding-001`, `output_dimensionality=768` | — | Matryoshka, latest quality, future-proof. |

**Why not `gemini-2.5-flash-lite`:** less reliable JSON schema adherence than Flash. The 500ms saved on clarify turns isn't worth two-model operational complexity.

**Why not `gemini-2.5-pro`:** scanned all 10 sample conversations — Flash handles every turn type cleanly. Pro's ~3s latency would eat the 30s budget. Reserved as a future fallback for compare turns flagged as ungrounded by self-check.

---

## 3. Agent Orchestration Architecture

**Decision: Three-tier hybrid — deterministic Policy + agentic Planning + deterministic Safety.**

```
POLICY  (deterministic — Python features + 1 structured-output call)
   Router with pre-computed feature bundle, no tools
       ↓
PLANNING  (agentic — tool-calling micro-agents per intent)
   Per-intent handler with curated toolkit + iter cap + fallback
       ↓
SAFETY/ASSEMBLY  (deterministic — Python only)
   ID validation, materialization, end_of_conversation, schema check
```

**Alternatives considered:**
- **Pure deterministic state machine (Option A):** simple, but every new intent requires re-encoding the plan into prompts; refinement edge cases force handler complexity.
- **Pure tool-calling agent loop (Option C):** flexible, but model decides when to recommend / when to refuse — turning behavior probes into prompt engineering. Loop overhead also blows the latency budget.

**Reasoning for hybrid:**
1. Hard guarantees stay structural — schema, hallucination, turn-1 vagueness, off-topic refusal all enforced by Python in policy/safety layers. The model literally cannot violate them.
2. Agency lives where it pays — inside `recommend`/`refine`/`compare`, the model can chain tool calls to handle messy refinements. This is where pure A would either need a hand-coded refinement DSL or fall over.
3. Scaling is additive — new intent = new router enum case + new handler with curated tool subset. Tools are reused across handlers.

---

## 4. Router Design — Pre-computed Feature Bundle (Not Tool-Calling Router)

**Decision:** Router is a **single structured-output call** that consumes a Python-pre-computed feature bundle. The router does **not** call tools.

**Alternatives considered:**
- **Dispatch-tools router (Flavor 1):** structured output dressed up as tool calling. Pure overhead.
- **Diagnostic-tools router (Flavor 2):** router calls `peek_catalog`, `parse_prior_shortlist` before deciding intent. Real capability gain but doubles router latency, erodes determinism (turn-1-vagueness becomes a prompt rule instead of a Python rule), and makes the router susceptible to retrieval seduction.

**The chosen design (Flavor 3):**
- Python deterministically pre-computes evidence (peek-retrieval, prior-shortlist state, vagueness, off-topic, injection signals) before the router fires.
- Evidence is stuffed into the router's prompt as context.
- Router emits one structured-output JSON: `{intent, search_query, filters, compare_pair, clarifying_question, refuse_reason, is_final_turn}`.
- Tools are exclusively a handler-layer concept.

**Reasoning:**
- **Same signal value as Flavor 2 with the latency and determinism of structured output.** No tool-call roundtrips.
- **Discipline:** tools = capabilities the LLM should reason about choosing; features = context the LLM always wants. Peek-retrieval, vagueness, and prior-shortlist presence are always wanted, so pre-compute them.
- **Open-set router** (where tools earn their keep) is a different system — not what this task needs.

---

## 5. Retrieval Strategy

**Decision:** Hybrid BM25 + dense embeddings, RRF fusion (k=60), hard facet pre-filters, numpy in-memory, **category-coverage candidate injection** (diversity-aware along the `test_type` axis).

**Components:**
- **BM25** (`rank_bm25`) over a structured concat per item (`NAME / KEYS / JOB_LEVELS / LANGUAGES / DURATION / DESCRIPTION_HEAD`). Sharp on technical/exact match.
- **Dense** via `gemini-embedding-001` @ 768 dims. Captures abstract intent ("safety-critical reliability" → DSI).
- **RRF fusion** with k=60 — robust to score-scale differences between BM25 and dense.
- **Hard facet pre-filters** applied before scoring: `test_type`, `languages`, `duration_max`, `job_level`, `remote`, `adaptive`. Surgical filtering before fuzzy retrieval.
- **Category-coverage injection:** if the conversation context implies a `test_type` category and retrieval doesn't surface one in top-K, inject the catalog's strongest exemplar of that category into the candidate pool. The LLM selector still makes the final pick.

**Why not standalone reranker (cross-encoder or LLM-as-reranker):**
- The recommend/refine handler's selection LLM call already serves as a reranker — it scores candidates against full conversation context and picks the final shortlist. Adding a separate reranker duplicates this for a 2–5% gain at a 1.5s latency cost. Save the reranker pattern for >10K corpora.

**Why not multi-query expansion (RAG-Fusion):**
- Vague queries are routed to `clarify` and never reach retrieval — the case expansion targets is small.
- Conversation-aware query composition by the router (using full message history, not just latest) achieves most of expansion's benefit for free in an LLM call we're already making.

**Why category-coverage isn't gaming the metric:**
- Always-on default injection would be metric-fitting (encoding patterns from the visible 10 traces).
- Category-coverage is **gap-filling, not always-on** — exemplars only enter the candidate pool when retrieval underrepresents an implied category. The LLM still selects with full conversation context, and explicit user exclusions still drop categories.
- This is the standard IR pattern of diversity-aware candidate generation, applied along the `test_type` taxonomy. It would still be the right design under Recall@5 / Recall@20 / MRR / human eval.

---

## 6. Catalog & Index Storage

**Decision:** Pre-compute embeddings + BM25 index at build time and bake artifacts into the Docker image.

| Artifact | Format | Purpose |
|---|---|---|
| `catalog.parquet` | normalized records | Fast load, immutable runtime data |
| `embeddings.npy` | float32 (377, 768) | numpy in-memory cosine retrieval |
| `bm25_index.pkl` | rank-bm25 pickle | Lexical retrieval |

**Reasoning:**
- 377 items: numpy beats FAISS / Chroma — no native binary dep, no extra latency.
- Catalog is static (`scraped_at` is dated). No reason to embed at startup. Pre-baking saves ~10–30s of cold-start time.
- One artifact directory keeps the deploy artifact reproducible and inspectable.

---

## 7. Catalog Scope — No Further Filtering

**Decision:** Trust the provided `shl_product_catalog.json` (377 items) as already filtered to Individual Test Solutions. Do not apply name-based pre-packaged filters.

**Reasoning:**
- No field in the JSON disambiguates Individual Test Solutions vs Pre-packaged Job Solutions.
- All 377 items share identical schema and URL pattern (`/products/product-catalog/view/...`).
- Sample conversations actively recommend items that match a pre-packaged naming heuristic (C3's "Customer Service Phone Simulation", C5's "Sales Transformation 2.0", C6's "Safety & Dependability 8.0").
- Filtering would directly hurt Recall@10 on traces that follow these patterns.

**Note:** the JSON has a JSON control-character issue around line 4795 — `normalize.py` will sanitize and re-emit clean parquet.

---

## 8. Conversation State — Embedded Markdown Tables

**Decision:** Persist the active shortlist by embedding a markdown table in each assistant `reply` whenever a shortlist is committed. On the next turn, parse the most recent assistant reply's table to recover the prior shortlist for refinement.

**Reasoning:**
- The API request schema is stateless — it carries `messages[]` only; the prior `recommendations[]` array is **not** echoed back.
- Sample conversations consistently embed markdown tables in recommend/refine turns.
- The markdown table format is readable to the user, parseable in Python (using a lookup of `name` → `entity_id` against the catalog), and serves as the durable cross-turn state.

---

## 9. Behavior Model

**The four conversational behaviors:**

| Intent | Trigger | Handler |
|---|---|---|
| `clarify` | Insufficient context (vague query, missing role/seniority) | One LLM call, no tools, returns single clarifying question |
| `recommend` | Sufficient context for a fresh shortlist | Tool-using loop: search → optional get_assessment → selection |
| `refine` | User modifies prior shortlist (add / drop / swap constraints) | Parse prior shortlist → diff with router-emitted deltas → re-search → selection |
| `compare` | User asks differential question between two assessments | Parallel `get_assessment` × 2 → grounded explainer (no tools after fetch) |
| `refuse` | Off-topic, legal, prompt injection, general hiring advice | Canned template per sub-category, no LLM |

**Refuse sub-categories with templates:**

| Sub-category | Detection | Template hook |
|---|---|---|
| Prompt injection | regex / keyword detector ("ignore previous", "system prompt", "you are now") | "I'm scoped to recommending SHL assessments. What role are you hiring for?" |
| Off-topic | embedding cosine to anchor + LLM judgment in router | "I can help with SHL assessment selection — not [topic]. Want to share the role instead?" |
| Legal / regulatory | keyword detector ("legally required", "HIPAA", "EEOC", "lawsuit") | "That's a legal question outside what I can advise on. I can help select assessments; legal interpretation is for your counsel." |
| General hiring advice | LLM judgment | "I focus on assessment selection. For broader hiring strategy, your TA team is the right resource. Want to share role specifics?" |

**Critical rule: `refuse` never sets `end_of_conversation = true`.** The conversation may continue (sample C7 turn 3 demonstrates this).

---

## 10. `end_of_conversation` Semantics

**Decision:** Orthogonal `is_final_turn: bool` flag emitted by the router. The assembly layer computes `end_of_conversation = is_final_turn AND len(recommendations) > 0`.

**Alternatives considered:**
- **`end` as a separate intent:** can't model C10's refine+end-in-same-turn pattern. Forces a turn split that the 8-turn cap punishes.

**Detection signals (pre-computed features feeding the router):**
- `last_user_confirmation`: regex against confirmation lexicon ("perfect", "confirmed", "locked in", "that works", "good")
- `has_prior_shortlist`: parsed from prior assistant messages
- The router judges whether the user is also asking new questions in the same message

**Invariant:** assembly enforces that `end=true` is impossible without a non-empty shortlist.

---

## 11. Turn-Budget Awareness

**Decision:** Encode 8-turn cap awareness in the router prompt + features.

**Pre-computed feature:** `turns_remaining = 8 - len(messages)`.

**Router rule:** *"If `turns_remaining <= 2` and no prior shortlist exists, prefer `recommend` over `clarify`; ask only the single most important clarification if still vague."*

**Reasoning:** prevents the failure mode "agent clarifies forever and never recommends." The simulated user may be slow to confirm; we must commit a shortlist before turns run out.

---

## 12. Fallback Hierarchy

**Five-tier fallback for failure modes:**

| Failure mode | Strategy |
|---|---|
| Retrieval returns 0 candidates after hard filtering | Relax filters in priority order (drop `duration_max` first → `language` → `test_type`). If still empty, switch intent to `clarify` with a catalog-grounded explanation. |
| Selection LLM emits IDs not in catalog | Validator drops invalid IDs. ≥1 valid → emit those. 0 valid → fall back to top-K of retrieval by score. Logged as `validation_dropped: N`. |
| LLM tool call malformed | One retry with stricter prompt. Second failure → fall back to single-shot structured output (no tools). Third failure → canned safe response. Logged. |
| LLM call timeout / 5xx | Single retry with exponential backoff (300ms, 900ms). Total failure → canned response. |
| Compare target not in catalog | Fuzzy match against catalog names (RapidFuzz `WRatio` ≥ 85). No match → respond *"X isn't in the SHL catalog — did you mean [closest 2]?"*. Bridges into clarify-style continuation. |

---

## 13. Tool Inventory

**LLM-exposed tools** (function declarations passed to Gemini):

| Tool | Purpose |
|---|---|
| `search_catalog(query, filters)` | Hybrid BM25+dense retrieval with optional facet filters. Returns top-K with `entity_id`, `name`, `score`, `snippet`. |
| `get_assessment(entity_id)` | Full record lookup. |
| `find_similar(entity_id, k)` | Embedding-neighbor search. |
| `list_facets(query)` | Disambiguation helper (e.g., "which English variant for SVAR?"). |

**Python-internal utilities** (not exposed to LLM):

| Utility | Purpose |
|---|---|
| `parse_prior_shortlist(messages)` | Recover IDs from last assistant markdown table. |
| `validate_ids(ids)` | Filter to catalog. |
| `materialize(ids)` | `[{name, url, test_type}]`. |
| `detect_user_confirmation(text)` | Drives `is_final_turn`. |
| `is_vague(messages)` | Drives turn-1 no-recommend rule. |

**Per-handler tool exposure:**

| Handler | Exposed tools | Max iterations |
|---|---|---|
| `clarify` | None | 1 (single-shot) |
| `recommend` | `search_catalog`, `list_facets` | 3 |
| `refine` | `get_assessment`, `search_catalog`, `find_similar` | 3 |
| `compare` | `get_assessment` | 1 (parallel ×2 in Python) |
| `refuse` | None (canned) | 0 |

---

## 14. Deployment Target — GCP Cloud Run

**Decision:** Cloud Run in `us-central1` with Vertex AI Gemini auth via service account.

**Alternatives considered:** Render, Modal, Fly.io, Railway, Hugging Face Spaces, Cloudflare Workers.

**Reasoning:**
- **Intra-region networking to Vertex AI Gemini.** Saves ~30–80ms per LLM call vs cross-cloud. Real over a multi-turn conversation.
- **Service-account auth (no API keys in env).** IAM-based via `aiplatform.user` role. Production-grade.
- **80 concurrent requests / instance default.** Catalog and embeddings load once per instance, serve all in-flight requests.
- **Free tier covers eval load** (2M req/mo, 360k vCPU-s, 180k GiB-s).
- **`gcloud run deploy --source .`** for fast deploys.

**Service config:**
- Region: `us-central1`
- Memory: 512Mi
- CPU: 1 vCPU
- Concurrency: 80
- Min instances: 0
- Max instances: 4

**Cold start budget:** container pull (~1s) + Python boot (~1s) + FastAPI (~0.5s) + load embeddings pickle (~1s) ≈ 4–6s cold. The task's 2-minute allowance is overkill.

**`/health` is the readiness gate:** returns 200 only after catalog/embeddings have loaded. Prevents `/chat` traffic on a half-loaded instance.

---

## 15. Observability

**Decision:** Structured JSON logging to stdout. Cloud Logging auto-ingests with field-level indexing. No OpenTelemetry, no metrics endpoint.

**Per-request log shape:**
```json
{"request_id": "...", "turn_index": 3, "intent": "refine", "is_final_turn": false,
 "latency_ms": {"total": 2840, "features": 18, "router": 720, "handler": 2010, "assembly": 92},
 "retrieval": {"bm25_top_k": 30, "dense_top_k": 30, "fused_top_k": 20, "candidates_after_filter": 14, "category_injections": 1},
 "llm_calls": {"count": 2, "tokens_in": 4120, "tokens_out": 380, "model": "gemini-2.5-flash"},
 "fallbacks_triggered": [], "validation_errors": [],
 "recommendations_count": 5, "end_of_conversation": false}
```

**Per-LLM-call log entry (DEBUG level):** `{model, latency_ms, finish_reason, tool_calls_count, retry_count, schema_compliant}`.

**PII discipline:** log `request_id` (UUID) at INFO; message content at DEBUG only as a hash.

---

## 16. Testing Strategy

**Three blocking layers + behavior probes:**

| Layer | Purpose | Blocking? |
|---|---|---|
| Unit tests | Pure functions: feature pipeline, validators, materializers, RRF math, normalizer | Yes |
| Integration tests (mocked Gemini) | Full `/chat` round-trips with deterministic LLM stubs. Schema validation on every response. Edge cases: 0 retrieval results, malformed tool call, missing prior shortlist. | Yes |
| Replay harness (Flash-simulated user) | Persona + facts + expected shortlist extracted from `sample_conversations/C*.md`. Computes Recall@10 + schema compliance + behavior probes locally. | Yes |
| Behavior probes | turn-1-vagueness, prompt injection, off-topic, hallucination probe (fake assessment name) | Yes (subset) |
| Latency / load tests | `vegeta` or `locust`, p95 < 8s under 10 RPS | Optional |

**Replay simulated user:** small Flash call with the trace as system prompt:
> *"You are simulating a user with these facts: <facts>. Respond truthfully when asked. Say 'no preference' for things outside your facts. Confirm and end when the agent commits a shortlist matching your needs."*

**Iteration loop:** every prompt/feature change runs the replay harness; Recall@10 and probe pass rates are tracked.

**No CI/CD initially:** single-developer iteration via `gcloud run deploy --source .`. Add CI only if iteration pain justifies the ~30min Workload Identity Federation setup.

---

## 17. Things Explicitly NOT Built (YAGNI)

| Not built | Reason |
|---|---|
| CI/CD pipeline | Manual `gcloud` deploys are fast enough for a single developer. |
| Standalone reranker | LLM selection step is the reranker. |
| Multi-query expansion | Router does context-aware query composition. |
| Prompt caching | Prompts are <2k tokens; Vertex caching minimum is 32k. Note as future optimization. |
| OpenTelemetry / distributed tracing | Single-service deploy. |
| Per-conversation state on the server | Stateless per spec — state lives in `messages[]`. |
| Auth on `/chat` | Out of scope for this submission. |
| FAISS / Chroma / pgvector | numpy is sufficient at 377 items. |
| `gemini-2.5-flash-lite` mixing | 500ms saved on clarify is not worth two-model operational complexity. |
| `gemini-2.5-pro` | Flash handles every sample turn type cleanly. |

---

## Tech Stack Summary

| Concern | Choice |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI + Uvicorn |
| Dep manager | `uv` |
| Validation | Pydantic v2 |
| LLM SDK | `google-genai` (Vertex mode) |
| Lexical retrieval | `rank-bm25` |
| Vector retrieval | numpy cosine |
| Fuzzy match | `rapidfuzz` |
| Logging | `structlog` (JSON renderer) |
| Tests | `pytest`, `pytest-asyncio` |
| Container | `python:3.11-slim` (multi-stage Dockerfile) |
| Hosting | GCP Cloud Run, `us-central1`, Vertex AI Gemini |
