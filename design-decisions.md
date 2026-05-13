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
| Recommend / refine selection | `gemini-2.5-flash` | 0.1 | Closed-set ID emission over a Python-built candidate pool. |
| Compare | `gemini-2.5-flash` | 0.2 | Grounded explanation; minor variation in phrasing. |
| Clarify | `gemini-2.5-flash` | 0.3 | Single coherent question; natural variation across turns. |
| Refuse / End | None (canned text) | — | No model call. |
| Embeddings | `gemini-embedding-001`, `output_dimensionality=768` | — | Matryoshka, latest quality, future-proof. |

**Why not `gemini-2.5-flash-lite`:** less reliable JSON schema adherence than Flash. The 500ms saved on clarify turns isn't worth two-model operational complexity.

**Why not `gemini-2.5-pro`:** scanned all 10 sample conversations — Flash handles every turn type cleanly. Pro's ~3s latency would eat the 30s budget. Reserved as a future fallback for compare turns flagged as ungrounded by self-check.

---

## 3. Agent Orchestration Architecture

**Decision: Three-tier hybrid — deterministic Policy + per-intent Planning + deterministic Safety.**

```
POLICY  (deterministic — Python features + 1 structured-output call)
   Router with pre-computed feature bundle, no tools
       ↓
PLANNING  (per-intent handlers)
   Python candidate generation + closed-set LLM selection/explanation
       ↓
SAFETY/ASSEMBLY  (deterministic — Python only)
   ID validation, materialization, end_of_conversation, schema check
```

**Alternatives considered:**
- **Pure deterministic state machine (Option A):** simple, but every new intent requires re-encoding the plan into prompts; refinement edge cases force handler complexity.
- **Pure tool-calling agent loop (Option C):** flexible, but model decides when to recommend / when to refuse — turning behavior probes into prompt engineering. Loop overhead also blows the latency budget.

**Reasoning for hybrid:**
1. Hard guarantees stay structural — schema, hallucination, turn-1 vagueness, off-topic refusal all enforced by Python in policy/safety layers. The model literally cannot violate them.
2. Agency lives where it pays — inside `recommend`/`refine`/`compare`, the model reasons over catalog-grounded records or a closed candidate pool, but Python owns retrieval, validation, and fallback control.
3. Scaling is additive — new intent = new router enum case + new handler with a narrow interface. Catalog tool utilities remain available for future tool-loop variants, but the active recommend/refine path is explicit Python orchestration plus structured selection.

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

## 5b. Catalog Query Expansion (Slice 10c addition)

**Decision (amends §13 "Multi-query expansion" rejection):** add a deterministic candidate-expansion layer in front of the closed candidate pool. Lives in `src/shl_recommender/catalog/query_expansion.py`. Runs *before* the LLM selection call inside `handle_recommend` and merges expansion hits into the candidate pool (capped at `MAX_CANDIDATES_TO_LLM = 18`).

**Why we revised the §13 stance:** during live replay we observed that BM25 + dense + RRF, even with category-coverage injection, was failing to surface obvious catalog matches when the user phrased a skill in a way the retrieval text didn't share lexical or semantic surface with. Concrete failures: `DSI` ↔ `Dependability and Safety Instrument`, `Docker` ↔ a product literally named `Docker`, `OPQ` ↔ `Occupational Personality Questionnaire OPQ32r`. Multi-query *rewriting* (RAG-Fusion) was still the wrong tool for these — they're alias-resolution problems, not query-ambiguity problems. The right tool is deterministic alias matching against the catalog itself.

**Two halves, with different metric-fitting profiles:**

1. **Catalog-derived aliases** (`_aliases_for_name`) — at build/load time we generate aliases from each `CatalogItem.name`: full normalized name, parenthetical acronyms (`(DSI)` → `dsi`), uppercase-acronym tokens (`OPQ32r`, `G+`), acronym prefixes (`opq` from `opq32r`), and meaningful single-skill tokens (`docker`, `aws`, `excel`, `linux`, `spring`, `sql`). Query text that contains an alias promotes the alias's owning item into the candidate pool. **Catalog-derived** — generated by reading product names, no hand-curation against personas, would still work if the catalog tripled in size.

2. **Domain concept rules** (`_CONCEPT_RULES`) — 10 hand-curated rules keyed on regex triggers (e.g. `\bcontact cent(?:er|re)\b`, `\bsales\b`, `\bhipaa\b`) that promote a fixed handful of canonical catalog items per concept. **Hand-curated.**

**Honest disclosure of the metric-fitting risk:** the 10 concept rules are sample-informed and several correspond to visible trace archetypes (`contact_center`, `healthcare_administration`, `software_engineering_stack`, `office_productivity`, etc.). Mean Recall@10 improved from the first failed live run to 0.98 through a combination of fixes: Gemini JSON hardening, larger structured-output budgets, staged filter relaxation, catalog-derived alias expansion, and these concept rules. The concept rules are the highest metric-fitting risk because they encode visible scenario patterns as reusable domain concepts. Acknowledged trade-offs:

- The rules are framed as reusable hiring scenarios, not as `if sample_id == "C7"` branches. They should generalize within a domain.
- The closed-set ID validator is still the structural guarantee — expansion hits cannot become recommendations unless they exist in `data/build/catalog.parquet`. No hallucination path is opened.
- A held-out evaluation set (not built — out of scope for this submission) would reveal real-world recall vs sample-replay recall. Worth running before claiming production readiness.
- **Future direction:** lift the concept rules into catalog-derived clusters (e.g. tf-idf over descriptions → topic groups) so the layer becomes fully catalog-driven and metric-fitting risk disappears.

**Output promotion (`_promote_alias_ids` in `recommend.py`):** after the LLM emits its closed-set selection, expansion-matched items are *promoted to the top* of the final shortlist before LLM-selected items, bounded by `MAX_SHORTLIST = 8`. This is more aggressive than candidate-pool injection alone — it directly biases output toward expansion hits when they're present in the query. Logged via `retrieval.expanded_candidates` + `retrieval.matched_concepts` for observability.

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
| `recommend` | Sufficient context for a fresh shortlist | Query expansion + hybrid retrieval + closed-set selection |
| `refine` | User modifies prior shortlist (add / drop / swap constraints) | Parse prior shortlist → diff with router-emitted deltas → re-search → selection |
| `compare` | User asks differential question between two assessments | Parallel catalog-name resolution → grounded explainer |
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
| Retrieval returns 0 candidates after hard filtering | Relax filters in priority order: `duration_max` → `languages` → `test_types` → `job_levels` → `remote_only` + `adaptive_only`. If still empty, return a catalog-grounded "couldn't find X" message and ask the user to relax a constraint. |
| Selection LLM emits IDs not in catalog | Validator drops invalid IDs. ≥1 valid → emit those. 0 valid → fall back to top-K of retrieval by score. Logged as `dropped_invalid_or_out_of_pool_ids:N`. |
| Selection LLM returns truncated/invalid JSON | **JSON salvage** (Slice 10c) regexes `"\d+"` substrings out of the partial output and keeps only IDs already in the closed candidate pool. ≥1 salvaged → emit those with `salvaged_ids_from_invalid_json` validation flag. 0 salvaged → top-K fallback. The closed-set guarantee is preserved end-to-end. |
| LLM call timeout / 5xx | Tenacity `AsyncRetrying` with exponential backoff (multiplier 0.3, max 2.0s), 3 attempts, retrying only on `_is_transient(exc)` (timeouts, 5xx, rate limits, connection errors). Per-call wrapped in `asyncio.wait_for(timeout=SHL_LLM_TIMEOUT_SECONDS)`. Total failure raises `LLMError` → handler-specific fallback. |
| Whole `/chat` deadline exceeded | `SHL_REQUEST_TIMEOUT_SECONDS` (28s) bounds the entire turn so the 30s eval cap is never violated. |
| Compare target not in catalog | Fuzzy match against catalog names (RapidFuzz `WRatio` ≥ 85). No match → respond *"X isn't in the SHL catalog — did you mean [closest 2]?"*. Bridges into clarify-style continuation. |

---

## 13. Tool Inventory

**Catalog tool utilities** (implemented and available for future tool-loop variants; the active recommend/refine handlers call Python retrieval directly):

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

**Active per-handler planning path:**

| Handler | Path |
|---|---|
| `clarify` | One text-generation call, no tools |
| `recommend` | Query expansion + hybrid retrieval + structured selection |
| `refine` | Parse prior shortlist + retrieve/merge candidates + structured selection or prior fallback |
| `compare` | Python resolves both catalog records, then one grounded explanation call |
| `refuse` | Canned template, no model call |

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
 "timings": {"features_ms": 18, "router_ms": 720, "handler_ms": 2010, "assembly_ms": 92, "total_ms": 2840},
 "retrieval": {"candidates_after_filter": 18, "coverage_letters": ["P", "A"], "expanded_candidates": 4},
 "llm_calls": {"count": 2, "tokens_in": 4120, "tokens_out": 380, "models": ["gemini-2.5-flash"], "timeouts": 0},
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
| Integration tests (mocked Gemini) | Full `/chat` round-trips with deterministic LLM stubs. Schema validation on every response. Edge cases: hallucinated IDs dropped, refine drop-X removes-X, compare uses grounded records, missing prior shortlist. | Yes |
| Replay harness (Flash-simulated user) | Persona + facts + expected shortlist extracted from `sample_conversations/C*.md`. Computes Recall@10 + schema compliance + behavior probes locally. | Yes |
| Behavior probes | turn-1-vagueness, prompt injection, off-topic, hallucination probe (fake assessment name) | Yes (subset) |
| Latency / load tests | `vegeta` or `locust`, p95 < 8s under 10 RPS | Optional |

**Replay simulated user:** small Flash call with the trace as system prompt:
> *"You are simulating a user with these facts: <facts>. Respond truthfully when asked. Say 'no preference' for things outside your facts. Confirm and end when the agent commits a shortlist matching your needs."*

**Iteration loop:** every prompt/feature change runs the replay harness; Recall@10 and probe pass rates are tracked.

**No CI/CD initially:** single-developer iteration via `gcloud run deploy --source .`. Add CI only if iteration pain justifies the ~30min Workload Identity Federation setup.

---

## 17b. Runtime Configuration — Env-Driven Settings

**Decision:** Every model name and sampling temperature is sourced at runtime from environment variables (with defaults matching the locked design). Implemented in `src/shl_recommender/config.py` as a frozen `Settings` dataclass exposed via a `@lru_cache`d `get_settings()` accessor. `.env` at the project root is auto-loaded via `python-dotenv` from `main.py` and every entry-point script.

**Rationale:**
- Lets us A/B model swaps (Flash → Flash-Lite, 768 dims → 256) without touching code or rebuilding images.
- Single source of truth: changing `.env` flows through router, all four handlers, embedding pipeline, and `scripts/build_index.py` simultaneously.
- Tests can pin specific values via `monkeypatch.setenv` + `reset_settings_cache()` for deterministic behavior independent of operator environment.

**Env vars exposed:**

| Variable | Default | Purpose |
|---|---|---|
| `SHL_ROUTER_MODEL` | `gemini-2.5-flash` | Policy-layer LLM |
| `SHL_HANDLER_MODEL` | `gemini-2.5-flash` | All handler LLM calls |
| `SHL_EMBEDDING_MODEL` | `gemini-embedding-001` | Build index + query-time |
| `SHL_EMBEDDING_DIMS` | `768` | Matryoshka output dims |
| `SHL_EMBEDDING_BATCH_SIZE` | `32` | Vertex per-request size |
| `SHL_ROUTER_TEMPERATURE` | `0.0` | Deterministic policy |
| `SHL_RECOMMEND_TEMPERATURE` | `0.1` | Selection over closed pool |
| `SHL_REFINE_TEMPERATURE` | `0.1` | Same |
| `SHL_COMPARE_TEMPERATURE` | `0.2` | Grounded explainer |
| `SHL_CLARIFY_TEMPERATURE` | `0.3` | Natural variation |
| `SHL_TOP_P` | `0.95` | Shared across all calls |
| `SHL_LLM_TIMEOUT_SECONDS` | `10.0` | Per-call deadline (wraps every Gemini RPC) |
| `SHL_REQUEST_TIMEOUT_SECONDS` | `28.0` | Whole `/chat` deadline, keeps us under the 30s eval cap |

**Auth env vars** (handled in `agent/llm.py`): `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`, `GCP_API_KEY` (or `GOOGLE_API_KEY`/`GEMINI_API_KEY` aliases), or Application Default Credentials.

**Backward-compat:** `agent/llm.py` exposes `ROUTER_MODEL`, `HANDLER_MODEL`, `EMBEDDING_MODEL`, `EMBEDDING_DIMS` via `__getattr__` resolving through `get_settings()` so any external import site continues to work.

**Invalid values fall back to defaults silently** — a typo in `.env` doesn't break the service.

**Template:** `.env.example` at the project root documents every variable.

---

## 17c. Gemini Call Configuration (Slice 10c additions)

**Decision:** every `generate_*` call in `LLMClient` sets two SDK config flags that the earlier slices left on their defaults, and structured-output calls run their schema through a sanitizer before it goes on the wire.

| Flag | Setting | Why |
|---|---|---|
| `thinking_config` | `ThinkingConfig(thinking_budget=0)` | Gemini 2.5 spends an opaque chunk of `max_output_tokens` on thinking. With structured output capped at 2048 tokens, the recommend handler was running out of visible-output budget mid-JSON. Disabling thinking moved the budget back to the JSON we actually need. Live impact: invalid-JSON warnings on recommend dropped near-zero. |
| `automatic_function_calling` | `AutomaticFunctionCallingConfig(disable=True)` | The SDK's AFC path can silently invoke local Python callables passed as tools. Our live handlers use explicit Python orchestration, and any future tool loop should remain under our control. Disabling AFC removes a surprise execution path. |
| `pydantic_to_gemini_schema(...)` | Sanitizer in `agent/llm.py` | AI Studio's protobuf `Schema` rejects `additionalProperties`, raw `$ref`, and `anyOf: [{X}, {type: null}]`. Pydantic v2 emits all three. Sanitizer strips `additionalProperties` / `title` / `default` / `$defs`, inlines `$ref`s, and rewrites `anyOf`-with-null into `nullable: true`. Same payload now works on Vertex AI and AI Studio. |

**Schema-compliance observability:** structured-output calls pass a `schema_compliance_check` callable into `_call`; the result is logged on every `llm_call` DEBUG line as `schema_compliant=True|False`. Visible in per-request telemetry without exposing message content.

**Per-request LLM stats:** a `ContextVar`-scoped dict accumulates `count`, `tokens_in`, `tokens_out`, `models`, `timeouts` across all LLM calls inside a single `/chat`. Surfaced via `begin_llm_stats()` / `end_llm_stats()` in `agent/runner.py`. Survives concurrent requests because of `ContextVar` isolation; no global state.

---

## 18. Things Explicitly NOT Built (YAGNI)

| Not built | Reason |
|---|---|
| CI/CD pipeline | Manual `gcloud` deploys are fast enough for a single developer. |
| Standalone reranker | LLM selection step is the reranker. |
| Multi-query expansion (RAG-Fusion query rewriting) | Router still composes context-aware queries. We did add deterministic *alias* expansion + domain concept rules — see §5b — but that's catalog-side promotion, not query-side rewriting. |
| Prompt caching | Prompts are <2k tokens; Vertex caching minimum is 32k. Note as future optimization. |
| OpenTelemetry / distributed tracing | Single-service deploy. |
| Per-conversation state on the server | Stateless per spec — state lives in `messages[]`. |
| Auth on `/chat` | Out of scope for this submission. |
| FAISS / Chroma / pgvector | numpy is sufficient at 377 items. |
| `gemini-2.5-flash-lite` as default | 500ms saved on clarify is not worth two-model operational complexity — but `SHL_HANDLER_MODEL=gemini-2.5-flash-lite` would override per Section 17b. |
| `gemini-2.5-pro` as default | Flash handles every sample turn type cleanly — but `SHL_HANDLER_MODEL=gemini-2.5-pro` would override per Section 17b. |
| Pydantic-Settings library | The `config.py` accessor is small enough that pulling another dep wasn't justified. |
| Per-environment config files (dev/staging/prod) | A single `.env` plus per-env override of vars at the deploy layer covers our scope. |

---

## Tech Stack Summary

| Concern | Choice |
|---|---|
| Language | Python 3.11 |
| Web framework | FastAPI + Uvicorn |
| Dep manager | `uv` |
| Validation | Pydantic v2 |
| LLM SDK | `google-genai` (Vertex AI Gemini OR AI Studio surface) |
| Lexical retrieval | `rank-bm25` |
| Vector retrieval | numpy cosine |
| Fuzzy match | `rapidfuzz` |
| Logging | `structlog` (JSON renderer) |
| Retry | `tenacity` (async, exponential backoff) |
| Config | `python-dotenv` + `src/shl_recommender/config.py` (env-driven `Settings`) |
| Tests | `pytest`, `pytest-asyncio` (112 tests: 90 unit + 10 integration + 12 replay) |
| Container | `python:3.11-slim` (multi-stage Dockerfile, embeddings pre-baked) |
| Hosting | GCP Cloud Run, `us-central1` (project `shl-recommender-495908`) |

## Live verified results (pre-deploy)

| Metric | Value | Source |
|---|---|---|
| Mean Recall@10 (10 personas) | **0.98** | `scripts/replay_live.py` artifact `data/replay_runs/replay_1778512594.jsonl` |
| Schema compliance | 1.0 across every turn | same |
| Behavior probes | 7/7 pass | `scripts/probes_live.py` |
| Unit + integration + replay | 112/112 pass | `uv run pytest` |

These were measured against the sample-conversation personas. See §5b for the metric-fitting caveat on the concept-rule half of query expansion — these numbers describe sample-replay behavior, not held-out generalization.
