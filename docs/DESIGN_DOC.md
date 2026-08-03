# CouncilAI: System Design Document

**Last Updated:** May 2026

---

## 1. Context & Scope

**CouncilAI** is a multi-agent document deliberation and Q&A engine.

Instead of relying on a single Large Language Model (LLM), CouncilAI coordinates a multi-agent workflow: a **Router Agent** classifies queries (using BME sampling for document summary routing), parallel **Council Member Agents** propose responses, a **Peer-Review Loop** cross-evaluates candidates, a **Chairman Agent** synthesizes the consensus, and a **Self-Reflection Agent** audits the results. This produces answers with built-in confidence scoring and reasoning chains.

### User Journeys (In Scope)
* **Document Q&A**: Upload a PDF → ask questions → get council-synthesized answers grounded in the document.
* **General Q&A**: Ask questions without a document → get answers grounded with real-time Web Search.
* **Document Explanation**: Get adaptive explanations (beginner/intermediate/advanced depth).
* **Assessment Generation**: Generate MCQ or subjective questions from document content.

---

## 2. Goals & Non-Goals

### Goals
1. **Consensus-driven accuracy** via multi-model evaluation.
2. **Confidence scoring** — every answer carries a numeric confidence rating.
3. **High Throughput** — utilizes Redis Stack Vector Similarity Search (RediSearch) semantic caching to reduce redundant LLM API calls.
4. **Extensibility** — designed for modular integration of LLMs and OCR backends.
5. **Local Compatibility** — support for offline local vLLM models.

### Non-Goals
* **Multi-Year Searchable Chat Archiving**: While short-to-medium multi-turn session memory is supported via Redis (`ConversationStore`), persistent multi-year searchable archiving across historical user accounts is out of scope.
* **Role-Based Access Control (RBAC)**: Fine-grained permissions (Admin vs Editor) are not needed for this personal project.
* **Multi-Modal Output Generation**: Generating charts or images in the final answer is not in scope.

---

## 3. High-Level Design

### 3.1 Architecture Overview

```mermaid
graph TB
    subgraph Client["Client Layer"]
        Streamlit["Streamlit UI<br/>(localhost:8501)"]
        cURL["REST Client<br/>(curl / httpie)"]
    end

    subgraph Go["Go Control Plane (Port 8080)"]
        Router["Chi Router"]
        MW["Middleware Stack<br/>RequestID → RealIP →<br/>Logging → Recoverer → CORS"]
        Auth["JWT Auth Middleware"]
        RL["Redis Rate Limiter<br/>(sliding window)"]
        Handlers["Request Handlers<br/>query / ingest / explain /<br/>generate-questions"]
        RouterAgent["Router Agent<br/>(intent routing)"]
        Council["Multi-Agent Council<br/>(Deliberation Pipeline)"]
        Reflect["Self-Reflection Agent<br/>(Revision Loop)"]
        Cache["Exact Key Cache (L1)<br/>(Redis GET 1ms)"]
        SemCache["Semantic Vector Cache (L2)<br/>(Redis Stack RediSearch VSS)"]
        AuditLog["Audit Logger<br/>(structured JSON)"]
        Metrics["Prometheus Metrics<br/>/metrics endpoint"]
    end

    subgraph Python["Python RAG Service (Port 8000)"]
        Inspector["Document Inspector"]
        OCRRouter["Adaptive OCR Router"]
        DirectText["Direct Text Extractor"]
        LayoutOCR["Layout-Aware OCR<br/>(pdfplumber)"]
        Tesseract["Tesseract OCR"]
        Chunker["Layout-Aware Chunker"]
        Embedder["Transformer Embeddings<br/>(BAAI/bge-small-en-v1.5)"]
        ChromaDB["ChromaDB<br/>Vector Store"]
    end

    subgraph LLMs["External & Local LLM Providers"]
        OR["OpenRouter API<br/>(3 council models)"]
        vLLM["Local vLLM Model serving<br/>(microsoft/Phi-4-mini-instruct)"]
        Gemini["Google Gemini API<br/>(chairman model)"]
    end

    subgraph Infra["Infrastructure"]
        Redis[("Redis Stack Server<br/>(VSS + cache + rate limit)")]
        Prom["Prometheus<br/>(Port 9091)"]
        Grafana["Grafana<br/>(Port 3000)"]
    end

    Streamlit --> Router
    cURL --> Router
    Router --> MW --> Auth --> RL --> Handlers
    Handlers -->|1. Exact Match Check (0ms embedding)| Cache
    Cache -->|miss| SemCache
    SemCache -->|2. Vector VSS Check (miss)| RouterAgent
    RouterAgent -->|direct mode| Gemini
    RouterAgent -->|council mode| Council
    Council -->|retrieve chunks / web search| Python
    Council -->|fan-out 3x| OR & vLLM
    Council -->|synthesize| Gemini
    Gemini -->|deep mode| Reflect
    Reflect -->|needs revision| Gemini
    Handlers --> AuditLog
    Handlers --> Metrics
    RL --> Redis
    Cache --> Redis

    Inspector --> OCRRouter
    OCRRouter --> DirectText
    OCRRouter --> LayoutOCR
    OCRRouter --> Tesseract
    Chunker --> Embedder --> ChromaDB

    Metrics -.->|scrape 15s| Prom
    Prom -.->|dashboards| Grafana
```

### 3.2 Service Boundaries

| Service | Language | Port | Responsibility |
|---------|----------|------|----------------|
| **Go Backend** | Go 1.22 | 8080 | API gateway, auth, caching, LLM orchestration, metrics |
| **Python RAG** | Python 3.11 | 8000 | Document processing, OCR, chunking, embedding, retrieval |
| **Redis** | — | 6379 | Cache (1h TTL) + per-user rate limiting |
| **Prometheus** | — | 9091 | Metrics collection |
| **Grafana** | — | 3000 | Pre-built dashboards |

---

## 4. Detailed Design

### 4.1 Data Flow: Core Query (Cache Miss)

```mermaid
sequenceDiagram
    participant C as Client
    participant G as Go Backend
    participant R as Redis
    participant P as Python RAG
    participant OR as OpenRouter (3 models)
    participant GM as Gemini (Chairman)

    C->>G: POST /api/v1/query {question, doc_id}
    G->>G: Validate JWT → Rate Limit

    G->>R: 1. Try L1 Exact Key Cache (`cache:<doc_id>:<question>`). If hit, return in ~1ms (0ms embedding overhead).
    G->>P: 2. On L1 miss, call Python RAG `/embed` to fetch a 384-dimensional query vector.
    G->>R: 3. Try L2 Semantic Vector Cache (`RedisSemanticCache` RediSearch VSS). If similarity >= 0.85, return hit.
    R-->>G: L1 & L2 MISS

    alt doc_id provided
        G->>P: POST /retrieve {question, doc_id, top_k=5}
        P->>P: Embed query → ChromaDB similarity search
        P-->>G: Top-K chunks
    else doc_id omitted
        G->>G: Web search context
    end

    Note over G,OR: Stage 1 — Fan-Out (Parallel)
    par 3 concurrent goroutines
        G->>OR: Model 1: Generate(prompt)
        G->>OR: Model 2: Generate(prompt)
        G->>OR: Model 3: Generate(prompt)
    end
    OR-->>G: 3 independent candidate answers

    Note over G,OR: Stage 2 — Peer Review (Parallel)
    par 3 concurrent goroutines
        G->>OR: Review & rank all answers
    end
    OR-->>G: 3 peer reviews with rankings

    Note over G,GM: Stage 3 — Chairman Synthesis
    G->>GM: Synthesize(question, chunks, answers, reviews)
    GM-->>G: {answer, reasoning, confidence, source}

    G->>R: Populate L1 Exact Cache (key) & L2 Semantic Vector Cache (RediSearch VSS, TTL: 24h)
    G-->>C: 200 OK
```

### 4.2 Scalability & Reliability

*   **Go Backend Scaling:** The Go control plane is entirely stateless (auth is JWT-based, cache is in Redis/L1). It can be horizontally scaled behind an Nginx load balancer.
*   **Progressive Degradation:** The orchestrator handles LLM failures dynamically. If 2 out of 3 models fail, it skips peer review. If peer reviews fail, it picks the longest candidate. If Chairman fails, it falls back to the highest peer-reviewed answer.
*   **Security:** Rate limiting via Redis sliding windows protects against abuse. JWT handles stateless auth. All queries are audit logged to structured JSON.

---

## 5. Alternatives Considered

*   **Single Monolithic Python Service vs. Multi-Service:** We considered writing the entire stack in FastAPI (Python). 
    * *Decision:* Rejected. Go provides vastly superior concurrent HTTP handling and goroutine orchestration necessary for the parallel fan-out loops of the multi-agent council. We accepted the ~5ms network overhead between Go and Python to use the best tool for each job (Go for orchestration, Python for ML/RAG).
*   **LangChain Default Splitters vs. Custom Layout-Aware Chunking:** We considered `RecursiveCharacterTextSplitter`. 
    * *Decision:* Rejected. It arbitrarily splits tables and captions, destroying context. We built a custom chunker that preserves semantic structure (tables remain whole, headings attach to body).

---

## 6. Architectural Decision Registry (ADR) & Trade-off Matrix

#### Decision 1: Redis Stack RediSearch VSS vs. In-Process C++ SIMD CGo Cache
* **Context**: Choosing between an in-process C++ AVX2 SIMD cache (`fastcache`) vs. a distributed Redis Stack RediSearch VSS engine (`RedisSemanticCache`).
* **Final Selection**: **Redis Stack Vector Similarity Search (RediSearch VSS)**.
* **Pros**:
  - **Stateless Backend Scaling**: Enables $100\%$ shared cache state across all horizontally scaled Go backend replicas.
  - **Zero CGo Toolchain Friction**: Allows pure Go compilation (`CGO_ENABLED=0`) across x86_64, ARM64, and Alpine containers without `gcc/g++` dependencies.
  - **Restart Persistence & TTL**: Entries survive container restarts via AOF/RDB snapshots with automatic 24-hour TTL expiration.
  - **Zero Go Heap GC Pressure**: Employs zero-copy `unsafe.Slice` float32 byte encoding ($0\text{ bytes}$ allocated).
  - **Vector Search Complexity**: $O(\log N)$ FLAT/HNSW index search natively inside the Redis engine.
* **Cons**:
  - **Network Socket Overhead**: Adds a $\sim 2.3\text{ ms}$ socket roundtrip overhead compared to process RAM lookups ($\sim 1.2\text{ ms}$).

#### Decision 2: Tiered Cache Lookup Order (L1 Exact Match → L2 Semantic Match → L3 LLM Council)
* **Context**: Determining the order of cache evaluation for incoming user queries.
* **Final Selection**: **L1 Exact Key Match (`cache:<doc_id>:<question>`) → L2 Semantic Vector Match (`RedisSemanticCache` VSS) → L3 LLM Council Deliberation**.
* **Pros**:
  - **Instant Exact Hits ($1.0\text{ ms}$)**: Exact duplicate queries bypass Python RAG `/embed` vector calculation entirely, saving $15\text{--}30\text{ ms}$ of network roundtrips.
  - **Resource Optimization**: Zero CPU/GPU tensor inference expended on exact query matches.
* **Cons**:
  - **Dual Schema Management**: Requires maintaining both string key entries and vector index entries in Redis Stack.

#### Decision 3: PyTorch Batch Vectorized Inference in `TransformerEmbeddings`
* **Context**: Optimizing document chunk embedding generation during multi-page ingestion.
* **Final Selection**: **Single-Pass PyTorch Tensor Batch Inference**.
* **Pros**:
  - **Throughput Speedup**: Reduces $N$ separate single-chunk model forward passes down to **1 single batch forward pass** ($>15\times$ embedding speedup).
* **Cons**:
  - **Batch Sequence Padding**: Requires sequence padding shorter chunks to the longest chunk length in the batch.

#### Decision 4: Zero-Fee Local Web Search (DDG + BeautifulSoup) vs. Headless Browser
* **Context**: Grounding off-topic queries locally without external API costs.
* **Final Selection**: **DuckDuckGo Search + BeautifulSoup4 HTML Parsing**.
* **Pros**:
  - **Lightweight Footprint**: Eliminates $>500\text{ MB}$ Chromium browser container binaries.
  - **Low Latency**: Parses HTML text in milliseconds without rendering DOM assets or JavaScript.
* **Cons**:
  - **Anti-Bot Susceptibility**: Vulnerable to layout changes or rate limits from search providers.

#### Gap 1: In-Memory User Ephemerality
* **Description**: User accounts are stored in-memory in `auth.go`. A server restart wipes the map, forcing users to sign up again.
* **Solution**: (Planned) Refactor to SQLite/Redis Hash persistence.

#### Gap 2: CGo Toolchain & AVX2 Lock-in [SUPERSEDED]
* **Description**: CGo bindings and AVX2 intrinsics created cross-compilation friction on ARM64 / Apple Silicon.
* **Solution**: Completely removed CGo fastcache and migrated to pure Go (`CGO_ENABLED=0`) with Redis Stack RediSearch VSS and zero-copy `unsafe.Slice` float32 vector serialization.

#### Gap 3: Path Traversal Vulnerabilities in Document Ingestion [RESOLVED]
* **Description**: Python RAG `/ingest` route handles raw filenames directly when generating `doc_id`.
* **Solution**: Added rigorous regex scrubbers inside `ingest.py` before building `doc_id`.

#### Gap 4: Missing Generated `doc_id` in Go Audit Logging [RESOLVED]
* **Description**: Go backend logged an empty string in `h.Audit.LogIngest`.
* **Solution**: Unmarshaled the `/ingest` response in `ingest.go` to capture the generated `doc_id`.

#### Gap 5: FIFO Eviction vs. True LRU Caching [RESOLVED]
* **Description**: `SemanticCache` behaved conceptually as a FIFO cache.
* **Solution**: Upgraded `Get` lock scopes to `std::unique_lock` on hits to safely promote accessed keys to the front of `lru_list_`.

### 6.1 Engineering Gaps & Solutions

**1. Go/Python Context Propagation**
* **Gap**: Currently, context cancellation in the Go backend (e.g. client disconnect) is not fully propagated to the Python RAG service during long ingestion or retrieval tasks.
* **Solution**: Implement context-aware HTTP requests in Go using `req.WithContext(ctx)` when calling the Python service. The Python service (using FastAPI/Starlette) should listen for client disconnects via `request.is_disconnected()`.
* **Pros**: Prevents dangling resource locks and wasted GPU/CPU cycles on orphaned requests.
* **Cons**: Requires rewriting Python endpoints to support async polling on `request.is_disconnected()`.

**2. Database Persistence for Users and Logs**
* **Gap**: Users are stored in an ephemeral in-memory map. Audit logs are written to flat JSON files. 
* **Solution**: Introduce SQLite or PostgreSQL for relational persistence of users, and ship logs to a robust TSDB or structured logging aggregator like Loki.
* **Pros**: Enables persistent accounts across restarts and scalable analytics.
* **Cons**: Increases deployment complexity and requires database migrations.

### 6.2 Conceptual Gaps & Solutions

**1. Document Summary-Aware Routing**
* **Gap**: The Router agent was previously blind to document content. It routed queries to the LLM Council even when the user asked entirely unrelated questions (e.g. "What is the capital of France?" while a highly technical PDF was attached), wasting expensive RAG retrieval and context window.
* **Solution**: Implemented BME (Beginning-Middle-End) document sampling in the Python RAG to generate a concise representation of the document upon ingestion. The `IngestAgent` in Go generates a high-level summary, which is injected into the Router's prompt for query classification.
* **Pros**: Massive cost savings and latency reduction by dynamically skipping RAG and falling back to 'Direct' mode for off-topic queries.
* **Cons**: The BME summary adds latency during document ingestion (one-time cost) and may miss niche topics hidden deep in large documents.

**2. Metric Evaluation of Council Consensus**
* **Gap**: The system lacks an automated, scalable way to verify the objective accuracy of the Council's synthesized output against a known ground truth.
* **Solution**: Implemented a canonical benchmark suite (`tests/bench_semantic_accuracy.py`) to systematically test accuracy and caching latency.
* **Pros**: Creates an empirical feedback loop for prompt engineering and model selection.
* **Cons**: Maintaining a high-quality, ground-truth benchmark dataset requires continuous manual curation.
