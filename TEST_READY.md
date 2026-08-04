# CouncilAI Test Readiness & Execution Summary (TEST_READY.md)

**Version:** 3.0.0  
**Last Verified:** August 2026  
**Execution Environment:** Go 1.22+, Python 3.11+, Redis Stack 7.2+, k6 v0.49+

---

## 1. Primary Test Runner Commands

### 1.1 Unified Test & Report Script (Core Quality Gate)
Executes all Go unit tests and Python RAG unit tests, outputting structured JSON and Markdown summary reports to `tests/reports/`:
```bash
./scripts/run_tests_with_reports.sh
```

### 1.2 Go Backend Unit & Integration Tests
```bash
# Run all Go package unit tests with JSON report output
cd services/go-backend && go test -v -json ./... > "../../tests/reports/go_test_results.json"

# Run tests with code coverage profiling
cd services/go-backend && go test -v -coverprofile=coverage.out ./...
go tool cover -html=services/go-backend/coverage.out -o tests/reports/go_coverage.html
```

### 1.3 Python RAG Service Unit Tests
```bash
# Run Python unit tests for chunking, OCR routing, and RAG logic
PYTHONPATH=services/python-rag python3 -m unittest discover services/python-rag/tests/ > tests/reports/python_test_results.txt 2>&1

# Run with pytest and coverage reporting
PYTHONPATH=services/python-rag pytest services/python-rag/tests/ --cov=services/python-rag --cov-report=html:tests/reports/python_coverage_html
```

### 1.4 Benchmark Execution Commands
```bash
# Run layout-aware document chunking benchmark
python3 tests/benchmarks/bench_chunking.py

# Run L1 Redis KV vs L2 RediSearch VSS caching benchmark
python3 tests/benchmarks/bench_caching_layer.py

# Run Go memory-zero-allocation float32 vector serialization benchmark
cd services/go-backend && go test -bench=BenchmarkSemanticCache -benchmem ./internal/cache/...
```

### 1.5 k6 Load & Concurrency Test Runner
```bash
# Execute end-to-end API load test (simulates 50 virtual users over 2 minutes)
k6 run tests/load/load_test.js --summary-export=tests/reports/k6_load_test_summary.json
```

### 1.6 Evaluation Test Suite Runner Commands (Future Test Suites)
```bash
# Run End-to-End Workflow Test Suite across Tiers 1-4
pytest tests/e2e/test_e2e_workflows.py -v --json-report --json-report-file=tests/reports/e2e_report.json

# Run Chairman Hallucination & Factuality Evaluation Suite
python3 tests/eval/eval_hallucination.py --dataset=tests/fixtures/hallucination_eval_dataset.json --report=tests/reports/hallucination_eval_report.json

# Run ChromaDB Vector Retrieval Recall & Metric Benchmarks
python3 tests/benchmarks/bench_retrieval_recall.py --top-k=5 --dataset=tests/fixtures/retrieval_benchmark.json --report=tests/reports/retrieval_recall_report.json
```

---

## 2. Four-Tier Coverage Summary Matrix

| Test Tier | Target Coverage | Planned / Executed Tests | Readiness Status | Automated Runner Command | Focus & Primary Assertions |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Tier 1: Feature Coverage** | 100% (8 API features) | 40 Unit & E2E Tests ($\ge 5$/feature) | **READY & PASSING** | `./scripts/run_tests_with_reports.sh` | Verifies basic path resolution, JSON payload unmarshaling, status code compliance, JWT claims parsing, standard ingestion, query response formats, explanation generation, MCQ creation, session clearing, health and Prometheus metrics. |
| **Tier 2: Boundary & Corner Cases** | 90% Edge Coverage | 40 Edge Case Scenarios ($\ge 5$/feature) | **SPECIFIED (M3)** | `pytest tests/e2e/test_e2e_workflows.py -k "Tier2"` | Validates zero-byte uploads, corrupt PDF handling, expired/invalid JWT tokens, out-of-bounds `top_k`/`num_questions`, non-existent document IDs, prompt injection refutation, unicode encoding, XSS injection, and error recovery. |
| **Tier 3: Cross-Feature Combinations** | 85% Integration Paths | 10 State & Concurrency Tests | **SPECIFIED (M3)** | `pytest tests/e2e/test_e2e_workflows.py -k "Tier3"` | Validates multi-session conversation isolation, session purging followed by fresh Q&A turns, sequential ingest-explain-quiz workflows, cache hit progression (L1 $\rightarrow$ L2 $\rightarrow$ LLM), cache invalidation on re-ingestion, and concurrent load resilience. |
| **Tier 4: Real-World Application Scenarios** | 100% Core Scenarios | 3 Deep E2E Journeys | **SPECIFIED (M3)** | `pytest tests/e2e/test_e2e_workflows.py -k "Tier4"` | Simulates full end-to-end user journeys: Financial 10-K report audit, multi-page technical architecture analysis, and regulatory contract compliance review with false-premise refutation. |

---

## 3. Feature Readiness & Test Infrastructure Checklist

### Feature 1: Authentication & Authorization (F1)
- [x] JWT Token Generation (`/api/v1/register`, `/api/v1/login`) — *Unit Tested (`jwt_test.go`)*
- [x] JWT Middleware Claim Validation (`Authorization: Bearer <token>`) — *Unit Tested (`handlers_test.go`)*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`handlers_test.go`)*
- [ ] Tier 2 Boundary Tests (Expired tokens, tampered signatures, empty bodies) — *Specified (M3 Specification)*
- [ ] User Persistence Store (Redis Hash / Relational DB Migration) — *Planned (TODO.md)*

### Feature 2: Document Ingestion & Indexing (F2)
- [x] PyPDF Direct Text Extraction Routing — *Unit Tested (`test_chunking.py`)*
- [x] pdfplumber Layout-Aware Table Extraction — *Unit Tested (`test_chunking.py`)*
- [x] PyTorch Batch Vectorization (`BAAI/bge-small-en-v1.5`) — *Unit Tested (`test_rag.py`)*
- [x] Path Traversal Filename Sanitization — *Unit & Integration Tested*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`test_chunking.py`)*
- [ ] Tier 2 Boundary Tests (Zero-byte files, corrupt PDFs, oversized files, encrypted PDFs) — *Specified (M3 Specification)*

### Feature 3: Grounded Deliberation Query (F3)
- [x] OpenRouter 3x Parallel Council Fan-Out Loop — *Unit Tested (`handlers_test.go`)*
- [x] Parallel Peer Review Cross-Evaluation Matrix — *Unit Tested (`handlers_test.go`)*
- [x] Gemini Chairman Consensus Synthesis — *Unit Tested (`handlers_test.go`)*
- [x] Self-Reflection Quality Audit Loop — *Unit Tested (`handlers_test.go`)*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`handlers_test.go`)*
- [ ] Tier 2 Boundary Tests (Non-existent doc IDs, empty questions, prompt injection, top_k bounds) — *Specified (M3 Specification)*
- [ ] Chairman Hallucination Evaluation Suite (Factuality, Faithfulness, CCI, CP) — *Specified (M3 Specification)*
- [ ] Adversarial Prompt & False Premise Benchmark Dataset — *Specified (M3 Specification)*

### Feature 4: General Knowledge Q&A & Web Search (F4)
- [x] Query Intent Classification & Router Routing — *Unit Tested (`handlers_test.go`)*
- [x] DuckDuckGo Real-Time Web Search Integration — *Unit Tested (`handlers_test.go`)*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`handlers_test.go`)*
- [ ] Tier 2 Boundary Tests (Search engine timeouts, nonsensical inputs, unicode scripts) — *Specified (M3 Specification)*

### Feature 5: Adaptive Document Explanation (F5)
- [x] Adaptive Explanation Generation (`/api/v1/explain`) — *Unit Tested (`handlers_test.go`)*
- [x] Knowledge Level Parameters (`beginner`, `intermediate`, `advanced`) — *Unit Tested (`handlers_test.go`)*
- [x] Parameter Alias Support (`level` / `knowledge_level`) — *Unit Tested (`handlers_test.go`)*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`handlers_test.go`)*
- [ ] Tier 2 Boundary Tests (Invalid level names, empty focus topic lists, non-existent doc IDs) — *Specified (M3 Specification)*

### Feature 6: Assessment Question Generation (F6)
- [x] Assessment Question Generation (`/api/v1/generate-questions`) — *Unit Tested (`handlers_test.go`)*
- [x] MCQ & Subjective Question Formatting — *Unit Tested (`handlers_test.go`)*
- [x] Parameter Alias Support (`count` / `num_questions`) — *Unit Tested (`handlers_test.go`)*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`handlers_test.go`)*
- [ ] Tier 2 Boundary Tests (Out-of-bounds question counts, invalid bloom levels, short context docs) — *Specified (M3 Specification)*

### Feature 7: Session History & State Clear (F7)
- [x] Multi-Turn Session Clear (`/api/v1/conversation/clear`) — *Unit Tested (`handlers_test.go`)*
- [x] Redis ConversationStore Context Storage — *Unit Tested (`handlers_test.go`)*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`handlers_test.go`)*
- [ ] Tier 2 Boundary Tests (Empty session IDs, long key strings, concurrent clear calls) — *Specified (M3 Specification)*

### Feature 8: Observability, Health & Metrics (F8)
- [x] System Health Check (`GET /health`) — *Unit Tested (`handlers_test.go`)*
- [x] Prometheus Metrics Collector (`GET /metrics`) — *Unit Tested (`handlers_test.go`)*
- [x] Structured JSON Audit Logging — *Unit & Integration Tested*
- [x] Tier 1 Feature Sanity Tests (5/5 cases) — *Verified (`handlers_test.go`)*
- [ ] Tier 2 Boundary Tests (Redis failure health status, Python RAG down status, metrics high load) — *Specified (M3 Specification)*

### Multi-Tier Caching Infrastructure
- [x] L1 Exact Key-Value Redis Cache (`cache:<doc_id>:<question>`) — *Unit Tested (`semantic_redis_test.go`)*
- [x] L2 Semantic RediSearch Vector Similarity Cache (`RedisSemanticCache`) — *Unit & Benchmark Tested*
- [x] Zero-Allocation Float32 Byte Casting (`unsafe.Slice`) — *Benchmark Tested (`semantic_redis_bench_test.go`)*

### Vector Retrieval Infrastructure (ChromaDB)
- [x] ChromaDB Collection Storage & Similarity Search — *Unit Tested (`test_rag.py`)*
- [x] Top-K Relevant Chunk Retrieval — *Unit Tested (`test_rag.py`)*
- [ ] ChromaDB Retrieval Recall Benchmark Suite (Recall@K, MRR@K, NDCG@K) — *Specified (M3 Specification)*
- [ ] Distance Metric Benchmarking (Cosine vs L2 vs IP) — *Specified (M3 Specification)*
- [ ] Chunk Overlap Resilience Evaluation — *Specified (M3 Specification)*

---
*Summary generated for CouncilAI Milestone 3 Execution Validation.*
