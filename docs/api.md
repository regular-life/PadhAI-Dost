# CouncilAI REST API Reference

The CouncilAI API is served from the Go control plane (default port `8080`). All Q&A and document operations require authentication via a JWT token.

---

## Authentication

### `POST /api/v1/register`
Create a new user account and receive an initial JWT token.
* **Auth Required:** No
* **Body:**
  ```json
  {
    "username": "demo",
    "password": "demo123"
  }
  ```
* **Response (201 Created):**
  ```json
  {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user_id": "demo",
    "message": "user created successfully"
  }
  ```

### `POST /api/v1/login`
Retrieve a JWT token for authenticated operations.
* **Auth Required:** No
* **Body:**
  ```json
  {
    "username": "demo",
    "password": "demo123"
  }
  ```
* **Response (200 OK):**
  ```json
  {
    "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
    "user_id": "demo"
  }
  ```

---

## Document & Q&A Operations

### `POST /api/v1/ingest`
Upload a document (PDF) for parsing, OCR routing, layout-aware chunking, and vector embedding.
* **Auth Required:** Yes
* **Format:** `multipart/form-data`
* **Fields:**
  * `file` (required): The document file binary (e.g., `@your_document.pdf`).
  * `doc_id` (optional): A custom ID for this document. If omitted, one is generated automatically.
* **cURL Example:**
  ```bash
  curl http://localhost:8080/api/v1/ingest \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@your_document.pdf"
  ```
* **Response (200 OK):**
  ```json
  {
    "doc_id": "your_document.pdf_a1b2c3d4e5f6",
    "chunk_count": 12,
    "metadata": {
      "has_text_layer": true,
      "is_scanned": false,
      "has_tables": true,
      "is_multicolumn": false,
      "page_count": 5,
      "file_type": "application/pdf",
      "file_name": "your_document.pdf"
    },
    "preview_text": "Executive Summary: This document presents...",
    "message": "Successfully ingested 12 chunks using PyPDF OCR"
  }
  ```

### `POST /api/v1/query`
Ask a question. If `doc_id` is provided, the answer is grounded in the document context. If `doc_id` is omitted, the query defaults to general knowledge with DuckDuckGo real-time Web Search grounding.

Supports both **Server-Sent Events (SSE) Streaming** and standard **Synchronous JSON** responses via HTTP content negotiation (`Accept` header).

* **Auth Required:** Yes
* **Headers:**
  * `Authorization: Bearer <token>` (Required)
  * `Content-Type: application/json` (Required)
  * `Accept: text/event-stream` (Optional, activates real-time SSE streaming)
  * `Accept: application/json` (Optional / default, returns monolithic JSON response)
* **Body:**
  ```json
  {
    "question": "What are the core architectural trade-offs?",
    "doc_id": "your_document.pdf_a1b2c3d4e5f6",
    "top_k": 5,
    "session_id": "session_123"
  }
  ```
  * Note: `doc_id` (optional), `top_k` (optional integer, default `5`), `session_id` (optional string).

#### Mode 1: Server-Sent Events (SSE) Streaming (`Accept: text/event-stream`)

When `Accept: text/event-stream` is requested, the server immediately flushes HTTP 200 headers and progressively streams deliberation lifecycle frames.

* **Response Headers:**
  * `Content-Type: text/event-stream; charset=utf-8`
  * `Cache-Control: no-cache`
  * `Connection: keep-alive`
  * `X-Accel-Buffering: no`
* **cURL Example:**
  ```bash
  curl -N http://localhost:8080/api/v1/query \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -H "Accept: text/event-stream" \
    -d '{"question":"What are the core architectural trade-offs?","doc_id":"your_doc_id"}'
  ```
* **Event Frames Lifecycle:**

1. **`event: candidate_draft`** (Stage 1 Fan-Out, emitted asynchronously per model with TTFT < 1.5s):
   ```http
   event: candidate_draft
   data: {"index":0,"model":"openrouter:anthropic/claude-3.5-sonnet","model_name":"openrouter:anthropic/claude-3.5-sonnet","answer":"Draft candidate response text...","content":"Draft candidate response text...","latency_ms":450}

   ```

2. **`event: peer_review`** (Stage 2 Peer Review, emitted asynchronously per reviewer):
   ```http
   event: peer_review
   data: {"index":0,"reviewer":"openrouter:openai/gpt-4o","review":"RANKING: A, B\nREASONING: Response A provides superior depth...","critique":"Response A provides superior depth...","ranking":["A","B"],"scores":{"A":2,"B":1},"latency_ms":520}

   ```

3. **`event: final_answer`** (Stage 3 Chairman Consensus / Cache Hit):
   ```http
   event: final_answer
   data: {"answer":"The core architectural trade-offs involve Redis Stack RediSearch VSS...","confidence":0.95,"source":"chairman:gemini-2.0-flash","strategy":"council","reasoning":"Synthesized from Claude and GPT-4o with peer review weighting.","peer_reviewed":true,"reflection":null,"candidates":[{"model":"openrouter:anthropic/claude-3.5-sonnet","answer":"..."},{"model":"openrouter:openai/gpt-4o","answer":"..."}],"latency":"1.120s","cache_hit":false}

   ```

4. **`event: error`** (Fatal error during stream):
   ```http
   event: error
   data: {"code":500,"message":"LLM council failed","error":"all council members failed to respond"}

   ```

#### Mode 2: Synchronous JSON Response (`Accept: application/json` or omitted)

* **cURL Example:**
  ```bash
  curl http://localhost:8080/api/v1/query \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"question":"What are the core architectural trade-offs?","doc_id":"your_doc_id"}'
  ```
* **Response (200 OK, `Content-Type: application/json`):**
  ```json
  {
    "answer": "The core architectural trade-offs involve Redis Stack RediSearch VSS for vector similarity caching...",
    "confidence": 0.92,
    "source": "council",
    "strategy": "council",
    "reasoning": "Synthesized consensus from 3 independent LLM council members with peer review evaluation.",
    "peer_reviewed": true,
    "reflection": {
      "approved": true,
      "critique": "Well supported by source text.",
      "revised_answer": ""
    },
    "candidates": [
      {
        "model": "anthropic/claude-3.5-sonnet",
        "answer": "Candidate response...",
        "score": 0.95
      }
    ],
    "latency": "1.234s",
    "cache_hit": false
  }
  ```

### `POST /api/v1/explain`
Generate a structured document explanation tailored to a specific knowledge level and depth.
* **Auth Required:** Yes
* **Body:**
  ```json
  {
    "doc_id": "your_document.pdf_a1b2c3d4e5f6",
    "knowledge_level": "intermediate",
    "level": "intermediate",
    "depth": "section-wise",
    "focus_topics": ["architecture", "caching"]
  }
  ```
  * Note: Accepts either `"knowledge_level"` or `"level"` (both are supported). Values: `"beginner"`, `"intermediate"`, `"advanced"` (default: `"beginner"`).
  * Optional: `"depth"` (`"brief"`, `"section-wise"`, `"detailed"`, default: `"section-wise"`), `"focus_topics"` (array of strings).
* **Response (200 OK):**
  ```json
  {
    "explanation": "This document outlines the core architecture of CouncilAI...",
    "sections": [
      {
        "heading": "1. Overview",
        "content": "The system utilizes a multi-agent deliberation framework...",
        "page_refs": [1, 2]
      }
    ],
    "confidence": 0.88,
    "source": "council",
    "latency": "850ms",
    "cache_hit": false
  }
  ```

### `POST /api/v1/generate-questions`
Generate assessment questions (MCQ or subjective) from document context.
* **Auth Required:** Yes
* **Body:**
  ```json
  {
    "doc_id": "your_document.pdf_a1b2c3d4e5f6",
    "num_questions": 5,
    "count": 5,
    "difficulty": 5,
    "question_type": "mcq",
    "bloom_level": "analysis"
  }
  ```
  * Note: Accepts either `"num_questions"` or `"count"` (both are supported, integer 1-20, default: `5`).
  * Optional: `"difficulty"` (integer 1-10, default: `5`), `"question_type"` (`"mcq"` or `"subjective"`, default: `"subjective"`), `"bloom_level"` (string).
* **Response (200 OK):**
  ```json
  {
    "questions": [
      {
        "question": "What primary database is used for semantic caching?",
        "answer": "A) Redis Stack RediSearch VSS",
        "explanation": "Redis Stack provides native vector similarity search via RediSearch VSS.",
        "source_chunk": "Section 3.1 describes Redis Stack Server for VSS...",
        "options": [
          "A) Redis Stack RediSearch VSS",
          "B) PostgreSQL pgvector",
          "C) Pinecone",
          "D) MongoDB"
        ]
      }
    ],
    "raw_output": "[{\n  \"question\": ...\n}]",
    "confidence": 0.90,
    "source": "council",
    "latency": "1.120s",
    "cache_hit": false
  }
  ```

### `POST /api/v1/conversation/clear`
Clear multi-turn conversation history for a given session.
* **Auth Required:** Yes
* **Body:**
  ```json
  {
    "session_id": "session_123"
  }
  ```
* **Response (200 OK):**
  ```json
  {
    "status": "cleared"
  }
  ```

---

## System Health & Metrics

### `GET /health`
Returns operational health status of the Go control plane, Redis cache, and Python RAG service.
* **Auth Required:** No
* **Response (200 OK):**
  ```json
  {
    "status": "healthy",
    "service": "go-backend",
    "version": "2.0.0",
    "redis": "healthy",
    "rag_service": "healthy"
  }
  ```

### `GET /metrics`
Exposes Prometheus-compatible metrics (`councilai_request_count_total`, `councilai_cache_operations_total`, etc.).
* **Auth Required:** No
* **Response (200 OK):** Text/plain Prometheus metrics format.
