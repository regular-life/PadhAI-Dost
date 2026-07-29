# CouncilAI REST API Reference

The CouncilAI API is served from the Go control plane (default port `8080`). All Q&A and document operations require authentication via a JWT token.

## Authentication

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
* **Response:**
  ```json
  {
    "token": "eyJhbGciOiJIUzI1NiIs..."
  }
  ```

### `POST /api/v1/register`
Create a new account.
* **Auth Required:** No
* **Body:**
  ```json
  {
    "username": "demo",
    "password": "demo123"
  }
  ```

---

## Document & Q&A Operations

### `POST /api/v1/ingest`
Upload a document (PDF) for parsing, OCR, and embedding.
* **Auth Required:** Yes
* **Format:** `multipart/form-data`
* **Fields:**
  * `file`: The document file (e.g., `@your_document.pdf`).
  * `doc_id` (optional): A custom ID for this document. If omitted, one is generated automatically.
* **cURL Example:**
  ```bash
  curl http://localhost:8080/api/v1/ingest \
    -H "Authorization: Bearer $TOKEN" \
    -F "file=@your_document.pdf"
  ```

### `POST /api/v1/query`
Ask a question. If `doc_id` is provided, the answer is grounded in the document. If omitted, the answer is grounded via real-time Web Search.
* **Auth Required:** Yes
* **Body:**
  ```json
  {
    "question": "What are the key concepts?",
    "doc_id": "your_doc_id",
    "top_k": 5,
    "session_id": "session_123"
  }
  ```
* **cURL Example:**
  ```bash
  curl http://localhost:8080/api/v1/query \
    -H "Authorization: Bearer $TOKEN" \
    -H "Content-Type: application/json" \
    -d '{"question":"What are the key concepts?","doc_id":"your_doc_id"}'
  ```

### `POST /api/v1/explain`
Generate a document explanation targeted at a specific proficiency level (beginner, intermediate, advanced).
* **Auth Required:** Yes
* **Body:**
  ```json
  {
    "doc_id": "your_doc_id",
    "level": "intermediate"
  }
  ```

### `POST /api/v1/generate-questions`
Generate assessment questions (MCQ or subjective) based on document context.
* **Auth Required:** Yes
* **Body:**
  ```json
  {
    "doc_id": "your_doc_id",
    "question_type": "mcq",
    "count": 5
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

---

## System Health & Metrics

### `GET /health`
Returns the operational health of the Go backend, Redis cache, and Python RAG service.
* **Auth Required:** No

### `GET /metrics`
Exposes Prometheus-compatible metrics (`councilai_request_count_total`, `councilai_cache_operations_total`, etc.).
* **Auth Required:** No
