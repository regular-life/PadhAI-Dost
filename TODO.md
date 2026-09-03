# CouncilAI - Future Work & TODOs

This document tracks known engineering gaps, technical debt, and future feature requests for the CouncilAI platform.

## Go Backend

### Authentication & Persistence
- [x] `auth.go`: Persist users in PostgreSQL with connection pool, schema initialization, and bcrypt hashing (`PostgresUserRepository`).
- [ ] `auth.go`: Track failed login attempts in cache to throttle brute-force attacks.
- [ ] `auth.go`: Implement password complexity constraints and email verification routes.

### State & Conversation Management
- [ ] `conversation.go`: Support selective purging of individual turns or scaling limits by message ID.

### Document & RAG Operations
- [ ] `explain.go`: Implement cache pre-warming logic for newly ingested documents.
- [ ] `ingest.go`: Support multi-part file uploads of other formats (docx, txt).
- [ ] `ingest.go`: Return detailed status report on pipeline component health.

### Query Delivery & Analytics
- [x] `query.go`: Implement streaming delivery to the client via Server-Sent Events (`Accept: text/event-stream`).
- [ ] `questions.go`: Parse generated options and format them as structured JSON quizzes.

### Configuration & Tooling
- [ ] `config.go`: Support custom config paths via a CLI flag (e.g., `-config=/path/to/config.yaml`).
- [ ] `factory.go`: Support dynamic custom timeout overrides per model in ProviderURLs.

## Python RAG Service

### Server & Config
- [ ] `config.py`: Implement dynamic validation and parsing of custom config files.
- [ ] `main.py`: Configure dynamic CORS whitelist mapping from config.yaml.

### OCR & Parsing Operations
- [ ] `layout_aware.py`: Parallelize page extraction loop for large PDF documents.
- [ ] `layout_aware.py`: Use intersection over area ratio instead of simple coordinate bounds overlap check.
- [ ] `router.py`: Implement page character-density threshold checks before direct routing.
- [ ] `tesseract.py`: Support localized language codes dynamically for pytesseract config.
- [ ] `tesseract.py`: Parallelize page rendering and processing to optimize latency.

### Retrieval & Re-ranking
- [ ] `chroma_store.py`: Implement asynchronous vector insertions.
- [ ] `chroma_store.py`: Migrate from in-memory fetch to database cursor streams for very large docs.
- [ ] `reranker.py`: Quantize model to ONNX format to improve CPU/GPU inference latency.
- [ ] `reranker.py`: Support batch-wise prediction for very large document sets.
