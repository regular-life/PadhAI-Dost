# CouncilAI

![Go Version](https://img.shields.io/badge/Go-1.22+-00ADD8?style=flat&logo=go)
![Python Version](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat&logo=python)
![Docker](https://img.shields.io/badge/Docker-Supported-2496ED?style=flat&logo=docker)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

CouncilAI is an open-source, multi-agent document deliberation and Q&A engine. 
It replaces single-model inference with an ensemble "council-of-agents" architecture to process documents and generate reliable, cross-reviewed answers.

## Architecture

CouncilAI coordinates a multi-agent workflow:
1. **Router Agent**: Analyzes user queries (and optional document summaries) using BME (Beginning-Middle-End) sampling to route requests dynamically to Direct Mode, the LLM Council, or Web Search fallback.
2. **Council Member Agents**: Concurrent multi-model agents that generate candidate answers in parallel.
3. **Peer-Review Loop**: Agents cross-evaluate and rank candidate responses.
4. **Chairman Agent**: Synthesizes the final consensus response, including confidence scores.
5. **Self-Reflection Agent**: Audits the result for quality and triggers a revision loop if needed.

The backend infrastructure relies on a Go control plane for parallel orchestration and a Python microservice for RAG operations (OCR, chunking, and embedding). It utilizes a high-performance Redis Stack Semantic Cache (RediSearch Vector Similarity Search) to bypass redundant LLM calls.

## Quick Start

### Prerequisites
- Docker and Docker Compose
- An OpenRouter API key or local vLLM endpoint configuration (configured in `.env`)

### Setup

```bash
git clone https://github.com/regular-life/CouncilAI
cd CouncilAI
./setup.sh
docker compose up --build -d
```

- **UI**: `http://localhost:8501` (Default login: `demo` / `demo123`)
- **API**: `http://localhost:8080`

For comprehensive API endpoints and authentication patterns, refer to the [REST API Reference](docs/api.md).

## Documentation

- **[System Design](docs/DESIGN_DOC.md)**: Architectural overview, data flow, and components.
- **[API Reference](docs/api.md)**: REST endpoints and payloads.

## Testing and Benchmarks

CouncilAI uses a multi-tier testing architecture separating service-owned unit tests from repository-wide load and benchmark tests:

```bash
# Run all Go and Python unit tests (<1s execution)
./scripts/run_tests_with_reports.sh

# Run k6 Virtual User (VU) load test
k6 run tests/load/load_test.js

# Run RAG document chunking benchmark
python3 tests/benchmarks/bench_chunking.py
```

## Contributing

This is a personal open-source project. Bug reports, feature suggestions, and pull requests are welcome.

---
*CouncilAI was originally built under the legacy name "PadhAI Dost" ("Study Friend" in Hindi).*
