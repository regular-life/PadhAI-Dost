package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"strings"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/agent"
	"github.com/regular-life/CouncilAI/go-backend/internal/audit"
	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/memory"
)

// Handlers encapsulates all system dependencies for the REST API endpoints.
type Handlers struct {
	RAGServiceURL          string
	Council                *council.Orchestrator
	Cache                  *cache.RedisCache
	SemanticCache          cache.SemanticCache
	Audit                  *audit.Logger
	Router                 *agent.Router
	IngestAgent            *agent.IngestAgent
	Memory                 *memory.ConversationStore
	HTTPClient             *http.Client
	SemanticCacheThreshold float32
}

// NewHandlers creates a Handlers instance with fully injected dependencies.
// Pools HTTP connections and customizes transport parameters to optimize socket reuse.
func NewHandlers(
	ragURL string,
	council *council.Orchestrator,
	redisCache *cache.RedisCache,
	semCache cache.SemanticCache,
	auditLogger *audit.Logger,
	router *agent.Router,
	ingestAgent *agent.IngestAgent,
	memoryStore *memory.ConversationStore,
	semanticCacheThreshold float32,
) *Handlers {
	transport := &http.Transport{
		Proxy: http.ProxyFromEnvironment,
		DialContext: (&net.Dialer{
			Timeout:   30 * time.Second,
			KeepAlive: 30 * time.Second,
		}).DialContext,
		MaxIdleConns:          100,
		MaxIdleConnsPerHost:   100,
		IdleConnTimeout:       90 * time.Second,
		TLSHandshakeTimeout:   10 * time.Second,
		ExpectContinueTimeout: 1 * time.Second,
	}

	return &Handlers{
		RAGServiceURL:          ragURL,
		Council:                council,
		Cache:                  redisCache,
		SemanticCache:          semCache,
		Audit:                  auditLogger,
		Router:                 router,
		IngestAgent:            ingestAgent,
		Memory:                 memoryStore,
		HTTPClient:             &http.Client{Timeout: 120 * time.Second, Transport: transport},
		SemanticCacheThreshold: semanticCacheThreshold,
	}
}

// retrieveAllChunks retrieves every text chunk for a doc_id from the RAG service.
func (h *Handlers) retrieveAllChunks(docID string) ([]string, error) {
	reqBody := map[string]string{"doc_id": docID}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	resp, err := h.HTTPClient.Post(h.RAGServiceURL+"/retrieve-all", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("retrieve-all request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("retrieve-all failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Chunks []struct {
			Content string `json:"content"`
		} `json:"chunks"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	chunks := make([]string, len(result.Chunks))
	for i, c := range result.Chunks {
		chunks[i] = c.Content
	}
	return chunks, nil
}

// retrieveChunks retrieves the top-K relevant chunks for a question.
func (h *Handlers) retrieveChunks(r *http.Request, req QueryRequest) ([]string, error) {
	reqBody := map[string]interface{}{
		"question": req.Question,
		"doc_id":   req.DocID,
		"top_k":    req.TopK,
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", h.RAGServiceURL+"/retrieve", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := h.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("retrieval request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("retrieval failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Chunks []struct {
			Content string `json:"content"`
		} `json:"chunks"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	chunks := make([]string, len(result.Chunks))
	for i, c := range result.Chunks {
		chunks[i] = c.Content
	}
	return chunks, nil
}

// getEmbedding generates a vector embedding for the input text from the RAG service.
func (h *Handlers) getEmbedding(ctx context.Context, text string) ([]float32, error) {
	reqBody := map[string]string{"text": text}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", h.RAGServiceURL+"/embed", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	resp, err := h.HTTPClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("embed request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("embed failed (status %d): %s", resp.StatusCode, string(body))
	}

	var result struct {
		Embedding []float32 `json:"embedding"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}
	return result.Embedding, nil
}

// extractQuestionsJSON extracts a clean JSON array from raw LLM responses.
func extractQuestionsJSON(s string) string {
	s = strings.TrimSpace(s)
	if strings.HasPrefix(s, "```json") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	} else if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	}
	start := strings.Index(s, "[")
	end := strings.LastIndex(s, "]")
	if start >= 0 && end > start {
		s = s[start : end+1]
	}
	s = stripControlChars(s)
	s = sanitizeJSONBackslashes(s)
	return s
}

// stripControlChars strips non-printable ASCII control characters.
func stripControlChars(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		if r >= 0x20 || r == '\t' || r == '\n' || r == '\r' {
			b.WriteRune(r)
		}
	}
	return b.String()
}

// sanitizeJSONBackslashes double-escapes invalid JSON backslashes.
func sanitizeJSONBackslashes(s string) string {
	var result strings.Builder
	result.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			next := s[i+1]
			if next == '"' || next == '\\' || next == '/' ||
				next == 'b' || next == 'f' || next == 'n' ||
				next == 'r' || next == 't' || next == 'u' {
				result.WriteByte(s[i])
				result.WriteByte(next)
				i++
			} else {
				result.WriteString("\\\\")
			}
		} else {
			result.WriteByte(s[i])
		}
	}
	return result.String()
}

// jsonResponse helper to send JSON data.
func jsonResponse(w http.ResponseWriter, data interface{}) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(data)
}

// jsonError helper to send JSON error responses.
func jsonError(w http.ResponseWriter, message string, code int) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(map[string]string{"error": message})
}

// retrieveWebSearch queries the Python RAG service's /search endpoint for live grounding context.
func (h *Handlers) retrieveWebSearch(ctx context.Context, query string) ([]string, error) {
	reqBody := map[string]interface{}{"query": query, "max_results": 3}
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal search request: %w", err)
	}

	resp, err := h.HTTPClient.Post(h.RAGServiceURL+"/search", "application/json", bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("search request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("search failed with status %d: %s", resp.StatusCode, string(body))
	}

	var result struct {
		Results []string `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("failed to decode search response: %w", err)
	}
	return result.Results, nil
}
