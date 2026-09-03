package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/regular-life/CouncilAI/go-backend/internal/agent"
	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
	"github.com/regular-life/CouncilAI/go-backend/internal/memory"
)

// mockL2SemanticCache implements cache.SemanticCache for testing L2 cache behaviors.
type mockL2SemanticCache struct {
	mu       sync.RWMutex
	getErr   error
	putErr   error
	found    bool
	response QueryResponse
	putCalls int
	getCalls int
}

func (m *mockL2SemanticCache) EnsureIndex(ctx context.Context) error { return nil }
func (m *mockL2SemanticCache) Put(ctx context.Context, docID string, vector []float32, response interface{}) error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.putCalls++
	return m.putErr
}
func (m *mockL2SemanticCache) Get(ctx context.Context, docID string, vector []float32, threshold float32, dest interface{}) (bool, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.getCalls++
	if m.getErr != nil {
		return false, m.getErr
	}
	if m.found {
		if d, ok := dest.(*QueryResponse); ok {
			*d = m.response
			return true, nil
		}
	}
	return false, nil
}
func (m *mockL2SemanticCache) Close() error { return nil }

func setupCacheTestHandlers(t *testing.T, l1 *cache.RedisCache, l2 cache.SemanticCache, ragURL string) *Handlers {
	t.Helper()

	mockLLM := &handlerMockLLMClient{
		Name: "mock:test-model",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Mock deliberation response", Model: "mock:test-model"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: A\nREASONING: Top quality", Model: "mock:test-model"}, nil
		},
	}
	mockChairman := &handlerMockLLMClient{
		Name: "mock:chairman",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer:     `{"answer":"Synthesized consensus answer from council","confidence":0.95,"source":"mock:chairman"}`,
				Model:      "mock:chairman",
				Confidence: 0.95,
			}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{
				Answer:     `{"answer":"Synthesized consensus answer from council","confidence":0.95,"source":"mock:chairman"}`,
				Model:      "mock:chairman",
				Confidence: 0.95,
			}, nil
		},
	}

	orch := council.NewOrchestrator([]llm.LLMClient{mockLLM}, mockChairman, 5*time.Second)

	return &Handlers{
		RAGServiceURL:          ragURL,
		Council:                orch,
		Cache:                  l1,
		SemanticCache:          l2,
		HTTPClient:             &http.Client{Timeout: 5 * time.Second},
		SemanticCacheThreshold: 0.85,
	}
}

func TestQueryCache_L1ExactHit_JSON(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)

	cachedAns := QueryResponse{
		Answer:     "Exact match cached answer",
		Confidence: 0.99,
		Source:     "redis_exact_cache_hit",
		CacheHit:   true,
	}
	data, _ := json.Marshal(cachedAns)
	l1.SetMockData("cache:doc123:What is CouncilAI?", string(data))

	h := setupCacheTestHandlers(t, l1, nil, "")

	reqBody := `{"question": "What is CouncilAI?", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", res.StatusCode)
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Answer != "Exact match cached answer" {
		t.Errorf("expected cached answer, got %q", resp.Answer)
	}
	if !resp.CacheHit {
		t.Errorf("expected CacheHit true, got false")
	}
}

func TestQueryCache_L1ExactHit_SSE(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)

	cachedAns := QueryResponse{
		Answer:     "Exact match cached answer via SSE",
		Confidence: 0.99,
		Source:     "redis_exact_cache_hit",
		CacheHit:   true,
	}
	data, _ := json.Marshal(cachedAns)
	l1.SetMockData("cache:doc123:What is CouncilAI?", string(data))

	h := setupCacheTestHandlers(t, l1, nil, "")

	reqBody := `{"question": "What is CouncilAI?", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", res.StatusCode)
	}

	contentType := res.Header.Get("Content-Type")
	if !strings.Contains(contentType, "text/event-stream") {
		t.Fatalf("expected text/event-stream, got %q", contentType)
	}

	events := parseSSEEvents(w.Body.String())
	if len(events) == 0 {
		t.Fatal("expected at least 1 SSE event")
	}

	var foundFinal bool
	for _, ev := range events {
		if ev.Event == "final_answer" {
			foundFinal = true
			var qr QueryResponse
			if err := json.Unmarshal([]byte(ev.Data), &qr); err != nil {
				t.Fatalf("failed to parse final_answer JSON: %v", err)
			}
			if !qr.CacheHit {
				t.Errorf("expected CacheHit true in final_answer SSE frame")
			}
			if qr.Answer != "Exact match cached answer via SSE" {
				t.Errorf("unexpected answer in final_answer: %q", qr.Answer)
			}
		}
	}
	if !foundFinal {
		t.Errorf("expected event: final_answer in SSE stream")
	}
}

func TestQueryCache_L1Miss_FullDeliberation(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)

	h := setupCacheTestHandlers(t, l1, nil, "")

	reqBody := `{"question": "How does CouncilAI synthesize consensus?", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", res.StatusCode)
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.CacheHit {
		t.Errorf("expected CacheHit false on miss, got true")
	}
	if resp.Answer == "" {
		t.Errorf("expected non-empty deliberation answer")
	}

	// Verify L1 cache was populated
	var stored QueryResponse
	found, err := l1.Get(context.Background(), "cache:doc123:How does CouncilAI synthesize consensus?", &stored)
	if err != nil || !found {
		t.Errorf("expected L1 cache to be populated after miss, found=%v err=%v", found, err)
	}
}

func TestQueryCache_RedisDown_ReadFailure_DegradesToDeliberation(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l1.SetMockGetErr(errors.New("dial tcp 127.0.0.1:6379: connect: connection refused"))

	h := setupCacheTestHandlers(t, l1, nil, "")

	reqBody := `{"question": "Redis is offline question"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("FAIL: Redis read error caused HTTP %d (expected 200 OK): %s", res.StatusCode, w.Body.String())
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.CacheHit {
		t.Errorf("expected CacheHit false, got true")
	}
	if resp.Answer == "" {
		t.Errorf("expected valid non-empty LLM answer when Redis read fails")
	}
}

func TestQueryCache_RedisOOM_WriteFailure_Returns200OK_JSON(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l1.SetMockSetErr(errors.New("OOM command not allowed when used memory > 'maxmemory'"))

	h := setupCacheTestHandlers(t, l1, nil, "")

	reqBody := `{"question": "How does CouncilAI handle Redis OOM?", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	// CRITICAL INVARIANT: Must return HTTP 200 OK despite Redis OOM write error
	if res.StatusCode != http.StatusOK {
		t.Fatalf("FAIL: Redis OOM write error caused HTTP %d (expected 200 OK): %s", res.StatusCode, w.Body.String())
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.CacheHit {
		t.Errorf("expected CacheHit false, got true")
	}
	if resp.Answer == "" {
		t.Errorf("expected valid synthesized answer from deliberation")
	}
}

func TestQueryCache_RedisOOM_WriteFailure_Returns200OK_SSE(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l1.SetMockSetErr(errors.New("OOM command not allowed when used memory > 'maxmemory'"))

	h := setupCacheTestHandlers(t, l1, nil, "")

	reqBody := `{"question": "How does SSE handle Redis OOM?"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", res.StatusCode)
	}

	events := parseSSEEvents(w.Body.String())
	for _, ev := range events {
		if ev.Event == "error" {
			t.Fatalf("unexpected SSE error frame during OOM write degradation: %s", ev.Data)
		}
	}

	var foundFinal bool
	for _, ev := range events {
		if ev.Event == "final_answer" {
			foundFinal = true
		}
	}
	if !foundFinal {
		t.Errorf("expected event: final_answer in SSE stream despite Redis OOM write error")
	}
}

func TestQueryCache_L2SemanticHit_JSON(t *testing.T) {
	t.Parallel()

	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/embed") {
			vec := make([]float32, 384)
			for i := range vec {
				vec[i] = 0.05
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": vec})
			return
		}
		http.NotFound(w, r)
	}))
	defer ragServer.Close()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l2 := &mockL2SemanticCache{
		found: true,
		response: QueryResponse{
			Answer:     "Semantic RediSearch VSS hit answer",
			Confidence: 0.94,
			Source:     "semantic_cache_hit",
			CacheHit:   true,
		},
	}

	h := setupCacheTestHandlers(t, l1, l2, ragServer.URL)

	reqBody := `{"question": "Semantic query?", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", res.StatusCode)
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if resp.Answer != "Semantic RediSearch VSS hit answer" {
		t.Errorf("expected semantic cached answer, got %q", resp.Answer)
	}
	if !resp.CacheHit {
		t.Errorf("expected CacheHit true for semantic hit")
	}
}

func TestQueryCache_L2SemanticHit_SSE(t *testing.T) {
	t.Parallel()

	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/embed") {
			vec := make([]float32, 384)
			for i := range vec {
				vec[i] = 0.05
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": vec})
			return
		}
		http.NotFound(w, r)
	}))
	defer ragServer.Close()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l2 := &mockL2SemanticCache{
		found: true,
		response: QueryResponse{
			Answer:     "Semantic RediSearch SSE hit answer",
			Confidence: 0.94,
			Source:     "semantic_cache_hit",
			CacheHit:   true,
		},
	}

	h := setupCacheTestHandlers(t, l1, l2, ragServer.URL)

	reqBody := `{"question": "Semantic SSE query?", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", res.StatusCode)
	}

	events := parseSSEEvents(w.Body.String())
	var foundFinal bool
	for _, ev := range events {
		if ev.Event == "final_answer" {
			foundFinal = true
			var qr QueryResponse
			_ = json.Unmarshal([]byte(ev.Data), &qr)
			if !qr.CacheHit {
				t.Errorf("expected CacheHit true in final_answer SSE frame")
			}
		}
	}
	if !foundFinal {
		t.Errorf("expected final_answer event for semantic cache hit")
	}
}

func TestQueryCache_L2WriteFailure_OOM_GracefulDegradation(t *testing.T) {
	t.Parallel()

	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/embed") {
			vec := make([]float32, 384)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": vec})
			return
		}
		http.NotFound(w, r)
	}))
	defer ragServer.Close()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l2 := &mockL2SemanticCache{
		putErr: errors.New("OOM command not allowed when used memory > 'maxmemory'"),
	}

	h := setupCacheTestHandlers(t, l1, l2, ragServer.URL)

	reqBody := `{"question": "L2 write OOM test", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("FAIL: L2 write OOM error caused HTTP %d (expected 200 OK): %s", res.StatusCode, w.Body.String())
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Answer == "" {
		t.Errorf("expected valid answer despite L2 write OOM")
	}
}

func TestQueryCache_PythonRAGEmbedFailure_BypassesL2(t *testing.T) {
	t.Parallel()

	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "internal rag service error", http.StatusInternalServerError)
	}))
	defer ragServer.Close()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l2 := &mockL2SemanticCache{}

	h := setupCacheTestHandlers(t, l1, l2, ragServer.URL)

	reqBody := `{"question": "RAG embed down test", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK when RAG embed fails, got %d: %s", res.StatusCode, w.Body.String())
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.Answer == "" {
		t.Errorf("expected non-empty deliberation answer")
	}
	if l2.getCalls != 0 {
		t.Errorf("expected L2 get to be skipped when embed fails, got %d calls", l2.getCalls)
	}
}

func TestQueryCache_CircuitBreakerOpen_FastBypass(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	cb.Trip() // force open state immediately

	l1 := cache.NewMockRedisCache(cb)
	h := setupCacheTestHandlers(t, l1, nil, "")

	start := time.Now()
	reqBody := `{"question": "Fast fail test", "doc_id": "doc123"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	elapsed := time.Since(start)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK when circuit breaker is open, got %d", res.StatusCode)
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}
	if resp.CacheHit {
		t.Errorf("expected CacheHit false when circuit breaker is open, got true")
	}
	if elapsed > 1*time.Second {
		t.Errorf("expected fast-fail cache bypass, took %v", elapsed)
	}
}

func TestQueryCache_HighConcurrency_MixedCacheFaults(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.Config{
		FailureThreshold: 5,
		SuccessThreshold: 2,
		Timeout:          50 * time.Millisecond,
	})
	l1 := cache.NewMockRedisCache(cb)
	h := setupCacheTestHandlers(t, l1, nil, "")

	const requests = 30
	var wg sync.WaitGroup
	statusCodes := make([]int, requests)

	for i := 0; i < requests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			// Randomly inject faults
			if idx%3 == 0 {
				l1.SetMockGetErr(errors.New("connection reset by peer"))
			} else if idx%3 == 1 {
				l1.SetMockSetErr(errors.New("OOM command not allowed"))
			}

			reqBody := `{"question": "Concurrent resilience test"}`
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")

			w := httptest.NewRecorder()
			h.HandleQuery(w, req)

			statusCodes[idx] = w.Result().StatusCode
		}(i)
	}
	wg.Wait()

	for i, code := range statusCodes {
		if code != http.StatusOK {
			t.Errorf("request %d failed with status %d (expected 200 OK)", i, code)
		}
	}
}

// ── Graceful Degradation & Redis Blackout Resilience Tests ──

// setupMultiModelDeliberationHandlers constructs Handlers with 3 council members and 1 chairman,
// ensuring the full 3-stage deliberation pipeline executes (candidate_draft -> peer_review -> final_answer).
func setupMultiModelDeliberationHandlers(
	t *testing.T,
	l1 *cache.RedisCache,
	l2 cache.SemanticCache,
	mem *memory.ConversationStore,
	ragURL string,
) *Handlers {
	t.Helper()

	models := []string{"mock:claude-3-5-sonnet", "mock:gpt-4o", "mock:gemini-1-5-pro"}
	councilClients := make([]llm.LLMClient, len(models))

	for i, m := range models {
		modelName := m
		councilClients[i] = &handlerMockLLMClient{
			Name: modelName,
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{
					Answer: fmt.Sprintf("Candidate draft response from %s", modelName),
					Model:  modelName,
				}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return &llm.Response{
					Answer: fmt.Sprintf("RANKING: B, A, C\nREASONING: Review by %s selecting model B as best", modelName),
					Model:  modelName,
				}, nil
			},
		}
	}

	mockChairman := &handlerMockLLMClient{
		Name: "mock:chairman-lead",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer:     `{"answer":"Synthesized consensus answer across 3 council models","confidence":0.97,"source":"mock:chairman-lead","reasoning":"All 3 candidates reviewed and aggregated"}`,
				Model:      "mock:chairman-lead",
				Confidence: 0.97,
			}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{
				Answer:     `{"answer":"Synthesized consensus answer across 3 council models","confidence":0.97,"source":"mock:chairman-lead","reasoning":"All 3 candidates reviewed and aggregated"}`,
				Model:      "mock:chairman-lead",
				Confidence: 0.97,
			}, nil
		},
	}

	orch := council.NewOrchestrator(councilClients, mockChairman, 5*time.Second)

	return &Handlers{
		RAGServiceURL:          ragURL,
		Council:                orch,
		Cache:                  l1,
		SemanticCache:          l2,
		Memory:                 mem,
		HTTPClient:             &http.Client{Timeout: 5 * time.Second},
		SemanticCacheThreshold: 0.85,
		Router:                 agent.NewRouter(mockChairman),
	}
}

// 1. Total Redis Blackout in JSON mode:
// L1 Get, L2 Get, Doc Summary Get, Memory Get, L1 Set, L2 Put, Memory Append ALL fail with network errors.
// MUST return HTTP 200 with complete consensus answer, CacheHit == false, and NEVER return HTTP 500.
func TestAdversarialM3_TotalRedisBlackout_MultiModel_JSON(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l1.SetMockGetErr(errors.New("dial tcp 127.0.0.1:6379: connect: connection refused"))
	l1.SetMockSetErr(errors.New("dial tcp 127.0.0.1:6379: connect: connection refused"))

	l2 := &mockL2SemanticCache{
		getErr: errors.New("read tcp 127.0.0.1:6379: i/o timeout"),
		putErr: errors.New("write tcp 127.0.0.1:6379: broken pipe"),
	}

	// Memory store pointing to unreachable socket
	mem := memory.NewConversationStore("127.0.0.1:54321", "", 0, 10, time.Hour)

	// Mock RAG service for /embed
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/embed") {
			vec := make([]float32, 384)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": vec})
			return
		}
		http.NotFound(w, r)
	}))
	defer ragServer.Close()

	h := setupMultiModelDeliberationHandlers(t, l1, l2, mem, ragServer.URL)

	reqBody := `{"question": "How does CouncilAI withstand total Redis blackout?", "doc_id": "doc123", "session_id": "sess-blackout-1"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("CRITICAL BUG: Total Redis blackout caused HTTP %d (expected 200 OK): %s", res.StatusCode, w.Body.String())
	}

	var resp QueryResponse
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode JSON response: %v", err)
	}

	if resp.CacheHit {
		t.Errorf("expected CacheHit false under Redis blackout, got true")
	}
	if resp.Answer == "" {
		t.Errorf("expected non-empty synthesized deliberation answer")
	}
	if !strings.Contains(resp.Answer, "Synthesized consensus answer") {
		t.Errorf("unexpected answer content: %q", resp.Answer)
	}
	if !resp.PeerReviewed {
		t.Errorf("expected PeerReviewed true from 3-model council")
	}
	if len(resp.Candidates) != 3 {
		t.Errorf("expected 3 candidate answers from 3 council models, got %d", len(resp.Candidates))
	}
}

// 2. Total Redis Blackout in SSE streaming mode:
// Deliberation must stream candidate_draft (x3), peer_review (x3), and final_answer.
// MUST NOT emit any event: error frames, and MUST return HTTP 200.
func TestAdversarialM3_TotalRedisBlackout_MultiModel_SSE_FrameSequence(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l1.SetMockGetErr(errors.New("dial tcp 127.0.0.1:6379: connect: connection refused"))
	l1.SetMockSetErr(errors.New("dial tcp 127.0.0.1:6379: connect: connection refused"))

	l2 := &mockL2SemanticCache{
		getErr: errors.New("read tcp 127.0.0.1:6379: i/o timeout"),
		putErr: errors.New("write tcp 127.0.0.1:6379: broken pipe"),
	}

	mem := memory.NewConversationStore("127.0.0.1:54321", "", 0, 10, time.Hour)

	h := setupMultiModelDeliberationHandlers(t, l1, l2, mem, "")

	reqBody := `{"question": "Verify complete SSE frame stream under Redis blackout", "doc_id": "doc123", "session_id": "sess-sse-1"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("CRITICAL BUG: Redis blackout caused HTTP %d in SSE mode: %s", res.StatusCode, w.Body.String())
	}

	contentType := res.Header.Get("Content-Type")
	if !strings.Contains(contentType, "text/event-stream") {
		t.Fatalf("expected text/event-stream header, got %q", contentType)
	}

	events := parseSSEEvents(w.Body.String())
	if len(events) == 0 {
		t.Fatalf("expected SSE events, received none. Raw body: %q", w.Body.String())
	}

	var (
		candidateDrafts []ParsedSSEEvent
		peerReviews     []ParsedSSEEvent
		finalAnswers    []ParsedSSEEvent
		errorFrames     []ParsedSSEEvent
	)

	for _, ev := range events {
		switch ev.Event {
		case "candidate_draft":
			candidateDrafts = append(candidateDrafts, ev)
		case "peer_review":
			peerReviews = append(peerReviews, ev)
		case "final_answer":
			finalAnswers = append(finalAnswers, ev)
		case "error":
			errorFrames = append(errorFrames, ev)
		default:
			t.Errorf("unexpected SSE event type: %q", ev.Event)
		}
	}

	if len(errorFrames) > 0 {
		t.Fatalf("CRITICAL BUG: Unhandled error frame emitted over SSE stream during Redis failure: %v", errorFrames)
	}

	if len(candidateDrafts) != 3 {
		t.Errorf("expected 3 candidate_draft events for 3 council members, got %d", len(candidateDrafts))
	}
	for i, cd := range candidateDrafts {
		var payload council.CandidateDraftPayload
		if err := json.Unmarshal([]byte(cd.Data), &payload); err != nil {
			t.Errorf("failed to unmarshal candidate_draft[%d]: %v", i, err)
		}
		if payload.Model == "" || payload.Answer == "" {
			t.Errorf("candidate_draft[%d] has missing model or answer: %+v", i, payload)
		}
	}

	if len(peerReviews) != 3 {
		t.Errorf("expected 3 peer_review events for 3 council members, got %d", len(peerReviews))
	}
	for i, pr := range peerReviews {
		var payload council.PeerReviewPayload
		if err := json.Unmarshal([]byte(pr.Data), &payload); err != nil {
			t.Errorf("failed to unmarshal peer_review[%d]: %v", i, err)
		}
		if payload.Reviewer == "" || payload.Review == "" {
			t.Errorf("peer_review[%d] has missing reviewer or review: %+v", i, payload)
		}
	}

	if len(finalAnswers) != 1 {
		t.Fatalf("expected exactly 1 final_answer event, got %d", len(finalAnswers))
	}

	var finalResp QueryResponse
	if err := json.Unmarshal([]byte(finalAnswers[0].Data), &finalResp); err != nil {
		t.Fatalf("failed to unmarshal final_answer payload: %v", err)
	}

	if finalResp.CacheHit {
		t.Errorf("expected final_answer CacheHit false, got true")
	}
	if !strings.Contains(finalResp.Answer, "Synthesized consensus answer") {
		t.Errorf("final_answer content unexpected: %q", finalResp.Answer)
	}
}

// 3. Redis OOM Write Failure across both L1 and L2 caches:
// Tests both JSON and SSE mode when Redis writes return "OOM command not allowed when used memory > 'maxmemory'".
// Verifies HTTP 200, valid consensus answer, and zero error frames.
func TestAdversarialM3_RedisOOM_SimultaneousWriteFailures_JSON_And_SSE(t *testing.T) {
	t.Parallel()

	oomErr := errors.New("OOM command not allowed when used memory > 'maxmemory'")

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	l1.SetMockSetErr(oomErr)

	l2 := &mockL2SemanticCache{
		putErr: oomErr,
	}

	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/embed") {
			vec := make([]float32, 384)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": vec})
			return
		}
		http.NotFound(w, r)
	}))
	defer ragServer.Close()

	h := setupMultiModelDeliberationHandlers(t, l1, l2, nil, ragServer.URL)

	t.Run("JSON_Mode_RedisOOM_Returns200OK", func(t *testing.T) {
		reqBody := `{"question": "Simultaneous L1 & L2 Redis OOM JSON test", "doc_id": "doc123"}`
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "application/json")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		res := w.Result()
		if res.StatusCode != http.StatusOK {
			t.Fatalf("expected HTTP 200 OK under Redis OOM, got %d: %s", res.StatusCode, w.Body.String())
		}

		var resp QueryResponse
		if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
			t.Fatalf("failed to decode response: %v", err)
		}
		if resp.CacheHit {
			t.Errorf("expected CacheHit false on OOM write failure")
		}
		if resp.Answer == "" {
			t.Errorf("expected valid non-empty answer")
		}
	})

	t.Run("SSE_Mode_RedisOOM_StreamsAllFramesWithoutError", func(t *testing.T) {
		reqBody := `{"question": "Simultaneous L1 & L2 Redis OOM SSE test", "doc_id": "doc123"}`
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		res := w.Result()
		if res.StatusCode != http.StatusOK {
			t.Fatalf("expected HTTP 200 OK under Redis OOM SSE, got %d", res.StatusCode)
		}

		events := parseSSEEvents(w.Body.String())
		var hasDraft, hasReview, hasFinal bool
		for _, ev := range events {
			if ev.Event == "error" {
				t.Fatalf("unexpected SSE error frame during OOM write failure: %s", ev.Data)
			}
			if ev.Event == "candidate_draft" {
				hasDraft = true
			}
			if ev.Event == "peer_review" {
				hasReview = true
			}
			if ev.Event == "final_answer" {
				hasFinal = true
			}
		}

		if !hasDraft || !hasReview || !hasFinal {
			t.Errorf("missing expected SSE frames (draft=%v, review=%v, final=%v)", hasDraft, hasReview, hasFinal)
		}
	})
}

// 4. Circuit Breaker Forced Open and Half-Open Saturation:
// Verifies that when Breaker is StateOpen or HalfOpen saturated, cache lookups immediately fast-fail
// without blocking or producing HTTP 500.
func TestAdversarialM3_CircuitBreaker_Open_And_HalfOpen_Saturation(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 2,
		Timeout:          5 * time.Second,
		HalfOpenMaxCalls: 1,
	})
	cb.Trip() // Force OPEN

	l1 := cache.NewMockRedisCache(cb)
	l2 := &mockL2SemanticCache{getErr: cache.ErrCircuitOpen, putErr: cache.ErrCircuitOpen}

	h := setupMultiModelDeliberationHandlers(t, l1, l2, nil, "")

	t.Run("StateOpen_FastBypass_JSON", func(t *testing.T) {
		start := time.Now()
		reqBody := `{"question": "Circuit open fast bypass test"}`
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		if w.Result().StatusCode != http.StatusOK {
			t.Fatalf("expected HTTP 200 when CB open, got %d", w.Result().StatusCode)
		}
		if time.Since(start) > 500*time.Millisecond {
			t.Errorf("expected near-instant cache bypass, took %v", time.Since(start))
		}
	})

	t.Run("StateOpen_FastBypass_SSE", func(t *testing.T) {
		reqBody := `{"question": "Circuit open SSE bypass test"}`
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("Accept", "text/event-stream")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		if w.Result().StatusCode != http.StatusOK {
			t.Fatalf("expected HTTP 200 when CB open, got %d", w.Result().StatusCode)
		}

		events := parseSSEEvents(w.Body.String())
		for _, ev := range events {
			if ev.Event == "error" {
				t.Fatalf("unexpected error frame when CB open: %s", ev.Data)
			}
		}
	})
}

// 5. Trip and Auto-Recovery Lifecycle through Handler:
// Simulates consecutive failures tripping the breaker from CLOSED -> OPEN -> HALF-OPEN -> CLOSED,
// verifying every single user query returns HTTP 200 OK throughout the entire lifecycle.
func TestAdversarialM3_CircuitBreaker_LifecycleThroughQueryHandler(t *testing.T) {
	t.Parallel()

	var virtualTime = time.Date(2026, 9, 3, 12, 0, 0, 0, time.UTC)
	var timeMu sync.Mutex
	getVirtualTime := func() time.Time {
		timeMu.Lock()
		defer timeMu.Unlock()
		return virtualTime
	}
	advanceVirtualTime := func(d time.Duration) {
		timeMu.Lock()
		defer timeMu.Unlock()
		virtualTime = virtualTime.Add(d)
	}

	cb := cache.NewCircuitBreaker(cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 2,
		Timeout:          10 * time.Second,
		NowFunc:          getVirtualTime,
	})

	l1 := cache.NewMockRedisCache(cb)
	l1.SetMockGetErr(errors.New("connection reset by peer"))
	l1.SetMockSetErr(errors.New("connection reset by peer"))

	h := setupMultiModelDeliberationHandlers(t, l1, nil, nil, "")

	sendQuery := func(q string) int {
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(fmt.Sprintf(`{"question": "%s"}`, q)))
		req.Header.Set("Content-Type", "application/json")
		w := httptest.NewRecorder()
		h.HandleQuery(w, req)
		return w.Result().StatusCode
	}

	// 1. Initial state CLOSED
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected initial state CLOSED, got %v", cb.State())
	}

	// Query 1: Get fails (1), Set fails (2) -> trips to OPEN -> returns 200 OK
	if code := sendQuery("Query 1"); code != http.StatusOK {
		t.Errorf("Query 1 failed with status %d", code)
	}

	if cb.State() != cache.StateOpen {
		t.Fatalf("expected state OPEN after failing query with Get and Set failures, got %v", cb.State())
	}

	// Query 2: Fast bypass during OPEN -> returns 200 OK
	if code := sendQuery("Query 2"); code != http.StatusOK {
		t.Errorf("Query 2 failed with status %d", code)
	}

	// Advance virtual time by 11 seconds (exceeding 10s cooldown)
	advanceVirtualTime(11 * time.Second)

	// Breaker should now allow a probe in HALF_OPEN
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected state HALF_OPEN after timeout, got %v", cb.State())
	}

	// Clear mock get and set errors to simulate Redis recovery
	l1.SetMockGetErr(nil)
	l1.SetMockSetErr(nil)

	// Query 3: Probe query -> Get succeeds (1) and Set succeeds (2) -> transitions back to CLOSED -> returns 200 OK
	if code := sendQuery("Query 3 probe"); code != http.StatusOK {
		t.Errorf("Query 3 probe failed with status %d", code)
	}

	if cb.State() != cache.StateClosed {
		t.Errorf("expected state CLOSED after 2 probe successes, got %v", cb.State())
	}
}

// 6. Direct Strategy Deliberation under Redis Failure in SSE mode:
// Verifies candidate_draft and final_answer frames stream cleanly when strategy is direct.
func TestAdversarialM3_DirectStrategy_RedisFailure_SSE(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	cb.Trip() // Forced open

	l1 := cache.NewMockRedisCache(cb)
	h := setupMultiModelDeliberationHandlers(t, l1, nil, nil, "")

	// Force direct strategy via mock router client
	mockDirectRouterClient := &handlerMockLLMClient{
		Name: "mock:router-direct",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"strategy":"direct","reasoning":"simple query handled directly","needs_doc":false}`,
				Model:  "mock:router-direct",
			}, nil
		},
	}
	h.Router = agent.NewRouter(mockDirectRouterClient)

	// Configure direct model in orchestrator chairmanClient
	mockDirectClient := &handlerMockLLMClient{
		Name: "mock:direct-model",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: "Direct quick response to query",
				Model:  "mock:direct-model",
			}, nil
		},
	}
	h.Council = council.NewOrchestrator(nil, mockDirectClient, 5*time.Second)

	reqBody := `{"question": "Direct strategy query under Redis failure"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d", res.StatusCode)
	}

	events := parseSSEEvents(w.Body.String())
	var hasDraft, hasFinal bool
	for _, ev := range events {
		if ev.Event == "error" {
			t.Fatalf("unexpected error frame in direct strategy: %s", ev.Data)
		}
		if ev.Event == "candidate_draft" {
			hasDraft = true
		}
		if ev.Event == "final_answer" {
			hasFinal = true
		}
	}

	if !hasDraft || !hasFinal {
		t.Errorf("expected candidate_draft and final_answer in direct mode, got draft=%v final=%v", hasDraft, hasFinal)
	}
}

// 7. Real Offline Redis Socket:
// Points a real redis.Client to an unreachable local port (127.0.0.1:63998).
// Ensures go-redis dial network error does not trigger panics or HTTP 500 errors in HandleQuery.
func TestAdversarialM3_RealOfflineRedisSocket_EndToEnd(t *testing.T) {
	t.Parallel()

	realOfflineClient := redis.NewClient(&redis.Options{
		Addr:        "127.0.0.1:63998", // Unreachable port
		DialTimeout: 50 * time.Millisecond,
		ReadTimeout: 50 * time.Millisecond,
	})
	defer realOfflineClient.Close()

	cb := cache.NewCircuitBreaker(cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 2,
		Timeout:          1 * time.Second,
	})

	l1 := cache.NewRedisCacheWithBreaker(realOfflineClient, cb, 1*time.Hour)
	h := setupMultiModelDeliberationHandlers(t, l1, nil, nil, "")

	for i := 1; i <= 3; i++ {
		reqBody := fmt.Sprintf(`{"question": "Real socket failure test %d"}`, i)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		res := w.Result()
		if res.StatusCode != http.StatusOK {
			t.Fatalf("iteration %d: real offline Redis socket caused HTTP %d (expected 200 OK)", i, res.StatusCode)
		}

		var resp QueryResponse
		if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
			t.Fatalf("iteration %d: failed to decode response: %v", i, err)
		}
		if resp.CacheHit {
			t.Errorf("iteration %d: expected CacheHit false", i)
		}
	}
}

// 8. Client Disconnect Mid-Stream:
// Cancels request context mid-stream. Verifies no goroutine leak, deadlock, or panic occurs.
func TestAdversarialM3_SSE_ClientPrematureDisconnect(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	l1 := cache.NewMockRedisCache(cb)
	h := setupMultiModelDeliberationHandlers(t, l1, nil, nil, "")

	ctx, cancel := context.WithCancel(context.Background())

	reqBody := `{"question": "Premature disconnect test"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()

	// Cancel context quickly to simulate network drop
	go func() {
		time.Sleep(2 * time.Millisecond)
		cancel()
	}()

	done := make(chan struct{})
	go func() {
		defer close(done)
		h.HandleQuery(w, req)
	}()

	select {
	case <-done:
		// Clean exit
	case <-time.After(3 * time.Second):
		t.Fatal("HandleQuery hung on premature client disconnect")
	}
}

// 9. High-Concurrency Mixed Traffic Chaos Stress Test:
// 100 concurrent requests mixing JSON and SSE modes with varied injected cache faults
// (clean miss, Redis network read timeout, Redis OOM write error, and circuit breaker open).
// Executed with -race to prove complete thread safety and HTTP 200 resilience.
func TestAdversarialM3_HighConcurrency_MixedTrafficChaos_100Goroutines(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.Config{
		FailureThreshold: 5,
		SuccessThreshold: 2,
		Timeout:          50 * time.Millisecond,
	})
	l1 := cache.NewMockRedisCache(cb)

	oomErr := errors.New("OOM command not allowed when used memory > 'maxmemory'")
	netErr := errors.New("read tcp 127.0.0.1:6379: i/o timeout")

	h := setupMultiModelDeliberationHandlers(t, l1, nil, nil, "")

	const concurrency = 100
	var wg sync.WaitGroup
	var http200Count int64
	var http500Count int64
	var sseErrorFramesCount int64

	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()

			// Interleave faults
			switch idx % 4 {
			case 0:
				// Clean cache miss (normal)
			case 1:
				l1.SetMockGetErr(netErr)
			case 2:
				l1.SetMockSetErr(oomErr)
			case 3:
				cb.Trip()
			}

			isSSE := (idx % 2) == 0
			reqBody := fmt.Sprintf(`{"question": "Chaos concurrent query %d", "doc_id": "doc_chaos"}`, idx)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			if isSSE {
				req.Header.Set("Accept", "text/event-stream")
			} else {
				req.Header.Set("Accept", "application/json")
			}

			w := httptest.NewRecorder()
			h.HandleQuery(w, req)

			code := w.Result().StatusCode
			if code == http.StatusOK {
				atomic.AddInt64(&http200Count, 1)
			} else {
				atomic.AddInt64(&http500Count, 1)
			}

			if isSSE {
				events := parseSSEEvents(w.Body.String())
				for _, ev := range events {
					if ev.Event == "error" {
						atomic.AddInt64(&sseErrorFramesCount, 1)
					}
				}
			}
		}(i)
	}

	wg.Wait()

	if http200Count != concurrency {
		t.Fatalf("FAILED: %d/%d requests succeeded with 200 OK (500s: %d)", http200Count, concurrency, http500Count)
	}
	if http500Count > 0 {
		t.Fatalf("FAILED: Encountered %d HTTP 500 responses under concurrent chaos", http500Count)
	}
	if sseErrorFramesCount > 0 {
		t.Fatalf("FAILED: Encountered %d unhandled SSE error frames under concurrent chaos", sseErrorFramesCount)
	}
}
