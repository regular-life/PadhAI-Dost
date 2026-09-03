package handlers

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
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
