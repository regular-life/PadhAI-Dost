package handlers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 1: Fast vs Slow Model Differential Latency & TTFT Stress
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_TTFT_DifferentialLatency(t *testing.T) {
	// Simulate 3 council models with extreme latency disparity:
	// - Ultra-fast model: 15ms
	// - Medium model: 400ms
	// - Slow model: 1200ms
	// Chairman: 50ms
	fastClient := &handlerMockLLMClient{
		Name:  "mock:ultra-fast",
		Delay: 15 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Ultra fast draft answer", Model: "mock:ultra-fast"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: A, B, C\nREASONING: Ultra fast ranking", Model: "mock:ultra-fast"}, nil
		},
	}
	mediumClient := &handlerMockLLMClient{
		Name:  "mock:medium",
		Delay: 400 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Medium draft answer", Model: "mock:medium"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: B, A, C\nREASONING: Medium ranking", Model: "mock:medium"}, nil
		},
	}
	slowClient := &handlerMockLLMClient{
		Name:  "mock:slow",
		Delay: 1200 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Slow draft answer", Model: "mock:slow"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: C, B, A\nREASONING: Slow ranking", Model: "mock:slow"}, nil
		},
	}
	chairmanClient := &handlerMockLLMClient{
		Name:  "mock:chairman",
		Delay: 50 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"answer":"Empirical synthesis from 3 models","confidence":0.92,"source":"mock:chairman","reasoning":"Integrated fast, medium, and slow insights"}`,
				Model:  "mock:chairman",
			}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, []llm.LLMClient{fastClient, mediumClient, slowClient}, chairmanClient)

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Empirical TTFT stress test"}`))
	req.Header.Set("Accept", "text/event-stream")

	w := NewStreamingRecorder()

	start := time.Now()
	done := make(chan struct{})
	go func() {
		h.HandleQuery(w, req)
		close(done)
	}()

	// 1. Initial Flush: Headers (200 OK)
	select {
	case <-w.FlushChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for HTTP response headers")
	}

	// 2. Second Flush: First candidate draft (must be from ultra-fast model and < 1.5s)
	select {
	case <-w.FlushChan:
		ttft := time.Since(start)
		t.Logf("Empirical TTFT measurement: %v", ttft)
		if ttft >= 1500*time.Millisecond {
			t.Fatalf("TTFT VIOLATION: First candidate took %v (expected < 1500ms)", ttft)
		}
		if ttft > 300*time.Millisecond {
			t.Errorf("TTFT slower than expected for 15ms mock: %v", ttft)
		}

		events := parseSSEEvents(w.Body.String())
		if len(events) == 0 {
			t.Fatalf("no SSE events received on first draft flush")
		}
		if events[0].Event != "candidate_draft" {
			t.Fatalf("expected first event to be candidate_draft, got %s", events[0].Event)
		}

		var payload council.CandidateDraftPayload
		if err := json.Unmarshal([]byte(events[0].Data), &payload); err != nil {
			t.Fatalf("failed to decode candidate draft payload: %v", err)
		}
		if payload.Model != "mock:ultra-fast" && payload.ModelName != "mock:ultra-fast" {
			t.Fatalf("expected first draft from mock:ultra-fast, got %s / %s", payload.Model, payload.ModelName)
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first candidate draft")
	}

	// Wait for full deliberation stream to complete
	select {
	case <-done:
	case <-time.After(5 * time.Second):
		t.Fatal("full deliberation exceeded 5s timeout")
	}

	totalDuration := time.Since(start)
	t.Logf("Full deliberation completed in: %v", totalDuration)

	allEvents := parseSSEEvents(w.Body.String())
	drafts := 0
	reviews := 0
	finals := 0

	for _, ev := range allEvents {
		switch ev.Event {
		case "candidate_draft":
			drafts++
		case "peer_review":
			reviews++
		case "final_answer":
			finals++
		}
	}

	if drafts != 3 {
		t.Errorf("expected 3 drafts, got %d", drafts)
	}
	if reviews != 3 {
		t.Errorf("expected 3 reviews, got %d", reviews)
	}
	if finals != 1 {
		t.Errorf("expected 1 final answer, got %d", finals)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 2: Malformed / Complex Accept Header Matrix
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_AcceptHeaderMatrix(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	h := setupTestHandlersWithCouncil(t, nil, nil)

	testCases := []struct {
		name            string
		acceptHeaders   []string
		expectSSE       bool
		expectedCT      string
		expectBodyHas   string
		expectBodyNot   string
	}{
		{
			name:          "Exact text/event-stream",
			acceptHeaders: []string{"text/event-stream"},
			expectSSE:     true,
			expectedCT:    "text/event-stream; charset=utf-8",
			expectBodyHas: "event: candidate_draft\ndata:",
		},
		{
			name:          "Complex weighted Accept header with SSE higher priority",
			acceptHeaders: []string{"text/event-stream;q=0.9,application/json;q=0.8"},
			expectSSE:     true,
			expectedCT:    "text/event-stream; charset=utf-8",
			expectBodyHas: "event: final_answer\ndata:",
		},
		{
			name:          "Complex weighted Accept header with JSON higher priority",
			acceptHeaders: []string{"application/json;q=0.9,text/event-stream;q=0.5"},
			expectSSE:     true, // contains text/event-stream
			expectedCT:    "text/event-stream; charset=utf-8",
			expectBodyHas: "event: candidate_draft\ndata:",
		},
		{
			name:          "Multi-value Accept header array",
			acceptHeaders: []string{"application/json", "text/event-stream; charset=utf-8"},
			expectSSE:     true,
			expectedCT:    "text/event-stream; charset=utf-8",
			expectBodyHas: "event: final_answer\ndata:",
		},
		{
			name:          "Standard JSON Accept",
			acceptHeaders: []string{"application/json"},
			expectSSE:     false,
			expectedCT:    "application/json",
			expectBodyHas: `"answer":`,
			expectBodyNot: "event:",
		},
		{
			name:          "Wildcard Accept header (*/*)",
			acceptHeaders: []string{"*/*"},
			expectSSE:     false,
			expectedCT:    "application/json",
			expectBodyHas: `"answer":`,
			expectBodyNot: "event:",
		},
		{
			name:          "Browser standard default Accept header",
			acceptHeaders: []string{"text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8"},
			expectSSE:     false,
			expectedCT:    "application/json",
			expectBodyHas: `"answer":`,
			expectBodyNot: "event:",
		},
		{
			name:          "Omitted / Empty Accept header",
			acceptHeaders: []string{""},
			expectSSE:     false,
			expectedCT:    "application/json",
			expectBodyHas: `"answer":`,
			expectBodyNot: "event:",
		},
		{
			name:          "Malformed garbage Accept header",
			acceptHeaders: []string{";;;invalid/content;;;q=garbage"},
			expectSSE:     false,
			expectedCT:    "application/json",
			expectBodyHas: `"answer":`,
			expectBodyNot: "event:",
		},
		{
			name:          "Upper-case and whitespace variations",
			acceptHeaders: []string{"   TEXT/EVENT-STREAM ; Q=1.0   "},
			expectSSE:     true,
			expectedCT:    "text/event-stream; charset=utf-8",
			expectBodyHas: "event: candidate_draft",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Accept header test"}`))
			req.Header.Del("Accept")
			for _, hVal := range tc.acceptHeaders {
				if hVal != "" {
					req.Header.Add("Accept", hVal)
				}
			}

			w := httptest.NewRecorder()
			h.HandleQuery(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusOK {
				t.Fatalf("expected HTTP 200, got %d", res.StatusCode)
			}

			ct := res.Header.Get("Content-Type")
			if !strings.HasPrefix(ct, tc.expectedCT) {
				t.Errorf("expected Content-Type starting with %q, got %q", tc.expectedCT, ct)
			}

			body := w.Body.String()
			if tc.expectBodyHas != "" && !strings.Contains(body, tc.expectBodyHas) {
				t.Errorf("expected body to contain %q, got: %s", tc.expectBodyHas, body)
			}
			if tc.expectBodyNot != "" && strings.Contains(body, tc.expectBodyNot) {
				t.Errorf("expected body NOT to contain %q, got: %s", tc.expectBodyNot, body)
			}

			if !tc.expectSSE {
				var jsonResp QueryResponse
				if err := json.NewDecoder(res.Body).Decode(&jsonResp); err != nil {
					t.Fatalf("failed to decode JSON response: %v", err)
				}
				if jsonResp.Answer == "" {
					t.Errorf("expected valid non-empty Answer in JSON response")
				}
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 3: Strict SSE Framing & Protocol Conformance Scanner
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_StrictSSEFramingValidation(t *testing.T) {
	// Create mock clients with complex multiline answers containing quotes, newlines, tabs, and unicode
	mockClaude := &handlerMockLLMClient{
		Name: "openrouter:anthropic/claude-3.5-sonnet",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			multilineAnswer := "Line 1: CouncilAI deliberation\nLine 2: Multi-agent consensus\r\nLine 3: {\"nested\":\"json\", \"emoji\": \"🚀✨\"}\n\nLine 4: Double newlines in text!"
			return &llm.Response{Answer: multilineAnswer, Model: "openrouter:anthropic/claude-3.5-sonnet"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{
				Answer: "RANKING: A\nREASONING: Detailed reasoning\nwith multiple\nlines\nand quotes: \"Hello World\"",
				Model:  "openrouter:anthropic/claude-3.5-sonnet",
			}, nil
		},
	}
	mockGPT := &handlerMockLLMClient{
		Name: "openrouter:openai/gpt-4o",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "GPT-4o response with special chars: <>&\"'", Model: "openrouter:openai/gpt-4o"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: B, A\nREASONING: Balanced", Model: "openrouter:openai/gpt-4o"}, nil
		},
	}
	mockChairman := &handlerMockLLMClient{
		Name: "chairman:mock",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"answer":"Final consensus with \n escaped newlines and emoji 🌟","confidence":0.99,"source":"chairman:mock"}`,
				Model:  "chairman:mock",
			}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, []llm.LLMClient{mockClaude, mockGPT}, mockChairman)

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Complex multiline SSE test"}`))
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	rawBody := w.Body.Bytes()
	t.Logf("Raw SSE stream length: %d bytes", len(rawBody))

	// Strict SSE Parser using standard W3C SSE line scanner
	scanner := bufio.NewScanner(bytes.NewReader(rawBody))
	var (
		currentEvent string
		currentData  []string
		eventCount   int
	)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			// Empty line marks the end of an SSE message block
			if currentEvent != "" || len(currentData) > 0 {
				eventCount++
				joinedData := strings.Join(currentData, "\n")

				// Verify data is strictly valid JSON
				var js json.RawMessage
				if err := json.Unmarshal([]byte(joinedData), &js); err != nil {
					t.Fatalf("SSE Event #%d (%s) data is not valid JSON: %v\nData: %s", eventCount, currentEvent, err, joinedData)
				}

				t.Logf("✓ Validated SSE Event #%d: event=%s, bytes=%d", eventCount, currentEvent, len(joinedData))
				currentEvent = ""
				currentData = nil
			}
			continue
		}

		if strings.HasPrefix(line, "event:") {
			currentEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			currentData = append(currentData, strings.TrimSpace(strings.TrimPrefix(line, "data:")))
		} else if strings.HasPrefix(line, ":") {
			// SSE comment
			continue
		} else {
			t.Fatalf("STRICT SSE VIOLATION: Unexpected line in stream: %q", line)
		}
	}

	if err := scanner.Err(); err != nil {
		t.Fatalf("Scanner error reading SSE stream: %v", err)
	}

	if eventCount < 3 {
		t.Errorf("expected at least 3 SSE events (candidate_draft, peer_review, final_answer), got %d", eventCount)
	}

	// Verify the stream terminates with double newline "\n\n"
	if !bytes.HasSuffix(rawBody, []byte("\n\n")) {
		t.Errorf("SSE stream did not terminate with double newline '\\n\\n'")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 4: Dual-Alias Field Compatibility & JSON Serialization
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_DualAliasCompatibility(t *testing.T) {
	// Test CandidateDraftPayload dual alias serialization & deserialization
	draftJSON := `{"index":0,"model":"claude-3.5-sonnet","model_name":"claude-3.5-sonnet","answer":"Draft 1","content":"Draft 1","latency_ms":42}`

	// Deserialization check 1: Standard struct
	var draft council.CandidateDraftPayload
	if err := json.Unmarshal([]byte(draftJSON), &draft); err != nil {
		t.Fatalf("failed to unmarshal CandidateDraftPayload: %v", err)
	}
	if draft.Model != "claude-3.5-sonnet" || draft.ModelName != "claude-3.5-sonnet" {
		t.Errorf("mismatched model aliases: Model=%q, ModelName=%q", draft.Model, draft.ModelName)
	}
	if draft.Answer != "Draft 1" || draft.Content != "Draft 1" {
		t.Errorf("mismatched answer/content aliases: Answer=%q, Content=%q", draft.Answer, draft.Content)
	}

	// Deserialization check 2: Legacy client struct expecting only `model` and `answer`
	type LegacyCandidateDraft struct {
		Model  string `json:"model"`
		Answer string `json:"answer"`
	}
	var legacyDraft LegacyCandidateDraft
	if err := json.Unmarshal([]byte(draftJSON), &legacyDraft); err != nil {
		t.Fatalf("legacy client failed to unmarshal: %v", err)
	}
	if legacyDraft.Model != "claude-3.5-sonnet" || legacyDraft.Answer != "Draft 1" {
		t.Errorf("legacy client field mismatch: %+v", legacyDraft)
	}

	// Deserialization check 3: Modern client struct expecting `model_name` and `content`
	type ModernCandidateDraft struct {
		ModelName string `json:"model_name"`
		Content   string `json:"content"`
		LatencyMs int64  `json:"latency_ms"`
	}
	var modernDraft ModernCandidateDraft
	if err := json.Unmarshal([]byte(draftJSON), &modernDraft); err != nil {
		t.Fatalf("modern client failed to unmarshal: %v", err)
	}
	if modernDraft.ModelName != "claude-3.5-sonnet" || modernDraft.Content != "Draft 1" || modernDraft.LatencyMs != 42 {
		t.Errorf("modern client field mismatch: %+v", modernDraft)
	}

	// Test CouncilResult dual alias serialization for candidates list
	result := council.CouncilResult{
		FinalAnswer: "Synthesized consensus",
		Answer:      "Synthesized consensus",
		CandidateAnswers: []council.CandidateAnswer{
			{Model: "gpt-4o", Answer: "GPT answer", Content: "GPT answer"},
		},
		Candidates: []council.CandidateAnswer{
			{Model: "gpt-4o", Answer: "GPT answer", Content: "GPT answer"},
		},
	}
	resBytes, err := json.Marshal(result)
	if err != nil {
		t.Fatalf("failed to marshal CouncilResult: %v", err)
	}

	var resMap map[string]interface{}
	if err := json.Unmarshal(resBytes, &resMap); err != nil {
		t.Fatalf("failed to unmarshal CouncilResult to map: %v", err)
	}

	if resMap["candidate_answers"] == nil || resMap["candidates"] == nil {
		t.Errorf("CouncilResult JSON missing candidate_answers or candidates field: %s", string(resBytes))
	}
	if resMap["final_answer"] == nil || resMap["answer"] == nil {
		t.Errorf("CouncilResult JSON missing final_answer or answer field: %s", string(resBytes))
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 5: Client Disconnect / Context Cancellation Stress Test
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_ClientEarlyDisconnect(t *testing.T) {
	// Setup slow models to simulate long deliberation
	slowClient := &handlerMockLLMClient{
		Name:  "mock:slow-member",
		Delay: 2 * time.Second,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Slow", Model: "mock:slow-member"}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, []llm.LLMClient{slowClient}, nil)

	ctx, cancel := context.WithCancel(context.Background())
	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Disconnect test"}`)).WithContext(ctx)
	req.Header.Set("Accept", "text/event-stream")

	w := NewStreamingRecorder()

	done := make(chan struct{})
	go func() {
		h.HandleQuery(w, req)
		close(done)
	}()

	// Wait for initial headers flush
	select {
	case <-w.FlushChan:
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for header flush")
	}

	// Cancel context immediately to simulate client closing TCP connection
	cancel()

	// Handler must terminate cleanly within 100ms without hanging or panicking
	select {
	case <-done:
		t.Log("✓ Handler terminated immediately upon client context cancellation")
	case <-time.After(500 * time.Millisecond):
		t.Fatal("LEAK/HANG: Handler did not terminate after client disconnection")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 6: High Concurrency Load (Mixed SSE & JSON Requests)
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_HighConcurrency_MixedTraffic(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	h := setupTestHandlersWithCouncil(t, nil, nil)

	const totalRequests = 50
	var (
		wg         sync.WaitGroup
		successSSE int64
		successJSON int64
	)

	start := time.Now()

	for i := 0; i < totalRequests; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			isStreaming := id%2 == 0

			req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(fmt.Sprintf(`{"question": "Concurrent test query #%d"}`, id)))
			if isStreaming {
				req.Header.Set("Accept", "text/event-stream")
			} else {
				req.Header.Set("Accept", "application/json")
			}

			w := httptest.NewRecorder()
			h.HandleQuery(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusOK {
				t.Errorf("Request %d failed with status %d", id, res.StatusCode)
				return
			}

			if isStreaming {
				events := parseSSEEvents(w.Body.String())
				if len(events) > 0 {
					atomic.AddInt64(&successSSE, 1)
				}
			} else {
				var jsonResp QueryResponse
				if err := json.NewDecoder(res.Body).Decode(&jsonResp); err == nil && jsonResp.Answer != "" {
					atomic.AddInt64(&successJSON, 1)
				}
			}
		}(i)
	}

	wg.Wait()
	duration := time.Since(start)

	t.Logf("Completed %d concurrent requests (%d SSE, %d JSON) in %v (avg %.2fms/req)",
		totalRequests, successSSE, successJSON, duration, float64(duration.Milliseconds())/float64(totalRequests))

	if successSSE != int64(totalRequests/2) {
		t.Errorf("expected %d successful SSE requests, got %d", totalRequests/2, successSSE)
	}
	if successJSON != int64(totalRequests-totalRequests/2) {
		t.Errorf("expected %d successful JSON requests, got %d", totalRequests-totalRequests/2, successJSON)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 7: L2 Semantic Cache Hit Streaming Behavior
// ─────────────────────────────────────────────────────────────────────────────

type mockSemanticCache struct {
	response QueryResponse
}

func (m *mockSemanticCache) EnsureIndex(ctx context.Context) error { return nil }
func (m *mockSemanticCache) Put(ctx context.Context, docID string, vector []float32, response interface{}) error {
	return nil
}
func (m *mockSemanticCache) Get(ctx context.Context, docID string, vector []float32, threshold float32, dest interface{}) (bool, error) {
	if d, ok := dest.(*QueryResponse); ok {
		*d = m.response
		return true, nil
	}
	return false, nil
}
func (m *mockSemanticCache) Close() error { return nil }

func TestEmpirical_SemanticCacheHit_SSEStreaming(t *testing.T) {
	// Start a mock RAG server to serve /embed for semantic cache lookup
	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/embed" {
			emb := make([]float32, 384)
			for i := range emb {
				emb[i] = 0.05
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": emb})
			return
		}
		http.NotFound(w, r)
	}))
	defer ragServer.Close()

	semCache := &mockSemanticCache{
		response: QueryResponse{
			Answer:     "Cached semantic deliberation response",
			Confidence: 0.98,
			Source:     "redis_semantic_vss_hit",
			CacheHit:   true,
		},
	}

	h := &Handlers{
		RAGServiceURL: ragServer.URL,
		SemanticCache: semCache,
		HTTPClient:    ragServer.Client(),
	}

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "semantic-cache-test", "doc_id": "doc123"}`))
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200, got %d", res.StatusCode)
	}

	ct := res.Header.Get("Content-Type")
	if ct != "text/event-stream; charset=utf-8" {
		t.Errorf("expected Content-Type text/event-stream; charset=utf-8, got %s", ct)
	}

	events := parseSSEEvents(w.Body.String())
	if len(events) != 1 {
		t.Fatalf("expected exactly 1 SSE event for semantic cache hit (final_answer), got %d: %s", len(events), w.Body.String())
	}

	if events[0].Event != "final_answer" {
		t.Errorf("expected event final_answer, got %s", events[0].Event)
	}

	var data QueryResponse
	if err := json.Unmarshal([]byte(events[0].Data), &data); err != nil {
		t.Fatalf("failed to decode cached final_answer payload: %v", err)
	}

	if data.Answer != "Cached semantic deliberation response" {
		t.Errorf("expected cached answer, got %q", data.Answer)
	}
	if !data.CacheHit {
		t.Errorf("expected CacheHit true, got false")
	}
}

