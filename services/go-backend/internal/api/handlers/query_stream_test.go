package handlers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
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
		name          string
		acceptHeaders []string
		expectSSE     bool
		expectedCT    string
		expectBodyHas string
		expectBodyNot string
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
		wg          sync.WaitGroup
		successSSE  int64
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

// ── Concurrency, Disconnects, and Failure Matrices ─────────

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 1: Immediate Client Context Cancellation / Abort After First Draft
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_Scenario1_ClientAbortAfterFirstDraft_NoGoroutineLeak(t *testing.T) {
	// Baseline goroutine count before test
	runtime.GC()
	time.Sleep(20 * time.Millisecond)
	initialGoroutines := runtime.NumGoroutine()

	// 3 mock models with staggered response times: 10ms, 80ms, 150ms
	model1 := &handlerMockLLMClient{
		Name:  "mock:fast-model",
		Delay: 10 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Fast draft answer", Model: "mock:fast-model"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: A, B, C\nREASONING: Fast ranking", Model: "mock:fast-model"}, nil
		},
	}
	model2 := &handlerMockLLMClient{
		Name:  "mock:mid-model",
		Delay: 80 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Mid draft answer", Model: "mock:mid-model"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: B, A, C\nREASONING: Mid ranking", Model: "mock:mid-model"}, nil
		},
	}
	model3 := &handlerMockLLMClient{
		Name:  "mock:slow-model",
		Delay: 150 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Slow draft answer", Model: "mock:slow-model"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: C, B, A\nREASONING: Slow ranking", Model: "mock:slow-model"}, nil
		},
	}
	chairman := &handlerMockLLMClient{
		Name:  "mock:chairman",
		Delay: 50 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"answer":"Synthesized consensus answer","confidence":0.95}`,
				Model:  "mock:chairman",
			}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, []llm.LLMClient{model1, model2, model3}, chairman)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "What is the meaning of deliberation?"}`)).WithContext(ctx)
	req.Header.Set("Accept", "text/event-stream")

	w := NewStreamingRecorder()

	handlerDone := make(chan struct{})
	go func() {
		defer func() {
			if r := recover(); r != nil {
				t.Errorf("HandleQuery PANICKED on client abort: %v", r)
			}
			close(handlerDone)
		}()
		h.HandleQuery(w, req)
	}()

	// 1. Wait for stream header flush
	select {
	case <-w.FlushChan:
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for HTTP response headers")
	}

	// 2. Wait for the first candidate draft event
	select {
	case <-w.FlushChan:
		t.Log("Received first candidate draft flush, immediately cancelling client context...")
		cancel() // Abort client connection immediately
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for first candidate draft")
	}

	// 3. Verify handler exits immediately without hanging
	select {
	case <-handlerDone:
		t.Log("✓ Handler returned cleanly after client abort")
	case <-time.After(500 * time.Millisecond):
		t.Fatal("FAIL: Handler did not terminate within 500ms after client cancellation (potential deadlock/leak)")
	}

	// 4. Wait for background goroutines in orchestrator to finish cleanup
	time.Sleep(250 * time.Millisecond)
	runtime.GC()
	time.Sleep(50 * time.Millisecond)

	finalGoroutines := runtime.NumGoroutine()
	t.Logf("Goroutine count: initial=%d, final=%d", initialGoroutines, finalGoroutines)

	// Allow small delta (<= 2) for runtime/testing GC threads, but no leaked deliberation goroutines
	if finalGoroutines > initialGoroutines+2 {
		t.Errorf("LEAKED GOROUTINES DETECTED: initial=%d, final=%d, diff=%d",
			initialGoroutines, finalGoroutines, finalGoroutines-initialGoroutines)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 2: Slow Client Consumer Simulation (Slow HTTP Reader)
// ─────────────────────────────────────────────────────────────────────────────

type slowReaderBody struct {
	io.ReadCloser
	delayPerChunk time.Duration
	chunkSize     int
}

func (s *slowReaderBody) Read(p []byte) (n int, err error) {
	if s.delayPerChunk > 0 {
		time.Sleep(s.delayPerChunk)
	}
	readLen := len(p)
	if s.chunkSize > 0 && readLen > s.chunkSize {
		readLen = s.chunkSize
	}
	return s.ReadCloser.Read(p[:readLen])
}

func TestEmpirical_Scenario2_SlowClientConsumer_NoDeadlock(t *testing.T) {
	model1 := &handlerMockLLMClient{
		Name:  "mock:model-1",
		Delay: 10 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Candidate 1 answer with extensive detail about deliberation patterns.", Model: "mock:model-1"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: A, B\nREASONING: Model 1 provides strong rationale.", Model: "mock:model-1"}, nil
		},
	}
	model2 := &handlerMockLLMClient{
		Name:  "mock:model-2",
		Delay: 15 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Candidate 2 answer with alternative viewpoints and deep evidence.", Model: "mock:model-2"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: B, A\nREASONING: Model 2 is superior in tone.", Model: "mock:model-2"}, nil
		},
	}
	chairman := &handlerMockLLMClient{
		Name:  "mock:chairman",
		Delay: 10 * time.Millisecond,
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"answer":"Final consensus answer delivered to slow consumer cleanly.","confidence":0.98,"source":"mock:chairman","reasoning":"Comprehensive synthesis"}`,
				Model:  "mock:chairman",
			}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, []llm.LLMClient{model1, model2}, chairman)

	// Create real HTTP test server to test actual network buffer & backpressure
	server := httptest.NewServer(http.HandlerFunc(h.HandleQuery))
	defer server.Close()

	req, err := http.NewRequest("POST", server.URL, strings.NewReader(`{"question": "Slow client consumer test"}`))
	if err != nil {
		t.Fatalf("failed to create HTTP request: %v", err)
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 10 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		t.Fatalf("HTTP request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK, got %d", resp.StatusCode)
	}

	// Wrap body in slow reader: 20ms delay per 32-byte read
	slowBody := &slowReaderBody{
		ReadCloser:    resp.Body,
		delayPerChunk: 20 * time.Millisecond,
		chunkSize:     32,
	}

	scanner := bufio.NewScanner(slowBody)
	var events []ParsedSSEEvent
	var (
		curEvent string
		curData  string
	)

	startRead := time.Now()
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if curEvent != "" || curData != "" {
				events = append(events, ParsedSSEEvent{Event: curEvent, Data: curData})
				curEvent = ""
				curData = ""
			}
			continue
		}
		if strings.HasPrefix(line, "event:") {
			curEvent = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
		} else if strings.HasPrefix(line, "data:") {
			curData = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
		}
	}

	if err := scanner.Err(); err != nil {
		t.Fatalf("scanner error during slow consumer read: %v", err)
	}

	duration := time.Since(startRead)
	t.Logf("Slow consumer read completed in %v, received %d SSE events", duration, len(events))

	// Expected events: 2 candidate_draft + 2 peer_review + 1 final_answer = 5 events
	if len(events) != 5 {
		t.Fatalf("expected 5 SSE events, got %d", len(events))
	}

	draftCount := 0
	reviewCount := 0
	finalFound := false

	for _, ev := range events {
		switch ev.Event {
		case "candidate_draft":
			draftCount++
		case "peer_review":
			reviewCount++
		case "final_answer":
			finalFound = true
			var qr QueryResponse
			if err := json.Unmarshal([]byte(ev.Data), &qr); err != nil {
				t.Errorf("failed to unmarshal final_answer QueryResponse: %v", err)
			}
			if !strings.Contains(qr.Answer, "slow consumer") {
				t.Errorf("unexpected final answer content: %s", qr.Answer)
			}
		}
	}

	if draftCount != 2 {
		t.Errorf("expected 2 candidate drafts, got %d", draftCount)
	}
	if reviewCount != 2 {
		t.Errorf("expected 2 peer reviews, got %d", reviewCount)
	}
	if !finalFound {
		t.Errorf("final_answer was not received by slow consumer")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 3: Partial Model Failure During Deliberation (1 of 3, 2 of 3, etc.)
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpirical_Scenario3_PartialModelFailureMatrix(t *testing.T) {
	t.Run("3A: 1 of 3 Models Errors Out -> 2 Candidates Deliberate Cleanly", func(t *testing.T) {
		clientGood1 := &handlerMockLLMClient{
			Name: "mock:good-1",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Good 1 draft", Model: "mock:good-1"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return &llm.Response{Answer: "RANKING: A, B\nREASONING: Good 1 review", Model: "mock:good-1"}, nil
			},
		}
		clientFailing := &handlerMockLLMClient{
			Name: "mock:failing-model",
			Fail: true,
		}
		clientGood2 := &handlerMockLLMClient{
			Name: "mock:good-2",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Good 2 draft", Model: "mock:good-2"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return &llm.Response{Answer: "RANKING: B, A\nREASONING: Good 2 review", Model: "mock:good-2"}, nil
			},
		}
		chairman := &handlerMockLLMClient{
			Name: "mock:chairman",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{
					Answer: `{"answer":"Synthesized answer from 2 remaining healthy models","confidence":0.88,"source":"mock:chairman"}`,
					Model:  "mock:chairman",
				}, nil
			},
		}

		h := setupTestHandlersWithCouncil(t, []llm.LLMClient{clientGood1, clientFailing, clientGood2}, chairman)

		req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Partial failure 1 of 3 test"}`))
		req.Header.Set("Accept", "text/event-stream")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		events := parseSSEEvents(w.Body.String())
		if len(events) == 0 {
			t.Fatalf("no SSE events received: %s", w.Body.String())
		}

		var errorDraftFound, finalAnswerFound bool
		var validDraftCount int

		for _, ev := range events {
			if ev.Event == "candidate_draft" {
				var draft council.CandidateDraftPayload
				if err := json.Unmarshal([]byte(ev.Data), &draft); err != nil {
					t.Fatalf("failed to decode draft: %v", err)
				}
				if draft.Error != "" {
					errorDraftFound = true
					if draft.Model != "mock:failing-model" {
						t.Errorf("expected failing model to be mock:failing-model, got %s", draft.Model)
					}
				} else {
					validDraftCount++
				}
			}
			if ev.Event == "final_answer" {
				finalAnswerFound = true
				var res QueryResponse
				if err := json.Unmarshal([]byte(ev.Data), &res); err != nil {
					t.Fatalf("failed to unmarshal final answer: %v", err)
				}
				if res.Answer != "Synthesized answer from 2 remaining healthy models" {
					t.Errorf("unexpected final answer: %s", res.Answer)
				}
				if res.Confidence != 0.88 {
					t.Errorf("expected confidence 0.88, got %f", res.Confidence)
				}
			}
		}

		if !errorDraftFound {
			t.Errorf("expected an error candidate_draft event for failing model")
		}
		if validDraftCount != 2 {
			t.Errorf("expected 2 valid candidate drafts, got %d", validDraftCount)
		}
		if !finalAnswerFound {
			t.Errorf("expected final_answer event to be emitted successfully despite 1 model failure")
		}
	})

	t.Run("3B: 2 of 3 Models Error Out -> Single Model Fallback (Skip Review & Chairman)", func(t *testing.T) {
		clientFail1 := &handlerMockLLMClient{Name: "mock:fail-1", Fail: true}
		clientGood := &handlerMockLLMClient{
			Name: "mock:only-survivor",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Survivor response", Model: "mock:only-survivor"}, nil
			},
		}
		clientFail2 := &handlerMockLLMClient{Name: "mock:fail-2", Fail: true}

		h := setupTestHandlersWithCouncil(t, []llm.LLMClient{clientFail1, clientGood, clientFail2}, nil)

		req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "2 of 3 fail test"}`))
		req.Header.Set("Accept", "text/event-stream")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		events := parseSSEEvents(w.Body.String())
		var finalAnswerFound bool
		var peerReviewCount int

		for _, ev := range events {
			if ev.Event == "peer_review" {
				peerReviewCount++
			}
			if ev.Event == "final_answer" {
				finalAnswerFound = true
				var res QueryResponse
				if err := json.Unmarshal([]byte(ev.Data), &res); err != nil {
					t.Fatalf("failed to unmarshal final answer: %v", err)
				}
				if res.Answer != "Survivor response" {
					t.Errorf("expected survivor response, got: %s", res.Answer)
				}
				if !strings.Contains(res.Source, "single-response") {
					t.Errorf("expected source to indicate single-response, got: %s", res.Source)
				}
			}
		}

		if peerReviewCount > 0 {
			t.Errorf("peer review should be skipped when only 1 model succeeds, but got %d reviews", peerReviewCount)
		}
		if !finalAnswerFound {
			t.Errorf("final_answer event was not emitted for single model fallback")
		}
	})

	t.Run("3C: Peer Review Complete Failure -> Fallback to Best Candidate", func(t *testing.T) {
		client1 := &handlerMockLLMClient{
			Name: "mock:cand-1",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Short candidate answer", Model: "mock:cand-1"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return nil, context.DeadlineExceeded // All peer reviews fail
			},
		}
		client2 := &handlerMockLLMClient{
			Name: "mock:cand-2",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Longer comprehensive candidate answer", Model: "mock:cand-2"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return nil, context.DeadlineExceeded // All peer reviews fail
			},
		}

		h := setupTestHandlersWithCouncil(t, []llm.LLMClient{client1, client2}, nil)

		req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Peer review fail test"}`))
		req.Header.Set("Accept", "text/event-stream")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		events := parseSSEEvents(w.Body.String())
		var finalAnswerFound bool
		for _, ev := range events {
			if ev.Event == "final_answer" {
				finalAnswerFound = true
				var res QueryResponse
				if err := json.Unmarshal([]byte(ev.Data), &res); err != nil {
					t.Fatalf("failed to decode final answer: %v", err)
				}
				if res.Answer != "Longer comprehensive candidate answer" {
					t.Errorf("expected fallback to pick longest candidate, got: %s", res.Answer)
				}
				if !strings.Contains(res.Source, "peer-review-failed-fallback") {
					t.Errorf("expected source to contain 'peer-review-failed-fallback', got: %s", res.Source)
				}
			}
		}
		if !finalAnswerFound {
			t.Errorf("expected final_answer event on peer review failure fallback")
		}
	})

	t.Run("3D: Chairman Synthesis Failure -> Fallback to Best Peer-Reviewed Candidate", func(t *testing.T) {
		client1 := &handlerMockLLMClient{
			Name: "mock:cand-1",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Draft from model 1", Model: "mock:cand-1"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return &llm.Response{Answer: "RANKING: B, A\nREASONING: Model 2 is superior", Model: "mock:cand-1"}, nil
			},
		}
		client2 := &handlerMockLLMClient{
			Name: "mock:cand-2",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Draft from model 2 (Ranked Best)", Model: "mock:cand-2"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return &llm.Response{Answer: "RANKING: B, A\nREASONING: Model 2 is superior", Model: "mock:cand-2"}, nil
			},
		}
		chairmanFailing := &handlerMockLLMClient{
			Name: "mock:failing-chairman",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return nil, fmt.Errorf("simulated upstream chairman API 500 error")
			},
		}

		h := setupTestHandlersWithCouncil(t, []llm.LLMClient{client1, client2}, chairmanFailing)

		req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Chairman fail test"}`))
		req.Header.Set("Accept", "text/event-stream")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		events := parseSSEEvents(w.Body.String())
		var finalAnswerFound bool
		for _, ev := range events {
			if ev.Event == "final_answer" {
				finalAnswerFound = true
				var res QueryResponse
				if err := json.Unmarshal([]byte(ev.Data), &res); err != nil {
					t.Fatalf("failed to decode final answer: %v", err)
				}
				if res.Answer != "Draft from model 2 (Ranked Best)" {
					t.Errorf("expected chairman failure fallback to select top-ranked model 2, got: %s", res.Answer)
				}
				if !strings.Contains(res.Source, "chairman-failed-fallback") {
					t.Errorf("expected source to contain 'chairman-failed-fallback', got: %s", res.Source)
				}
			}
		}
		if !finalAnswerFound {
			t.Errorf("expected final_answer event on chairman failure fallback")
		}
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 4: Benchmarking Execution & Stress Verification
// ─────────────────────────────────────────────────────────────────────────────

func BenchmarkHandleQuery_SSE_Streaming(b *testing.B) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	orch := council.NewOrchestrator(nil, nil, 5*time.Second)
	h := &Handlers{Council: orch}

	reqBody := `{"question": "Benchmark streaming query throughput"}`

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Accept", "text/event-stream")
		w := httptest.NewRecorder()
		h.HandleQuery(w, req)
		if w.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", w.Code)
		}
	}
}

func BenchmarkHandleQuery_JSON_Standard(b *testing.B) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	orch := council.NewOrchestrator(nil, nil, 5*time.Second)
	h := &Handlers{Council: orch}

	reqBody := `{"question": "Benchmark JSON query throughput"}`

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Accept", "application/json")
		w := httptest.NewRecorder()
		h.HandleQuery(w, req)
		if w.Code != http.StatusOK {
			b.Fatalf("expected 200, got %d", w.Code)
		}
	}
}

func TestEmpirical_Stress50SequentialStreamRequests(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	h := setupTestHandlersWithCouncil(t, nil, nil)

	const iterations = 50
	var totalEvents int64

	start := time.Now()
	for i := 0; i < iterations; i++ {
		req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(fmt.Sprintf(`{"question": "Stress iteration %d"}`, i)))
		req.Header.Set("Accept", "text/event-stream")

		w := httptest.NewRecorder()
		h.HandleQuery(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("iteration %d failed with status %d", i, w.Code)
		}

		events := parseSSEEvents(w.Body.String())
		if len(events) == 0 {
			t.Fatalf("iteration %d received zero SSE events", i)
		}
		atomic.AddInt64(&totalEvents, int64(len(events)))
	}

	duration := time.Since(start)
	t.Logf("Completed %d sequential streaming requests (%d total events parsed) in %v (avg %.2fms/req)",
		iterations, totalEvents, duration, float64(duration.Milliseconds())/float64(iterations))
}
