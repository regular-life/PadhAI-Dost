package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"runtime"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

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
