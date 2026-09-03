package handlers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
	"github.com/regular-life/CouncilAI/go-backend/internal/telemetry"
)

var w3cRegex = regexp.MustCompile(`^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$`)

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 1: 5-Stage Linked Span Hierarchy & Exact Parent-Child IDs
// ─────────────────────────────────────────────────────────────────────────────
func TestChallengeM4_5StageLinkedSpanHierarchy_ExactParentChildIDs(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	incomingTraceID := "aa11bb22cc33dd44ee55ff6677889900"
	incomingParentID := "1122334455667788"
	incomingTraceparent := fmt.Sprintf("00-%s-%s-01", incomingTraceID, incomingParentID)

	reqBody := `{"question": "How does CouncilAI execute 5-stage linked tracing?", "doc_id": "doc_linked"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("traceparent", incomingTraceparent)

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d: %s", w.Code, w.Body.String())
	}

	spans := f.exporter.GetSpans()
	if len(spans) == 0 {
		t.Fatal("expected spans to be recorded, got 0")
	}

	tree := BuildSpanTree(t, spans)

	// 1. Root Server Span Verification
	root := tree.Root
	if root.Name != "HTTP POST /api/v1/query" {
		t.Fatalf("expected root span name 'HTTP POST /api/v1/query', got %q", root.Name)
	}
	if root.SpanKind != trace.SpanKindServer {
		t.Errorf("expected root span kind Server, got %v", root.SpanKind)
	}
	if root.SpanContext.TraceID().String() != incomingTraceID {
		t.Errorf("expected root TraceID %s, got %s", incomingTraceID, root.SpanContext.TraceID())
	}
	if root.Parent.SpanID().String() != incomingParentID {
		t.Errorf("expected root ParentSpanID %s, got %s", incomingParentID, root.Parent.SpanID())
	}

	rootSpanID := root.SpanContext.SpanID()

	// 2. Stage 1: L1 Exact Cache Lookup
	tree.AssertSpanCount(t, "cache.l1_lookup", 1)
	l1Span := tree.ByName["cache.l1_lookup"][0]
	if l1Span.SpanContext.TraceID().String() != incomingTraceID {
		t.Errorf("L1 span TraceID %s != %s", l1Span.SpanContext.TraceID(), incomingTraceID)
	}
	if l1Span.Parent.SpanID() != rootSpanID {
		t.Errorf("L1 span ParentSpanID %s != root SpanID %s", l1Span.Parent.SpanID(), rootSpanID)
	}
	if l1Span.SpanKind != trace.SpanKindInternal {
		t.Errorf("L1 span kind expected Internal, got %v", l1Span.SpanKind)
	}

	// 3. Stage 2: Python /embed RPC call
	tree.AssertSpanCount(t, "rag.embed", 1)
	embedSpan := tree.ByName["rag.embed"][0]
	if embedSpan.SpanContext.TraceID().String() != incomingTraceID {
		t.Errorf("rag.embed span TraceID %s != %s", embedSpan.SpanContext.TraceID(), incomingTraceID)
	}
	if embedSpan.Parent.SpanID() != rootSpanID {
		t.Errorf("rag.embed span ParentSpanID %s != root SpanID %s", embedSpan.Parent.SpanID(), rootSpanID)
	}
	if embedSpan.SpanKind != trace.SpanKindClient {
		t.Errorf("rag.embed span kind expected Client, got %v", embedSpan.SpanKind)
	}

	// 4. Stage 3: L2 RediSearch VSS Lookup
	tree.AssertSpanCount(t, "cache.l2_lookup", 1)
	l2Span := tree.ByName["cache.l2_lookup"][0]
	if l2Span.SpanContext.TraceID().String() != incomingTraceID {
		t.Errorf("L2 span TraceID %s != %s", l2Span.SpanContext.TraceID(), incomingTraceID)
	}
	if l2Span.Parent.SpanID() != rootSpanID {
		t.Errorf("L2 span ParentSpanID %s != root SpanID %s", l2Span.Parent.SpanID(), rootSpanID)
	}
	if l2Span.SpanKind != trace.SpanKindInternal {
		t.Errorf("L2 span kind expected Internal, got %v", l2Span.SpanKind)
	}

	// 5. Stage 4: Council Candidate Fan-Out
	tree.AssertSpanCount(t, "council.candidate_fan_out", 1)
	fanOutSpan := tree.ByName["council.candidate_fan_out"][0]
	if fanOutSpan.SpanContext.TraceID().String() != incomingTraceID {
		t.Errorf("fan_out span TraceID %s != %s", fanOutSpan.SpanContext.TraceID(), incomingTraceID)
	}
	if fanOutSpan.Parent.SpanID() != rootSpanID {
		t.Errorf("fan_out span ParentSpanID %s != root SpanID %s", fanOutSpan.Parent.SpanID(), rootSpanID)
	}
	if fanOutSpan.SpanKind != trace.SpanKindInternal {
		t.Errorf("fan_out span kind expected Internal, got %v", fanOutSpan.SpanKind)
	}

	fanOutSpanID := fanOutSpan.SpanContext.SpanID()

	// Child worker spans under Candidate Fan-Out
	candidateSpans := tree.ByName["council.candidate_model"]
	if len(candidateSpans) != 2 {
		t.Fatalf("expected 2 candidate_model spans, got %d", len(candidateSpans))
	}
	for i, cs := range candidateSpans {
		if cs.SpanContext.TraceID().String() != incomingTraceID {
			t.Errorf("candidate_model[%d] TraceID %s != %s", i, cs.SpanContext.TraceID(), incomingTraceID)
		}
		if cs.Parent.SpanID() != fanOutSpanID {
			t.Errorf("candidate_model[%d] ParentSpanID %s != fanOutSpanID %s", i, cs.Parent.SpanID(), fanOutSpanID)
		}
		if cs.SpanKind != trace.SpanKindClient {
			t.Errorf("candidate_model[%d] kind expected Client, got %v", i, cs.SpanKind)
		}
	}

	// 6. Stage 5: Council Chairman Deliberation
	tree.AssertSpanCount(t, "council.chairman_deliberation", 1)
	chairmanSpan := tree.ByName["council.chairman_deliberation"][0]
	if chairmanSpan.SpanContext.TraceID().String() != incomingTraceID {
		t.Errorf("chairman span TraceID %s != %s", chairmanSpan.SpanContext.TraceID(), incomingTraceID)
	}
	if chairmanSpan.Parent.SpanID() != rootSpanID {
		t.Errorf("chairman span ParentSpanID %s != root SpanID %s", chairmanSpan.Parent.SpanID(), rootSpanID)
	}
	if chairmanSpan.SpanKind != trace.SpanKindInternal {
		t.Errorf("chairman span kind expected Internal, got %v", chairmanSpan.SpanKind)
	}

	// Verify all spans share exact same TraceID
	for _, s := range spans {
		if s.SpanContext.TraceID().String() != incomingTraceID {
			t.Errorf("span %s has mismatched TraceID: %s != %s", s.Name, s.SpanContext.TraceID(), incomingTraceID)
		}
		if s.StartTime.After(s.EndTime) {
			t.Errorf("span %s StartTime %v is after EndTime %v", s.Name, s.StartTime, s.EndTime)
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 2: Outgoing W3C traceparent on /embed and /retrieve
// ─────────────────────────────────────────────────────────────────────────────
func TestChallengeM4_OutgoingW3C_Traceparent_AllEndpoints(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "Verify all outgoing W3C traceparent headers", "doc_id": "doc_w3c"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d", w.Code)
	}

	f.callsMu.Lock()
	calls := f.recordedCalls
	f.callsMu.Unlock()

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)
	rootTraceID := tree.Root.SpanContext.TraceID().String()
	embedSpan := tree.ByName["rag.embed"][0]
	expectedEmbedSpanID := embedSpan.SpanContext.SpanID().String()
	expectedRootSpanID := tree.Root.SpanContext.SpanID().String()

	foundEmbed := false
	foundRetrieve := false

	for _, call := range calls {
		if strings.HasSuffix(call.Path, "/embed") {
			foundEmbed = true
			matches := w3cRegex.FindStringSubmatch(call.Traceparent)
			if len(matches) != 4 {
				t.Fatalf("outgoing /embed traceparent %q does not match W3C specification", call.Traceparent)
			}
			if matches[1] != rootTraceID {
				t.Errorf("/embed trace ID %s != root TraceID %s", matches[1], rootTraceID)
			}
			if matches[2] != expectedEmbedSpanID {
				t.Errorf("/embed parent span ID %s != rag.embed span ID %s", matches[2], expectedEmbedSpanID)
			}
			if matches[3] != "01" {
				t.Errorf("/embed trace flags expected '01' (sampled), got %q", matches[3])
			}
		}

		if strings.HasSuffix(call.Path, "/retrieve") {
			foundRetrieve = true
			matches := w3cRegex.FindStringSubmatch(call.Traceparent)
			if len(matches) != 4 {
				t.Fatalf("outgoing /retrieve traceparent %q does not match W3C specification", call.Traceparent)
			}
			if matches[1] != rootTraceID {
				t.Errorf("/retrieve trace ID %s != root TraceID %s", matches[1], rootTraceID)
			}
			if matches[2] != expectedRootSpanID {
				t.Errorf("/retrieve parent span ID %s != root span ID %s", matches[2], expectedRootSpanID)
			}
		}
	}

	if !foundEmbed {
		t.Error("mock RAG server received no /embed requests")
	}
	if !foundRetrieve {
		t.Error("mock RAG server received no /retrieve requests")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 3: Concurrency Stress Test - Zero Cross-Contamination
// ─────────────────────────────────────────────────────────────────────────────
func TestChallengeM4_Adversarial_TraceConcurrency_NoContamination(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	concurrentRequests := 30
	var wg sync.WaitGroup
	wg.Add(concurrentRequests)

	type traceResult struct {
		reqTraceID   string
		respTraceID  string
		respTPHeader string
	}
	results := make([]traceResult, concurrentRequests)

	for i := 0; i < concurrentRequests; i++ {
		go func(idx int) {
			defer wg.Done()

			tID := fmt.Sprintf("%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
				idx, idx+1, idx+2, idx+3, idx+4, idx+5, idx+6, idx+7,
				idx+8, idx+9, idx+10, idx+11, idx+12, idx+13, idx+14, idx+15)
			pID := fmt.Sprintf("%02x%02x%02x%02x%02x%02x%02x%02x",
				idx+1, idx+2, idx+3, idx+4, idx+5, idx+6, idx+7, idx+8)
			tpHeader := fmt.Sprintf("00-%s-%s-01", tID, pID)

			reqBody := fmt.Sprintf(`{"question": "Concurrency query %d", "doc_id": "doc_conc_%d"}`, idx, idx)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("traceparent", tpHeader)

			w := httptest.NewRecorder()
			f.handlers.HandleQuery(w, req)

			results[idx] = traceResult{
				reqTraceID:   tID,
				respTraceID:  w.Header().Get("X-Trace-ID"),
				respTPHeader: w.Header().Get("traceparent"),
			}
		}(i)
	}

	wg.Wait()

	// Assert each concurrent request retained its exact TraceID
	for i, res := range results {
		if res.respTraceID != res.reqTraceID {
			t.Errorf("request %d: expected response X-Trace-ID %s, got %s", i, res.reqTraceID, res.respTraceID)
		}
		if !strings.Contains(res.respTPHeader, res.reqTraceID) {
			t.Errorf("request %d: response traceparent %s does not contain TraceID %s", i, res.respTPHeader, res.reqTraceID)
		}
	}

	// Group spans by TraceID and verify complete parent-child trees without cross-leakage
	spans := f.exporter.GetSpans()
	spansByTrace := make(map[string][]tracetest.SpanStub)
	for _, s := range spans {
		spansByTrace[s.SpanContext.TraceID().String()] = append(spansByTrace[s.SpanContext.TraceID().String()], s)
	}

	if len(spansByTrace) < concurrentRequests {
		t.Errorf("expected at least %d unique traces recorded, got %d", concurrentRequests, len(spansByTrace))
	}

	for i := 0; i < concurrentRequests; i++ {
		tID := results[i].reqTraceID
		traceSpans := spansByTrace[tID]
		if len(traceSpans) == 0 {
			t.Errorf("no spans recorded for trace ID %s", tID)
			continue
		}
		tree := BuildSpanTree(t, traceSpans)
		rootID := tree.Root.SpanContext.SpanID()
		for _, s := range traceSpans {
			if s.SpanContext.SpanID() == rootID {
				continue
			}
			if s.Name == "council.candidate_model" {
				fanOut := tree.ByName["council.candidate_fan_out"][0]
				if s.Parent.SpanID() != fanOut.SpanContext.SpanID() {
					t.Errorf("trace %s: candidate_model parent %s != fanOut %s", tID, s.Parent.SpanID(), fanOut.SpanContext.SpanID())
				}
			} else {
				if s.Parent.SpanID() != rootID {
					t.Errorf("trace %s: span %s parent %s != root %s", tID, s.Name, s.Parent.SpanID(), rootID)
				}
			}
		}
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 4: Adversarial Degradation & Error Status Tracing
// ─────────────────────────────────────────────────────────────────────────────
func TestChallengeM4_Adversarial_DegradationErrorTracing(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	// 1. Force L1 Redis cache to return error
	f.l1Cache.SetMockGetErr(fmt.Errorf("simulated redis timeout connection lost"))

	// 2. Force L2 Semantic cache to return error
	f.l2Cache.getErr = fmt.Errorf("simulated l2 circuit open error")

	reqBody := `{"question": "Degradation tracing under Redis collapse", "doc_id": "doc_degrade"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	// Graceful degradation: Client request MUST succeed (HTTP 200) despite Redis failures
	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 with graceful degradation, got %d: %s", w.Code, w.Body.String())
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	// L1 span MUST record error status
	l1Spans := tree.ByName["cache.l1_lookup"]
	if len(l1Spans) == 0 {
		t.Fatal("expected cache.l1_lookup span")
	}
	if l1Spans[0].Status.Code != codes.Error {
		t.Errorf("expected cache.l1_lookup status Error, got %v", l1Spans[0].Status.Code)
	}

	// L2 span MUST record error status
	l2Spans := tree.ByName["cache.l2_lookup"]
	if len(l2Spans) == 0 {
		t.Fatal("expected cache.l2_lookup span")
	}
	if l2Spans[0].Status.Code != codes.Error {
		t.Errorf("expected cache.l2_lookup status Error, got %v", l2Spans[0].Status.Code)
	}

	// Downstream council stages MUST still execute cleanly
	tree.AssertSpanCount(t, "council.candidate_fan_out", 1)
	tree.AssertSpanCount(t, "council.chairman_deliberation", 1)

	// Overall root span MUST be OK because query completed successfully
	if tree.Root.Status.Code != codes.Ok {
		t.Errorf("expected root span status OK, got %v", tree.Root.Status.Code)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Empirical Challenge 5: Performance Overhead Benchmarks (< 200µs/op)
// ─────────────────────────────────────────────────────────────────────────────

// Benchmark individual OpenTelemetry operations
func BenchmarkChallengeM4_StartSpan_InMemory(b *testing.B) {
	tracer, _ := telemetry.NewTestTracer()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, span := tracer.StartSpan(ctx, "benchmark.span")
		span.End()
	}
}

func BenchmarkChallengeM4_StartSpan_Noop(b *testing.B) {
	tracer := telemetry.NewNoopTracerProvider()
	ctx := context.Background()

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, span := tracer.StartSpan(ctx, "benchmark.span")
		span.End()
	}
}

func BenchmarkChallengeM4_InjectHTTPHeaders(b *testing.B) {
	tracer, _ := telemetry.NewTestTracer()
	ctx, span := tracer.StartSpan(context.Background(), "benchmark.parent")
	defer span.End()

	req, _ := http.NewRequest("POST", "http://localhost/embed", nil)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tracer.InjectHTTPHeaders(ctx, req)
	}
}

func BenchmarkChallengeM4_ExtractHTTPHeaders(b *testing.B) {
	tracer, _ := telemetry.NewTestTracer()
	req, _ := http.NewRequest("POST", "http://localhost/query", nil)
	req.Header.Set("traceparent", "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01")

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = tracer.ExtractHTTPHeaders(context.Background(), req)
	}
}

// Benchmark the entire HandleQuery pipeline comparing:
// 1. With InMemory Tracing
// 2. With Noop Tracing
// 3. Tracing Disabled (nil Tracer)
func BenchmarkChallengeM4_QueryPipeline_OverheadDelta(b *testing.B) {
	// Setup fast mock LLMs for benchmarking pipeline handler overhead
	mockClient1 := &handlerMockLLMClient{
		Name: "mock:1",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "A", Model: "mock:1"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: A", Model: "mock:1"}, nil
		},
	}
	mockClient2 := &handlerMockLLMClient{
		Name: "mock:2",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "B", Model: "mock:2"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: B", Model: "mock:2"}, nil
		},
	}
	mockChairman := &handlerMockLLMClient{
		Name: "mock:c",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: `{"answer":"Final","confidence":0.9}`, Model: "mock:c", Confidence: 0.9}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: `{"answer":"Final","confidence":0.9}`, Model: "mock:c", Confidence: 0.9}, nil
		},
	}

	ragServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if strings.HasSuffix(r.URL.Path, "/embed") {
			vec := make([]float32, 384)
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": vec})
			return
		}
		if strings.HasSuffix(r.URL.Path, "/retrieve") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"chunks": []map[string]string{{"content": "Chunk"}},
			})
			return
		}
		http.NotFound(w, r)
	}))
	defer ragServer.Close()

	makeHandlers := func(tp telemetry.TracerProvider) *Handlers {
		orch := council.NewOrchestrator([]llm.LLMClient{mockClient1, mockClient2}, mockChairman, 5*time.Second)
		if tp != nil {
			orch.SetTracer(tp)
		}
		cb := cache.NewCircuitBreaker("test", cache.DefaultConfig())
		h := &Handlers{
			RAGServiceURL:          ragServer.URL,
			Council:                orch,
			Cache:                  cache.NewMockRedisCache(cb),
			SemanticCache:          &mockL2SemanticCache{},
			HTTPClient:             &http.Client{Timeout: 5 * time.Second},
			SemanticCacheThreshold: 0.85,
			Tracer:                 tp,
		}
		return h
	}

	b.Run("WithInMemoryTracer", func(b *testing.B) {
		tracer, _ := telemetry.NewTestTracer()
		h := makeHandlers(tracer)
		reqBody := `{"question": "Bench Q", "doc_id": "doc_bench"}`

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			h.HandleQuery(w, req)
		}
	})

	b.Run("WithNoopTracer", func(b *testing.B) {
		tracer := telemetry.NewNoopTracerProvider()
		h := makeHandlers(tracer)
		reqBody := `{"question": "Bench Q", "doc_id": "doc_bench"}`

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			h.HandleQuery(w, req)
		}
	})

	b.Run("WithoutTracer", func(b *testing.B) {
		h := makeHandlers(nil)
		reqBody := `{"question": "Bench Q", "doc_id": "doc_bench"}`

		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()
			h.HandleQuery(w, req)
		}
	})
}
