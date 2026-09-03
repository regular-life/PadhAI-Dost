package handlers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	"go.opentelemetry.io/otel/trace"

	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
	"github.com/regular-life/CouncilAI/go-backend/internal/telemetry"
)

type capturedHTTPCall struct {
	Path        string
	Traceparent string
}

type testTelemetryFixture struct {
	handlers      *Handlers
	exporter      *tracetest.InMemoryExporter
	tracer        telemetry.TracerProvider
	ragServer     *httptest.Server
	l1Cache       *cache.RedisCache
	l2Cache       *mockL2SemanticCache
	recordedCalls []capturedHTTPCall
	callsMu       sync.Mutex
}

func setupTelemetryFixture(t *testing.T) *testTelemetryFixture {
	t.Helper()

	fixture := &testTelemetryFixture{}

	// 1. Mock RAG Server intercepting /embed and /retrieve with W3C header capture
	fixture.ragServer = httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fixture.callsMu.Lock()
		fixture.recordedCalls = append(fixture.recordedCalls, capturedHTTPCall{
			Path:        r.URL.Path,
			Traceparent: r.Header.Get("traceparent"),
		})
		fixture.callsMu.Unlock()

		if strings.HasSuffix(r.URL.Path, "/embed") {
			vec := make([]float32, 384)
			for i := range vec {
				vec[i] = 0.01
			}
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{"embedding": vec})
			return
		}
		if strings.HasSuffix(r.URL.Path, "/retrieve") {
			w.Header().Set("Content-Type", "application/json")
			_ = json.NewEncoder(w).Encode(map[string]interface{}{
				"chunks": []map[string]string{{"content": "Grounding document chunk"}},
			})
			return
		}
		http.NotFound(w, r)
	}))

	// 2. In-Memory Tracer
	tracer, exporter := telemetry.NewTestTracer()
	fixture.tracer = tracer
	fixture.exporter = exporter

	// 3. Mock Caches
	cb := cache.NewCircuitBreaker("test-breaker", cache.DefaultConfig())
	fixture.l1Cache = cache.NewMockRedisCache(cb)
	fixture.l2Cache = &mockL2SemanticCache{}

	// 4. Mock Council LLMs (at least 2 members for peer review and chairman deliberation)
	mockClient1 := &handlerMockLLMClient{
		Name: "mock:model-1",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Candidate answer from model 1", Model: "mock:model-1"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: A, B\nREASONING: Best", Model: "mock:model-1"}, nil
		},
	}
	mockClient2 := &handlerMockLLMClient{
		Name: "mock:model-2",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Candidate answer from model 2", Model: "mock:model-2"}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: B, A\nREASONING: Great", Model: "mock:model-2"}, nil
		},
	}
	mockChairman := &handlerMockLLMClient{
		Name: "mock:chairman",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer:     `{"answer":"Consensus final answer","confidence":0.92,"source":"mock:chairman"}`,
				Model:      "mock:chairman",
				Confidence: 0.92,
			}, nil
		},
		GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{
				Answer:     `{"answer":"Consensus final answer","confidence":0.92,"source":"mock:chairman"}`,
				Model:      "mock:chairman",
				Confidence: 0.92,
			}, nil
		},
	}

	orch := council.NewOrchestrator([]llm.LLMClient{mockClient1, mockClient2}, mockChairman, 5*time.Second)
	orch.SetTracer(tracer)

	fixture.handlers = &Handlers{
		RAGServiceURL:          fixture.ragServer.URL,
		Council:                orch,
		Cache:                  fixture.l1Cache,
		SemanticCache:          fixture.l2Cache,
		HTTPClient:             &http.Client{Timeout: 5 * time.Second},
		SemanticCacheThreshold: 0.85,
		Tracer:                 tracer,
	}

	return fixture
}

// SpanTree provides graph querying and parent-child assertions over recorded spans.
type SpanTree struct {
	Root     tracetest.SpanStub
	Spans    []tracetest.SpanStub
	BySpanID map[trace.SpanID]tracetest.SpanStub
	ByName   map[string][]tracetest.SpanStub
}

func BuildSpanTree(t *testing.T, spans []tracetest.SpanStub) *SpanTree {
	t.Helper()
	if len(spans) == 0 {
		t.Fatal("cannot build SpanTree from empty span list")
	}

	tree := &SpanTree{
		Spans:    spans,
		BySpanID: make(map[trace.SpanID]tracetest.SpanStub),
		ByName:   make(map[string][]tracetest.SpanStub),
	}

	var rootCandidates []tracetest.SpanStub
	for _, s := range spans {
		tree.BySpanID[s.SpanContext.SpanID()] = s
		tree.ByName[s.Name] = append(tree.ByName[s.Name], s)
		// Server root span has SpanKindServer or invalid parent ID
		if !s.Parent.SpanID().IsValid() || s.SpanKind == trace.SpanKindServer {
			rootCandidates = append(rootCandidates, s)
		}
	}

	if len(rootCandidates) == 0 {
		t.Fatalf("no root span identified in %d recorded spans", len(spans))
	}
	tree.Root = rootCandidates[0]
	return tree
}

func (tree *SpanTree) AssertParent(t *testing.T, childName, expectedParentName string) {
	t.Helper()
	children := tree.ByName[childName]
	if len(children) == 0 {
		t.Fatalf("child span with name %q was not found", childName)
	}
	parents := tree.ByName[expectedParentName]
	if len(parents) == 0 {
		t.Fatalf("parent span with name %q was not found", expectedParentName)
	}

	parentIDs := make(map[trace.SpanID]bool)
	for _, p := range parents {
		parentIDs[p.SpanContext.SpanID()] = true
	}

	for _, child := range children {
		if !parentIDs[child.Parent.SpanID()] {
			t.Errorf("span %q (ID=%s) expected parent %q, but got ParentSpanID=%s",
				childName, child.SpanContext.SpanID(), expectedParentName, child.Parent.SpanID())
		}
		if child.SpanContext.TraceID() != tree.Root.SpanContext.TraceID() {
			t.Errorf("span %q TraceID %s does not match root TraceID %s",
				childName, child.SpanContext.TraceID(), tree.Root.SpanContext.TraceID())
		}
	}
}

func (tree *SpanTree) AssertSpanCount(t *testing.T, name string, expectedCount int) {
	t.Helper()
	actual := len(tree.ByName[name])
	if actual != expectedCount {
		t.Errorf("expected %d occurrences of span %q, got %d", expectedCount, name, actual)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Incoming W3C traceparent header propagation to root span
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_IncomingTraceparent_PropagatedToRoot(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	incomingTrace := "1234567890abcdef1234567890abcdef"
	incomingParent := "abcdef1234567890"
	incomingHeader := "00-" + incomingTrace + "-" + incomingParent + "-01"

	reqBody := `{"question": "How does CouncilAI distribute spans?", "doc_id": "doc1"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("traceparent", incomingHeader)

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	spans := f.exporter.GetSpans()
	if len(spans) == 0 {
		t.Fatal("expected spans to be recorded, got 0")
	}

	tree := BuildSpanTree(t, spans)

	// Verify Root Span
	if tree.Root.SpanContext.TraceID().String() != incomingTrace {
		t.Errorf("expected Root TraceID %s, got %s", incomingTrace, tree.Root.SpanContext.TraceID())
	}
	if tree.Root.Parent.SpanID().String() != incomingParent {
		t.Errorf("expected Root ParentSpanID %s, got %s", incomingParent, tree.Root.Parent.SpanID())
	}

	// Verify all spans share incoming TraceID
	for _, s := range spans {
		if s.SpanContext.TraceID().String() != incomingTrace {
			t.Errorf("span %q TraceID %s does not match incoming TraceID %s",
				s.Name, s.SpanContext.TraceID(), incomingTrace)
		}
	}

	// Verify response traceparent header matches active trace
	respTraceparent := w.Header().Get("traceparent")
	if !strings.Contains(respTraceparent, incomingTrace) {
		t.Errorf("expected response traceparent header %q to contain incoming TraceID %s",
			respTraceparent, incomingTrace)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Missing traceparent header generates new root TraceID
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_NoIncomingTraceparent_GeneratesNewRoot(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "Explain tracing generation", "doc_id": "doc2"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	if !tree.Root.SpanContext.TraceID().IsValid() {
		t.Fatal("expected valid generated TraceID on root span")
	}
	if tree.Root.Parent.SpanID().IsValid() {
		t.Errorf("expected invalid ParentSpanID on newly generated root, got %s", tree.Root.Parent.SpanID())
	}

	// Verify response header contains the new trace ID
	respTraceID := w.Header().Get("X-Trace-ID")
	if respTraceID != tree.Root.SpanContext.TraceID().String() {
		t.Errorf("expected response X-Trace-ID %s, got %s", tree.Root.SpanContext.TraceID(), respTraceID)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Full pipeline child spans and parent-child link assertions
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_FullPipeline_SpanHierarchy(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "What is the full deliberation flow?", "doc_id": "doc_pipeline"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d: %s", w.Code, w.Body.String())
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	rootName := "HTTP POST /api/v1/query"
	tree.AssertParent(t, "cache.l1_lookup", rootName)
	tree.AssertParent(t, "rag.embed", rootName)
	tree.AssertParent(t, "cache.l2_lookup", rootName)
	tree.AssertParent(t, "council.candidate_fan_out", rootName)
	tree.AssertParent(t, "council.candidate_model", "council.candidate_fan_out")
	tree.AssertParent(t, "council.chairman_deliberation", rootName)

	// Verify Root Attributes
	rootAttrs := make(map[string]interface{})
	for _, a := range tree.Root.Attributes {
		rootAttrs[string(a.Key)] = a.Value.AsInterface()
	}
	if rootAttrs["http.method"] != "POST" {
		t.Errorf("expected root attribute http.method=POST, got %v", rootAttrs["http.method"])
	}
	if rootAttrs["query.doc_id"] != "doc_pipeline" {
		t.Errorf("expected root attribute query.doc_id=doc_pipeline, got %v", rootAttrs["query.doc_id"])
	}
	if rootAttrs["query.sse"] != false {
		t.Errorf("expected root attribute query.sse=false, got %v", rootAttrs["query.sse"])
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: L1 exact cache hit short-circuits pipeline spans
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_L1CacheHit_ShortCircuitsPipeline(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	// Seed L1 cache
	cachedResp := QueryResponse{
		Answer:   "L1 cached fast answer",
		Source:   "redis_exact_cache_hit",
		CacheHit: true,
	}
	cachedJSON, _ := json.Marshal(cachedResp)
	f.l1Cache.SetMockData("cache:doc_l1:cached query", string(cachedJSON))

	reqBody := `{"question": "cached query", "doc_id": "doc_l1"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d", w.Code)
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	// Exactly 2 spans expected: Root and L1 Lookup
	if len(spans) != 2 {
		t.Fatalf("expected exactly 2 spans on L1 hit, got %d", len(spans))
	}

	tree.AssertSpanCount(t, "HTTP POST /api/v1/query", 1)
	tree.AssertSpanCount(t, "cache.l1_lookup", 1)
	tree.AssertParent(t, "cache.l1_lookup", "HTTP POST /api/v1/query")

	// Verify downstream stages were NOT executed
	tree.AssertSpanCount(t, "rag.embed", 0)
	tree.AssertSpanCount(t, "cache.l2_lookup", 0)
	tree.AssertSpanCount(t, "council.candidate_fan_out", 0)
	tree.AssertSpanCount(t, "council.chairman_deliberation", 0)

	// Verify cache.hit attribute
	l1Span := tree.ByName["cache.l1_lookup"][0]
	hitFound := false
	for _, a := range l1Span.Attributes {
		if a.Key == "cache.hit" && a.Value.AsBool() == true {
			hitFound = true
			break
		}
	}
	if !hitFound {
		t.Error("expected cache.hit=true attribute on cache.l1_lookup span")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: L2 semantic cache hit short-circuits council deliberation
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_L2CacheHit_ShortCircuitsDeliberation(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	// Configure L2 semantic cache hit
	f.l2Cache.found = true
	f.l2Cache.response = QueryResponse{
		Answer:   "L2 semantic cached answer",
		Source:   "semantic_cache_hit",
		CacheHit: true,
	}

	reqBody := `{"question": "semantic hit question", "doc_id": "doc_l2"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d", w.Code)
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	// Exactly 4 spans expected: Root, L1, Embed, L2
	if len(spans) != 4 {
		t.Fatalf("expected exactly 4 spans on L2 hit, got %d", len(spans))
	}

	rootName := "HTTP POST /api/v1/query"
	tree.AssertParent(t, "cache.l1_lookup", rootName)
	tree.AssertParent(t, "rag.embed", rootName)
	tree.AssertParent(t, "cache.l2_lookup", rootName)

	// Council deliberation MUST NOT be invoked
	tree.AssertSpanCount(t, "council.candidate_fan_out", 0)
	tree.AssertSpanCount(t, "council.chairman_deliberation", 0)
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Outgoing HTTP client call contains valid W3C traceparent header
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_OutgoingHTTP_TraceparentInjected(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "verify outgoing traceparent header", "doc_id": "doc_http"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	f.callsMu.Lock()
	calls := f.recordedCalls
	f.callsMu.Unlock()

	if len(calls) == 0 {
		t.Fatal("expected at least 1 outgoing HTTP call to mock RAG server, got 0")
	}

	var embedCall *capturedHTTPCall
	for i, c := range calls {
		if strings.HasSuffix(c.Path, "/embed") {
			embedCall = &calls[i]
			break
		}
	}

	if embedCall == nil {
		t.Fatal("no outgoing /embed call was recorded by mock RAG server")
	}

	if embedCall.Traceparent == "" {
		t.Fatal("outgoing HTTP call to /embed did NOT contain a 'traceparent' header")
	}

	matches := regexp.MustCompile(`^00-([0-9a-f]{32})-([0-9a-f]{16})-01$`).FindStringSubmatch(embedCall.Traceparent)
	if len(matches) != 3 {
		t.Fatalf("outgoing traceparent header %q invalid format", embedCall.Traceparent)
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	outgoingTraceID := matches[1]
	outgoingSpanID := matches[2]

	if outgoingTraceID != tree.Root.SpanContext.TraceID().String() {
		t.Errorf("outgoing trace ID %s != root trace ID %s", outgoingTraceID, tree.Root.SpanContext.TraceID())
	}

	// Verify that outgoing span ID belongs to the rag.embed client span
	embedSpans := tree.ByName["rag.embed"]
	if len(embedSpans) == 0 {
		t.Fatal("rag.embed span not recorded in exporter")
	}
	if outgoingSpanID != embedSpans[0].SpanContext.SpanID().String() {
		t.Errorf("outgoing span ID %s does not match rag.embed SpanID %s",
			outgoingSpanID, embedSpans[0].SpanContext.SpanID())
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 7: SSE Streaming Deliberation preserves span hierarchy
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_SSEStreaming_SpanHierarchy(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "Stream deliberation spans", "doc_id": "doc_sse"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d", w.Code)
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	// Root span must have query.sse=true
	var sseAttrFound bool
	for _, a := range tree.Root.Attributes {
		if a.Key == "query.sse" && a.Value.AsBool() == true {
			sseAttrFound = true
			break
		}
	}
	if !sseAttrFound {
		t.Error("expected query.sse=true on root span for SSE request")
	}

	// Deliberation spans present
	tree.AssertSpanCount(t, "council.candidate_fan_out", 1)
	tree.AssertSpanCount(t, "council.chairman_deliberation", 1)
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 8: Sub-second execution benchmark guarantee (< 0.2s)
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_SubSecondPerformanceGuarantee(t *testing.T) {
	// Do not run in parallel with heavy bcrypt hashing tests in the same package
	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	iterations := 10
	start := time.Now()

	for i := 0; i < iterations; i++ {
		reqBody := `{"question": "Benchmark iteration", "doc_id": "doc_bench"}`
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		f.handlers.HandleQuery(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("iteration %d failed with status %d", i, w.Code)
		}
	}

	elapsed := time.Since(start)
	t.Logf("Executed %d full queries with span exports in %v (avg: %v/query)",
		iterations, elapsed, elapsed/time.Duration(iterations))

	// Guarantee: Tracer overhead is negligible, running all iterations in well under 500ms
	if elapsed > 500*time.Millisecond {
		t.Errorf("expected %d queries to execute in < 500ms, took %v", iterations, elapsed)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 9: Malformed traceparent header graceful fallback
// ─────────────────────────────────────────────────────────────────────────────
func TestQueryTelemetry_MalformedTraceparent_GracefulFallback(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "Malformed header test", "doc_id": "doc_malformed"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("traceparent", "invalid-garbage-value-not-hex")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d", w.Code)
	}

	spans := f.exporter.GetSpans()
	tree := BuildSpanTree(t, spans)

	if !tree.Root.SpanContext.TraceID().IsValid() {
		t.Fatal("expected valid generated TraceID on malformed header fallback")
	}
	if tree.Root.Parent.SpanID().IsValid() {
		t.Errorf("expected invalid ParentSpanID on malformed header fallback, got %s", tree.Root.Parent.SpanID())
	}
}
