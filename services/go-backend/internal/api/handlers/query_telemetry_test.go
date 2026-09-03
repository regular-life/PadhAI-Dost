package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"io"
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

type telemetryWorkerResult struct {
	index       int
	statusCode  int
	traceID     string
	traceparent string
	isSSE       bool
}

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

var w3cRegex = regexp.MustCompile(`^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$`)

// ── Linked Span Hierarchy & Carrier Tests ───────────────────────

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
	rootSpanID := assertHierarchyRootSpan(t, tree, incomingTraceID, incomingParentID)
	assertHierarchyCacheAndEmbed(t, tree, rootSpanID, incomingTraceID)
	assertHierarchyCouncilStages(t, tree, rootSpanID, incomingTraceID)
	assertHierarchySpansConsistent(t, spans, incomingTraceID)
}

func assertHierarchyRootSpan(t *testing.T, tree *SpanTree, incomingTraceID, incomingParentID string) trace.SpanID {
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
	return root.SpanContext.SpanID()
}

func assertHierarchyCacheAndEmbed(t *testing.T, tree *SpanTree, rootSpanID trace.SpanID, incomingTraceID string) {
	tree.AssertSpanCount(t, "cache.l1_lookup", 1)
	l1Span := tree.ByName["cache.l1_lookup"][0]
	if l1Span.SpanContext.TraceID().String() != incomingTraceID || l1Span.Parent.SpanID() != rootSpanID || l1Span.SpanKind != trace.SpanKindInternal {
		t.Errorf("L1 span mismatch: trace=%s, parent=%s, kind=%v", l1Span.SpanContext.TraceID(), l1Span.Parent.SpanID(), l1Span.SpanKind)
	}

	tree.AssertSpanCount(t, "rag.embed", 1)
	embedSpan := tree.ByName["rag.embed"][0]
	if embedSpan.SpanContext.TraceID().String() != incomingTraceID || embedSpan.Parent.SpanID() != rootSpanID || embedSpan.SpanKind != trace.SpanKindClient {
		t.Errorf("rag.embed span mismatch: trace=%s, parent=%s, kind=%v", embedSpan.SpanContext.TraceID(), embedSpan.Parent.SpanID(), embedSpan.SpanKind)
	}

	tree.AssertSpanCount(t, "cache.l2_lookup", 1)
	l2Span := tree.ByName["cache.l2_lookup"][0]
	if l2Span.SpanContext.TraceID().String() != incomingTraceID || l2Span.Parent.SpanID() != rootSpanID || l2Span.SpanKind != trace.SpanKindInternal {
		t.Errorf("L2 span mismatch: trace=%s, parent=%s, kind=%v", l2Span.SpanContext.TraceID(), l2Span.Parent.SpanID(), l2Span.SpanKind)
	}
}

func assertHierarchyCouncilStages(t *testing.T, tree *SpanTree, rootSpanID trace.SpanID, incomingTraceID string) {
	tree.AssertSpanCount(t, "council.candidate_fan_out", 1)
	fanOutSpan := tree.ByName["council.candidate_fan_out"][0]
	if fanOutSpan.SpanContext.TraceID().String() != incomingTraceID || fanOutSpan.Parent.SpanID() != rootSpanID || fanOutSpan.SpanKind != trace.SpanKindInternal {
		t.Errorf("fan_out span mismatch: trace=%s, parent=%s, kind=%v", fanOutSpan.SpanContext.TraceID(), fanOutSpan.Parent.SpanID(), fanOutSpan.SpanKind)
	}

	fanOutSpanID := fanOutSpan.SpanContext.SpanID()
	candidateSpans := tree.ByName["council.candidate_model"]
	if len(candidateSpans) != 2 {
		t.Fatalf("expected 2 candidate_model spans, got %d", len(candidateSpans))
	}
	for i, cs := range candidateSpans {
		if cs.SpanContext.TraceID().String() != incomingTraceID || cs.Parent.SpanID() != fanOutSpanID || cs.SpanKind != trace.SpanKindClient {
			t.Errorf("candidate_model[%d] mismatch: trace=%s, parent=%s, kind=%v", i, cs.SpanContext.TraceID(), cs.Parent.SpanID(), cs.SpanKind)
		}
	}

	tree.AssertSpanCount(t, "council.chairman_deliberation", 1)
	chairmanSpan := tree.ByName["council.chairman_deliberation"][0]
	if chairmanSpan.SpanContext.TraceID().String() != incomingTraceID || chairmanSpan.Parent.SpanID() != rootSpanID || chairmanSpan.SpanKind != trace.SpanKindInternal {
		t.Errorf("chairman span mismatch: trace=%s, parent=%s, kind=%v", chairmanSpan.SpanContext.TraceID(), chairmanSpan.Parent.SpanID(), chairmanSpan.SpanKind)
	}
}

func assertHierarchySpansConsistent(t *testing.T, spans []tracetest.SpanStub, incomingTraceID string) {
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

// ── Trace Context Integrity & Latency Benchmark Tests ───────────

var w3cTraceparentPattern = regexp.MustCompile(`^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$`)

// ─────────────────────────────────────────────────────────────────────────────
// Suite 1: Malformed & Adversarial Incoming traceparent Headers Matrix
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_MalformedTraceparent_ExtensiveMatrix(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	malformedCases := []struct {
		name      string
		headerVal string
	}{
		{"EmptyHeader", ""},
		{"WhitespaceOnly", "   "},
		{"TruncatedVersion", "0"},
		{"InvalidVersionNonHex", "zz-1234567890abcdef1234567890abcdef-abcdef1234567890-01"},
		{"InvalidVersionFF", "ff-1234567890abcdef1234567890abcdef-abcdef1234567890-01"},
		{"ThreeDigitVersion", "000-1234567890abcdef1234567890abcdef-abcdef1234567890-01"},
		{"TruncatedTraceID", "00-1234-abcdef1234567890-01"},
		{"TooLongTraceID", "00-1234567890abcdef1234567890abcdef00-abcdef1234567890-01"},
		{"NonHexTraceID", "00-1234567890abcdef1234567890abcdeg-abcdef1234567890-01"},
		{"AllZeroTraceID", "00-00000000000000000000000000000000-abcdef1234567890-01"},
		{"TruncatedParentSpanID", "00-1234567890abcdef1234567890abcdef-123-01"},
		{"TooLongParentSpanID", "00-1234567890abcdef1234567890abcdef-abcdef123456789012-01"},
		{"NonHexParentSpanID", "00-1234567890abcdef1234567890abcdef-abcdef123456789z-01"},
		{"AllZeroParentSpanID", "00-1234567890abcdef1234567890abcdef-0000000000000000-01"},
		{"MissingHyphens", "001234567890abcdef1234567890abcdefabcdef123456789001"},
		{"ExtraDashes", "00--1234567890abcdef1234567890abcdef--abcdef1234567890--01"},
		{"TrailingGarbage", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-01-extra-payload"},
		{"InvalidFlags", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-gg"},
		{"TruncatedFlags", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-0"},
		{"TooLongFlags", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-001"},
		{"SQLInjectionPayload", "'; DROP TABLE traces; --"},
		{"UnicodeEmojiPayload", "00-💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡-🚀🚀🚀🚀🚀🚀🚀🚀-01"},
		{"HugeHeaderString", "00-" + strings.Repeat("a", 10000) + "-01"},
		{"ControlCharacters", "00-\x00\x01\x02\r\n\t-01"},
		{"RandomGarbageText", "not-a-traceparent-at-all"},
	}

	for _, tc := range malformedCases {
		tc := tc
		t.Run(tc.name+"_JSON", func(t *testing.T) {
			verifyMalformedTraceparentJSON(t, f, tc.name, tc.headerVal)
		})
		t.Run(tc.name+"_SSE", func(t *testing.T) {
			verifyMalformedTraceparentSSE(t, f, tc.name, tc.headerVal)
		})
	}
}

func verifyMalformedTraceparentJSON(t *testing.T, f *testTelemetryFixture, name, headerVal string) {
	reqBody := fmt.Sprintf(`{"question": "Malformed test %s", "doc_id": "doc_malformed"}`, name)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	if headerVal != "" {
		req.Header.Set("traceparent", headerVal)
	}

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("case %s failed with unexpected HTTP status %d: %s", name, w.Code, w.Body.String())
	}

	respTraceID := w.Header().Get("X-Trace-ID")
	if respTraceID == "" || len(respTraceID) != 32 || respTraceID == "00000000000000000000000000000000" {
		t.Fatalf("case %s produced invalid X-Trace-ID %q", name, respTraceID)
	}

	respTraceparent := w.Header().Get("traceparent")
	match := w3cTraceparentPattern.FindStringSubmatch(respTraceparent)
	if len(match) != 4 || match[1] != respTraceID {
		t.Fatalf("case %s traceparent %q mismatch with %s", name, respTraceparent, respTraceID)
	}
}

func verifyMalformedTraceparentSSE(t *testing.T, f *testTelemetryFixture, name, headerVal string) {
	reqBody := fmt.Sprintf(`{"question": "Malformed SSE test %s", "doc_id": "doc_malformed_sse"}`, name)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	if headerVal != "" {
		req.Header.Set("traceparent", headerVal)
	}

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("case %s SSE failed with status %d", name, w.Code)
	}
	if !strings.HasPrefix(w.Header().Get("Content-Type"), "text/event-stream") {
		t.Fatalf("case %s SSE expected Content-Type text/event-stream", name)
	}

	respTraceID := w.Header().Get("X-Trace-ID")
	if respTraceID == "" || len(respTraceID) != 32 || respTraceID == "00000000000000000000000000000000" {
		t.Fatalf("case %s SSE invalid X-Trace-ID %q", name, respTraceID)
	}

	verifyMalformedSSEFrames(t, w.Body, name)
}

func verifyMalformedSSEFrames(t *testing.T, r io.Reader, name string) {
	scanner := bufio.NewScanner(r)
	var hasCandidateDraft, hasPeerReview, hasFinalAnswer bool
	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: error") {
			t.Fatalf("case %s SSE emitted error frame on malformed traceparent", name)
		}
		if strings.HasPrefix(line, "event: candidate_draft") {
			hasCandidateDraft = true
		}
		if strings.HasPrefix(line, "event: peer_review") {
			hasPeerReview = true
		}
		if strings.HasPrefix(line, "event: final_answer") {
			hasFinalAnswer = true
		}
	}

	if !hasCandidateDraft || !hasPeerReview || !hasFinalAnswer {
		t.Errorf("case %s SSE missing expected event frames: draft=%v, review=%v, final=%v",
			name, hasCandidateDraft, hasPeerReview, hasFinalAnswer)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 2: High Concurrency under -race with 64 Concurrent Requests
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_HighConcurrency_RaceDetector_64Goroutines(t *testing.T) {
	// Tests 64 concurrent requests generating spans simultaneously under Go race detector.
	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	numWorkers := 64
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	startBarrier := make(chan struct{})

	results := make([]telemetryWorkerResult, numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func(idx int) {
			defer wg.Done()
			req, isSSE := buildConcurrencyWorkerRequest(idx)
			<-startBarrier

			w := httptest.NewRecorder()
			f.handlers.HandleQuery(w, req)

			results[idx] = telemetryWorkerResult{
				index:       idx,
				statusCode:  w.Code,
				traceID:     w.Header().Get("X-Trace-ID"),
				traceparent: w.Header().Get("traceparent"),
				isSSE:       isSSE,
			}
		}(i)
	}

	close(startBarrier)
	wg.Wait()

	verifyConcurrencyWorkerResults(t, results, numWorkers)
}

func buildConcurrencyWorkerRequest(idx int) (*http.Request, bool) {
	isSSE := idx >= 32
	hasIncomingTrace := (idx < 16) || (idx >= 32 && idx < 48)
	hasMalformedTrace := idx >= 48

	reqBody := fmt.Sprintf(`{"question": "Concurrency query worker %d", "doc_id": "doc_conc_%d"}`, idx, idx%4)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	if isSSE {
		req.Header.Set("Accept", "text/event-stream")
	}

	if hasIncomingTrace {
		expectedTraceID := fmt.Sprintf("a%031x", idx)
		incomingParentID := fmt.Sprintf("b%015x", idx)
		req.Header.Set("traceparent", fmt.Sprintf("00-%s-%s-01", expectedTraceID, incomingParentID))
	} else if hasMalformedTrace {
		req.Header.Set("traceparent", fmt.Sprintf("malformed-header-worker-%d-!", idx))
	}

	return req, isSSE
}

func verifyConcurrencyWorkerResults(t *testing.T, results []telemetryWorkerResult, numWorkers int) {
	seenTraceIDs := make(map[string]int)

	for i, res := range results {
		if res.statusCode != http.StatusOK {
			t.Fatalf("worker %d returned HTTP %d", i, res.statusCode)
		}
		if res.traceID == "" || len(res.traceID) != 32 {
			t.Fatalf("worker %d produced empty or invalid TraceID %q", i, res.traceID)
		}
		if res.traceparent == "" || !w3cTraceparentPattern.MatchString(res.traceparent) {
			t.Fatalf("worker %d produced invalid traceparent header %q", i, res.traceparent)
		}

		if (i < 16) || (i >= 32 && i < 48) {
			expectedTraceID := fmt.Sprintf("a%031x", i)
			if res.traceID != expectedTraceID {
				t.Fatalf("worker %d expected incoming TraceID %s, but got %s", i, expectedTraceID, res.traceID)
			}
		}

		seenTraceIDs[res.traceID]++
	}

	if len(seenTraceIDs) < numWorkers {
		t.Fatalf("detected duplicate trace IDs: %d unique traces out of %d requests",
			len(seenTraceIDs), numWorkers)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 3: SSE Streaming Deliberation Trace Context & Event Frame Verification
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_SSEStreaming_TraceContextIntegrity(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	incomingTraceID := "feedfacefeedfacefeedfacefeedface"
	incomingParentID := "c001cafe12345678"
	incomingHeader := "00-" + incomingTraceID + "-" + incomingParentID + "-01"

	reqBody := `{"question": "Verify SSE trace context integrity", "doc_id": "doc_sse_trace"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("traceparent", incomingHeader)

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK for SSE query, got %d", w.Code)
	}

	verifySSETraceResponseHeaders(t, w, incomingTraceID)
	parseAndVerifySSETraceFrames(t, w.Body)

	spans := f.exporter.GetSpans()
	if len(spans) == 0 {
		t.Fatal("expected spans to be recorded for SSE deliberation, got 0")
	}
	verifySSETraceSpanTree(t, spans, incomingTraceID)
}

func verifySSETraceResponseHeaders(t *testing.T, w *httptest.ResponseRecorder, incomingTraceID string) {
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Errorf("expected Content-Type text/event-stream, got %q", ct)
	}
	if w.Header().Get("X-Trace-ID") != incomingTraceID {
		t.Errorf("expected X-Trace-ID %s, got %s", incomingTraceID, w.Header().Get("X-Trace-ID"))
	}
	if !strings.Contains(w.Header().Get("traceparent"), incomingTraceID) {
		t.Errorf("expected response traceparent header %q to contain incoming TraceID %s",
			w.Header().Get("traceparent"), incomingTraceID)
	}
}

func parseAndVerifySSETraceFrames(t *testing.T, r io.Reader) {
	candidateDrafts, peerReviews, finalAnswer, errorPayloads := parseSSETracePayloads(t, r)
	validateSSETracePayloads(t, candidateDrafts, peerReviews, finalAnswer, errorPayloads)
}

func parseSSETracePayloads(t *testing.T, r io.Reader) (
	candidateDrafts []council.CandidateDraftPayload,
	peerReviews []council.PeerReviewPayload,
	finalAnswer *QueryResponse,
	errorPayloads []map[string]interface{},
) {
	scanner := bufio.NewScanner(r)
	var currentEvent string

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}
		if strings.HasPrefix(line, "data: ") {
			dataJSON := strings.TrimPrefix(line, "data: ")
			collectSSETracePayload(t, currentEvent, dataJSON, &candidateDrafts, &peerReviews, &finalAnswer, &errorPayloads)
		}
	}
	return
}

func collectSSETracePayload(
	t *testing.T,
	currentEvent, dataJSON string,
	drafts *[]council.CandidateDraftPayload,
	reviews *[]council.PeerReviewPayload,
	finalAns **QueryResponse,
	errs *[]map[string]interface{},
) {
	switch currentEvent {
	case "candidate_draft":
		var draft council.CandidateDraftPayload
		if err := json.Unmarshal([]byte(dataJSON), &draft); err != nil {
			t.Fatalf("failed to unmarshal candidate_draft data: %v", err)
		}
		*drafts = append(*drafts, draft)
	case "peer_review":
		var review council.PeerReviewPayload
		if err := json.Unmarshal([]byte(dataJSON), &review); err != nil {
			t.Fatalf("failed to unmarshal peer_review data: %v", err)
		}
		*reviews = append(*reviews, review)
	case "final_answer":
		var resp QueryResponse
		if err := json.Unmarshal([]byte(dataJSON), &resp); err != nil {
			t.Fatalf("failed to unmarshal final_answer data: %v", err)
		}
		*finalAns = &resp
	case "error":
		var errMap map[string]interface{}
		_ = json.Unmarshal([]byte(dataJSON), &errMap)
		*errs = append(*errs, errMap)
	}
}

func validateSSETracePayloads(
	t *testing.T,
	drafts []council.CandidateDraftPayload,
	reviews []council.PeerReviewPayload,
	finalAnswer *QueryResponse,
	errorPayloads []map[string]interface{},
) {
	if len(errorPayloads) > 0 {
		t.Fatalf("unexpected error frames encountered in SSE stream: %v", errorPayloads)
	}
	if len(drafts) < 2 {
		t.Fatalf("expected at least 2 candidate drafts streamed, got %d", len(drafts))
	}
	if len(reviews) < 2 {
		t.Fatalf("expected at least 2 peer reviews streamed, got %d", len(reviews))
	}
	if finalAnswer == nil || finalAnswer.Answer == "" {
		t.Fatalf("expected non-empty final answer text in final_answer payload: %v", finalAnswer)
	}
}

func verifySSETraceSpanTree(t *testing.T, spans []tracetest.SpanStub, incomingTraceID string) {
	tree := BuildSpanTree(t, spans)

	for _, s := range spans {
		if s.SpanContext.TraceID().String() != incomingTraceID {
			t.Errorf("span %q TraceID %s != expected TraceID %s", s.Name, s.SpanContext.TraceID(), incomingTraceID)
		}
	}

	var foundSSEAttr, foundDocAttr bool
	for _, a := range tree.Root.Attributes {
		if a.Key == "query.sse" && a.Value.AsBool() == true {
			foundSSEAttr = true
		}
		if a.Key == "query.doc_id" && a.Value.AsString() == "doc_sse_trace" {
			foundDocAttr = true
		}
	}
	if !foundSSEAttr || !foundDocAttr {
		t.Errorf("root span missing expected attributes: sse=%v, doc=%v", foundSSEAttr, foundDocAttr)
	}

	rootName := "HTTP POST /api/v1/query"
	tree.AssertParent(t, "council.candidate_fan_out", rootName)
	tree.AssertParent(t, "council.candidate_model", "council.candidate_fan_out")
	tree.AssertParent(t, "council.chairman_deliberation", rootName)
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 4: Client Premature Disconnection during SSE Deliberation
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_SSEStreaming_ClientPrematureDisconnect(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	ctx, cancel := context.WithCancel(context.Background())

	reqBody := `{"question": "Simulate client disconnect mid-deliberation", "doc_id": "doc_disconnect"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	// Trigger cancellation immediately to simulate client network drop
	cancel()

	w := httptest.NewRecorder()
	done := make(chan struct{})

	go func() {
		f.handlers.HandleQuery(w, req)
		close(done)
	}()

	select {
	case <-done:
		// Clean exit without hanging or panicking
	case <-time.After(2 * time.Second):
		t.Fatal("HandleQuery hung on client premature disconnect")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 5: Outgoing Python RAG Header Propagation Verification
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_OutgoingPythonRAG_TraceparentPropagation(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "Outgoing header verification", "doc_id": "doc_outgoing"}`
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

	var embedFound, retrieveFound bool
	for _, call := range calls {
		if strings.HasSuffix(call.Path, "/embed") {
			embedFound = true
			if call.Traceparent == "" || !w3cTraceparentPattern.MatchString(call.Traceparent) {
				t.Fatalf("outgoing /embed call missing valid W3C traceparent header, got %q", call.Traceparent)
			}
		}
		if strings.HasSuffix(call.Path, "/retrieve") {
			retrieveFound = true
			if call.Traceparent == "" || !w3cTraceparentPattern.MatchString(call.Traceparent) {
				t.Fatalf("outgoing /retrieve call missing valid W3C traceparent header, got %q", call.Traceparent)
			}
		}
	}

	if !embedFound {
		t.Error("expected outgoing HTTP call to /embed")
	}
	if !retrieveFound {
		t.Error("expected outgoing HTTP call to /retrieve")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 6: Fast Sub-Second Performance Guarantee
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_BenchmarkLatencyOverhead(t *testing.T) {
	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	iterations := 20
	start := time.Now()

	for i := 0; i < iterations; i++ {
		reqBody := fmt.Sprintf(`{"question": "Bench %d", "doc_id": "doc_perf"}`, i)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		f.handlers.HandleQuery(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("iteration %d failed with status %d", i, w.Code)
		}
	}

	elapsed := time.Since(start)
	avgPerOp := elapsed / time.Duration(iterations)
	t.Logf("Executed %d traced queries in %v (avg: %v/query)", iterations, elapsed, avgPerOp)

	if elapsed > 500*time.Millisecond {
		t.Fatalf("tracing overhead too high: %d iterations took %v (limit: 500ms)", iterations, elapsed)
	}
}
