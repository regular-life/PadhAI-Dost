package telemetry_test

import (
	"context"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"

	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"

	"github.com/regular-life/CouncilAI/go-backend/internal/telemetry"
)

var traceparentRegex = regexp.MustCompile(`^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$`)

func TestTracer_NewAndShutdown(t *testing.T) {
	t.Parallel()

	tracer, exporter := telemetry.NewTestTracer()
	if tracer == nil {
		t.Fatal("expected non-nil tracer from NewTestTracer")
	}
	if exporter == nil {
		t.Fatal("expected non-nil exporter from NewTestTracer")
	}

	ctx, span := tracer.StartSpan(context.Background(), "lifecycle-span")
	span.End()

	spans := exporter.GetSpans()
	if len(spans) != 1 {
		t.Fatalf("expected 1 recorded span, got %d", len(spans))
	}

	shutdownCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()

	if err := tracer.Shutdown(shutdownCtx); err != nil {
		t.Fatalf("expected clean shutdown, got err: %v", err)
	}
}

func TestTracer_StartSpan_Basic(t *testing.T) {
	t.Parallel()

	tracer, exporter := telemetry.NewTestTracer()

	ctx := context.Background()
	startTime := time.Now()
	_, span := tracer.StartSpan(ctx, "basic-span", trace.WithSpanKind(trace.SpanKindInternal))

	sc := span.SpanContext()
	if !sc.IsValid() {
		t.Fatal("expected valid SpanContext")
	}
	if !sc.TraceID().IsValid() {
		t.Fatal("expected valid non-zero TraceID")
	}
	if !sc.SpanID().IsValid() {
		t.Fatal("expected valid non-zero SpanID")
	}

	span.End()

	stubs := exporter.GetSpans()
	if len(stubs) != 1 {
		t.Fatalf("expected 1 span stub, got %d", len(stubs))
	}

	stub := stubs[0]
	if stub.Name != "basic-span" {
		t.Errorf("expected name 'basic-span', got %q", stub.Name)
	}
	if stub.SpanKind != trace.SpanKindInternal {
		t.Errorf("expected SpanKindInternal, got %v", stub.SpanKind)
	}
	if stub.SpanContext.TraceID() != sc.TraceID() {
		t.Errorf("trace ID mismatch: stub=%s, span=%s", stub.SpanContext.TraceID(), sc.TraceID())
	}
	if stub.SpanContext.SpanID() != sc.SpanID() {
		t.Errorf("span ID mismatch: stub=%s, span=%s", stub.SpanContext.SpanID(), sc.SpanID())
	}
	if stub.StartTime.Before(startTime) {
		t.Errorf("stub start time %v is before test start %v", stub.StartTime, startTime)
	}
}

func TestTracer_Attributes_Status_Events(t *testing.T) {
	t.Parallel()

	tracer, exporter := telemetry.NewTestTracer()

	_, span := tracer.StartSpan(context.Background(), "annotated-operation")
	span.SetAttributes(
		attribute.String("cache.type", "exact"),
		attribute.Bool("cache.hit", true),
		attribute.Int64("cache.ttl_seconds", 3600),
		attribute.Float64("cache.similarity", 0.965),
	)
	span.SetStatus(codes.Error, "circuit breaker tripped")
	span.AddEvent("cache_miss_fallback", trace.WithAttributes(
		attribute.String("fallback.strategy", "llm_deliberation"),
	))
	span.End()

	stubs := exporter.GetSpans()
	if len(stubs) != 1 {
		t.Fatalf("expected 1 span stub, got %d", len(stubs))
	}

	stub := stubs[0]

	attrMap := make(map[string]interface{})
	for _, a := range stub.Attributes {
		attrMap[string(a.Key)] = a.Value.AsInterface()
	}

	if attrMap["cache.type"] != "exact" {
		t.Errorf("expected cache.type='exact', got %v", attrMap["cache.type"])
	}
	if attrMap["cache.hit"] != true {
		t.Errorf("expected cache.hit=true, got %v", attrMap["cache.hit"])
	}
	if attrMap["cache.ttl_seconds"] != int64(3600) {
		t.Errorf("expected cache.ttl_seconds=3600, got %v", attrMap["cache.ttl_seconds"])
	}
	if attrMap["cache.similarity"] != 0.965 {
		t.Errorf("expected cache.similarity=0.965, got %v", attrMap["cache.similarity"])
	}

	if stub.Status.Code != codes.Error {
		t.Errorf("expected status code Error, got %v", stub.Status.Code)
	}
	if stub.Status.Description != "circuit breaker tripped" {
		t.Errorf("expected description 'circuit breaker tripped', got %q", stub.Status.Description)
	}

	if len(stub.Events) != 1 {
		t.Fatalf("expected 1 event, got %d", len(stub.Events))
	}
	if stub.Events[0].Name != "cache_miss_fallback" {
		t.Errorf("expected event name 'cache_miss_fallback', got %q", stub.Events[0].Name)
	}
}

func TestTracer_W3C_InjectHTTPHeaders(t *testing.T) {
	t.Parallel()

	tracer, _ := telemetry.NewTestTracer()

	ctx, span := tracer.StartSpan(context.Background(), "outgoing-http-call", trace.WithSpanKind(trace.SpanKindClient))
	defer span.End()

	req := httptest.NewRequest(http.MethodPost, "http://python-rag:8001/embed", strings.NewReader(`{"text":"hello"}`))
	tracer.InjectHTTPHeaders(ctx, req)

	traceparent := req.Header.Get("traceparent")
	if traceparent == "" {
		t.Fatal("expected 'traceparent' header to be set, but was empty")
	}

	matches := traceparentRegex.FindStringSubmatch(traceparent)
	if len(matches) != 4 {
		t.Fatalf("invalid traceparent format %q (must match 00-<32hex>-<16hex>-01)", traceparent)
	}

	gotTraceID := matches[1]
	gotSpanID := matches[2]
	gotFlags := matches[3]

	expectedTraceID := span.SpanContext().TraceID().String()
	expectedSpanID := span.SpanContext().SpanID().String()

	if gotTraceID != expectedTraceID {
		t.Errorf("traceparent trace ID mismatch: got %s, expected %s", gotTraceID, expectedTraceID)
	}
	if gotSpanID != expectedSpanID {
		t.Errorf("traceparent span ID mismatch: got %s, expected %s", gotSpanID, expectedSpanID)
	}
	if gotFlags != "01" {
		t.Errorf("expected sampled flag '01', got %q", gotFlags)
	}
}

func TestTracer_W3C_ExtractHTTPHeaders_Valid(t *testing.T) {
	t.Parallel()

	tracer, exporter := telemetry.NewTestTracer()

	incomingTraceID := "4bf92f3577b34da6a3ce929d0e0e4736"
	incomingParentSpanID := "00f067aa0ba902b7"
	incomingHeader := "00-" + incomingTraceID + "-" + incomingParentSpanID + "-01"

	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", nil)
	req.Header.Set("traceparent", incomingHeader)

	extractedCtx := tracer.ExtractHTTPHeaders(context.Background(), req)
	_, rootSpan := tracer.StartSpan(extractedCtx, "server-root", trace.WithSpanKind(trace.SpanKindServer))
	rootSpan.End()

	stubs := exporter.GetSpans()
	if len(stubs) != 1 {
		t.Fatalf("expected 1 span stub, got %d", len(stubs))
	}

	stub := stubs[0]
	if stub.SpanContext.TraceID().String() != incomingTraceID {
		t.Errorf("root span TraceID mismatch: got %s, expected %s",
			stub.SpanContext.TraceID().String(), incomingTraceID)
	}
	if stub.Parent.SpanID().String() != incomingParentSpanID {
		t.Errorf("root span ParentSpanID mismatch: got %s, expected %s",
			stub.Parent.SpanID().String(), incomingParentSpanID)
	}
	if stub.Parent.TraceID().String() != incomingTraceID {
		t.Errorf("root span Parent TraceID mismatch: got %s, expected %s",
			stub.Parent.TraceID().String(), incomingTraceID)
	}
	if stub.SpanContext.SpanID().String() == incomingParentSpanID {
		t.Errorf("root span SpanID must be newly generated, should not match parent ID %s", incomingParentSpanID)
	}
}

func TestTracer_W3C_ExtractHTTPHeaders_MalformedOrEmpty(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name        string
		headerVal   string
		expectTrace bool
	}{
		{"EmptyHeader", "", true},
		{"TruncatedHex", "00-1234-5678-01", true},
		{"NonHexCharacters", "00-zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz-0000000000000000-01", true},
		{"WrongVersion", "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01", true},
		{"MissingDelimiters", "004bf92f3577b34da6a3ce929d0e0e473600f067aa0ba902b701", true},
	}

	for _, tc := range tests {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			tracer, exporter := telemetry.NewTestTracer()

			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", nil)
			if tc.headerVal != "" {
				req.Header.Set("traceparent", tc.headerVal)
			}

			extractedCtx := tracer.ExtractHTTPHeaders(context.Background(), req)
			_, span := tracer.StartSpan(extractedCtx, "fallback-span")
			span.End()

			stubs := exporter.GetSpans()
			if len(stubs) != 1 {
				t.Fatalf("expected 1 span stub, got %d", len(stubs))
			}

			stub := stubs[0]
			if !stub.SpanContext.TraceID().IsValid() {
				t.Error("expected valid generated TraceID on malformed/missing header")
			}
			if stub.Parent.SpanID().IsValid() {
				t.Errorf("expected invalid ParentSpanID on fallback, got %s", stub.Parent.SpanID())
			}
		})
	}
}

func TestTracer_ParentChildHierarchy(t *testing.T) {
	t.Parallel()

	tracer, exporter := telemetry.NewTestTracer()

	ctx := context.Background()
	ctx1, parent := tracer.StartSpan(ctx, "span-parent")
	ctx2, child := tracer.StartSpan(ctx1, "span-child")
	_, grandchild := tracer.StartSpan(ctx2, "span-grandchild")

	grandchild.End()
	child.End()
	parent.End()

	stubs := exporter.GetSpans()
	if len(stubs) != 3 {
		t.Fatalf("expected 3 spans, got %d", len(stubs))
	}

	var parentStub, childStub, grandchildStub *trace.SpanContext
	parentSpanID := parent.SpanContext().SpanID()
	childSpanID := child.SpanContext().SpanID()
	grandchildSpanID := grandchild.SpanContext().SpanID()
	sharedTraceID := parent.SpanContext().TraceID()

	for _, s := range stubs {
		switch s.Name {
		case "span-parent":
			sc := s.SpanContext
			parentStub = &sc
			if s.Parent.SpanID().IsValid() {
				t.Errorf("parent should not have valid parent span ID, got %s", s.Parent.SpanID())
			}
		case "span-child":
			sc := s.SpanContext
			childStub = &sc
			if s.Parent.SpanID() != parentSpanID {
				t.Errorf("child's ParentSpanID %s != parent's SpanID %s", s.Parent.SpanID(), parentSpanID)
			}
		case "span-grandchild":
			sc := s.SpanContext
			grandchildStub = &sc
			if s.Parent.SpanID() != childSpanID {
				t.Errorf("grandchild's ParentSpanID %s != child's SpanID %s", s.Parent.SpanID(), childSpanID)
			}
		}
	}

	if parentStub == nil || childStub == nil || grandchildStub == nil {
		t.Fatal("could not find all 3 spans in exporter")
	}

	if childStub.TraceID() != sharedTraceID || grandchildStub.TraceID() != sharedTraceID {
		t.Errorf("trace ID divergence: parent=%s, child=%s, grandchild=%s",
			sharedTraceID, childStub.TraceID(), grandchildStub.TraceID())
	}
	_ = grandchildSpanID
}

func TestTracer_InMemoryExporter_Reset(t *testing.T) {
	t.Parallel()

	tracer, exporter := telemetry.NewTestTracer()

	for i := 0; i < 5; i++ {
		_, span := tracer.StartSpan(context.Background(), "batch-span")
		span.End()
	}

	if len(exporter.GetSpans()) != 5 {
		t.Fatalf("expected 5 spans, got %d", len(exporter.GetSpans()))
	}

	exporter.Reset()

	if len(exporter.GetSpans()) != 0 {
		t.Fatalf("expected 0 spans after Reset(), got %d", len(exporter.GetSpans()))
	}

	_, span := tracer.StartSpan(context.Background(), "post-reset-span")
	span.End()

	if len(exporter.GetSpans()) != 1 {
		t.Fatalf("expected 1 span after reset and new span, got %d", len(exporter.GetSpans()))
	}
}

func TestNoopTracerProvider(t *testing.T) {
	t.Parallel()

	noop := telemetry.NewNoopTracerProvider()
	ctx, span := noop.StartSpan(context.Background(), "noop-span")
	if span == nil {
		t.Fatal("expected non-nil span from no-op provider")
	}
	span.End()

	req := httptest.NewRequest(http.MethodGet, "/test", nil)
	noop.InjectHTTPHeaders(ctx, req)
	extractedCtx := noop.ExtractHTTPHeaders(context.Background(), req)
	if extractedCtx == nil {
		t.Fatal("expected non-nil context from ExtractHTTPHeaders")
	}
	if err := noop.(telemetry.ShutdownableTracerProvider).Shutdown(context.Background()); err != nil {
		t.Errorf("expected clean shutdown from no-op provider, got %v", err)
	}
}

func TestTraceIDAndSpanIDFromContext(t *testing.T) {
	t.Parallel()

	// Empty context should return empty strings
	if tid := telemetry.TraceIDFromContext(context.Background()); tid != "" {
		t.Errorf("expected empty TraceID for empty context, got %q", tid)
	}
	if sid := telemetry.SpanIDFromContext(context.Background()); sid != "" {
		t.Errorf("expected empty SpanID for empty context, got %q", sid)
	}

	// Active span context
	tracer, _ := telemetry.NewTestTracer()
	ctx, span := tracer.StartSpan(context.Background(), "test-span")
	defer span.End()

	tid := telemetry.TraceIDFromContext(ctx)
	sid := telemetry.SpanIDFromContext(ctx)

	if tid != span.SpanContext().TraceID().String() {
		t.Errorf("expected TraceID %s, got %s", span.SpanContext().TraceID().String(), tid)
	}
	if sid != span.SpanContext().SpanID().String() {
		t.Errorf("expected SpanID %s, got %s", span.SpanContext().SpanID().String(), sid)
	}
}
