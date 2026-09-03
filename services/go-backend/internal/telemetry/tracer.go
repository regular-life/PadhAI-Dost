package telemetry

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/stdout/stdouttrace"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/sdk/trace/tracetest"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

// TracerProvider defines the contract for OpenTelemetry distributed tracing operations.
type TracerProvider interface {
	// StartSpan creates and starts a new trace span as a child of the span in ctx.
	StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span)

	// InjectHTTPHeaders injects the W3C traceparent (and tracestate/baggage) into outgoing HTTP request headers.
	InjectHTTPHeaders(ctx context.Context, req *http.Request)

	// ExtractHTTPHeaders extracts W3C traceparent headers from incoming HTTP requests into a context.
	ExtractHTTPHeaders(ctx context.Context, req *http.Request) context.Context
}

// ShutdownableTracerProvider extends TracerProvider with lifecycle termination.
type ShutdownableTracerProvider interface {
	TracerProvider
	Shutdown(ctx context.Context) error
}

// ExporterType defines the span export destination.
type ExporterType string

const (
	ExporterMemory ExporterType = "memory"
	ExporterStdout ExporterType = "stdout"
	ExporterNoop   ExporterType = "noop"
)

// Config holds configuration parameters for the OpenTelemetry Tracer.
type Config struct {
	ServiceName    string
	ServiceVersion string
	Environment    string
	ExporterType   ExporterType
	SampleRatio    float64
	StdoutWriter   io.Writer
}

// DefaultConfig returns production-safe default configurations.
func DefaultConfig() Config {
	return Config{
		ServiceName:    "councilai-go-backend",
		ServiceVersion: "2.0.0",
		Environment:    "development",
		ExporterType:   ExporterNoop,
		SampleRatio:    1.0,
		StdoutWriter:   os.Stdout,
	}
}

// Tracer implements TracerProvider and ShutdownableTracerProvider.
type Tracer struct {
	provider   *sdktrace.TracerProvider
	tracer     trace.Tracer
	propagator propagation.TextMapPropagator
	inMemory   *tracetest.InMemoryExporter
}

// NewTracerProvider initializes an OpenTelemetry TracerProvider according to Config.
func NewTracerProvider(cfg Config) (*Tracer, error) {
	if cfg.ServiceName == "" {
		cfg.ServiceName = "councilai-go-backend"
	}
	if cfg.ServiceVersion == "" {
		cfg.ServiceVersion = "2.0.0"
	}
	if cfg.SampleRatio <= 0 {
		cfg.SampleRatio = 1.0
	}

	// 1. Configure standard W3C TraceContext + Baggage composite propagator
	prop := propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	)
	otel.SetTextMapPropagator(prop)

	// 2. Build OpenTelemetry Resource with semantic conventions
	res, err := resource.New(
		context.Background(),
		resource.WithAttributes(
			semconv.ServiceNameKey.String(cfg.ServiceName),
			semconv.ServiceVersionKey.String(cfg.ServiceVersion),
			semconv.DeploymentEnvironmentKey.String(cfg.Environment),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("failed to create otel resource: %w", err)
	}

	// 3. Configure Sampler
	var sampler sdktrace.Sampler
	if cfg.SampleRatio >= 1.0 {
		sampler = sdktrace.AlwaysSample()
	} else {
		sampler = sdktrace.TraceIDRatioBased(cfg.SampleRatio)
	}

	tpOpts := []sdktrace.TracerProviderOption{
		sdktrace.WithResource(res),
		sdktrace.WithSampler(sampler),
	}

	var inMemoryExp *tracetest.InMemoryExporter

	// 4. Configure Exporter
	switch cfg.ExporterType {
	case ExporterMemory:
		inMemoryExp = tracetest.NewInMemoryExporter()
		// CRITICAL: Synchronous span export for deterministic, sub-second test execution
		tpOpts = append(tpOpts, sdktrace.WithSyncer(inMemoryExp))

	case ExporterStdout:
		w := cfg.StdoutWriter
		if w == nil {
			w = os.Stdout
		}
		exp, err := stdouttrace.New(stdouttrace.WithWriter(w))
		if err != nil {
			return nil, fmt.Errorf("failed to create stdout trace exporter: %w", err)
		}
		tpOpts = append(tpOpts, sdktrace.WithBatcher(exp))

	case ExporterNoop:
		// No span processor attached; tracer generates spans without export overhead
	default:
		// Default to no-op
	}

	tp := sdktrace.NewTracerProvider(tpOpts...)
	otel.SetTracerProvider(tp)

	namedTracer := tp.Tracer(
		cfg.ServiceName,
		trace.WithInstrumentationVersion(cfg.ServiceVersion),
	)

	return &Tracer{
		provider:   tp,
		tracer:     namedTracer,
		propagator: prop,
		inMemory:   inMemoryExp,
	}, nil
}

// NewInMemoryTracerProvider initializes a Tracer backed by tracetest.NewInMemoryExporter.
// Guaranteed to export synchronously for immediate assertion in unit/integration tests.
func NewInMemoryTracerProvider() (*Tracer, *tracetest.InMemoryExporter, error) {
	cfg := Config{
		ServiceName:  "test-tracer",
		ExporterType: ExporterMemory,
		SampleRatio:  1.0,
	}
	t, err := NewTracerProvider(cfg)
	if err != nil {
		return nil, nil, err
	}
	return t, t.inMemory, nil
}

// NewTestTracer creates an in-memory Tracer for test environments, panicking on failure.
func NewTestTracer() (*Tracer, *tracetest.InMemoryExporter) {
	t, exp, err := NewInMemoryTracerProvider()
	if err != nil {
		panic(fmt.Sprintf("failed to create test tracer: %v", err))
	}
	return t, exp
}

// StartSpan creates and starts a new trace span with given options.
func (t *Tracer) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	if ctx == nil {
		ctx = context.Background()
	}
	if t == nil || t.tracer == nil {
		return ctx, trace.SpanFromContext(ctx)
	}
	return t.tracer.Start(ctx, name, opts...)
}

// InjectHTTPHeaders serializes trace context into outgoing HTTP request headers via W3C traceparent.
func (t *Tracer) InjectHTTPHeaders(ctx context.Context, req *http.Request) {
	if req == nil {
		return
	}
	if req.Header == nil {
		req.Header = make(http.Header)
	}
	if ctx == nil {
		ctx = req.Context()
		if ctx == nil {
			ctx = context.Background()
		}
	}
	prop := t.propagator
	if prop == nil {
		prop = otel.GetTextMapPropagator()
	}
	prop.Inject(ctx, propagation.HeaderCarrier(req.Header))
}

// ExtractHTTPHeaders extracts W3C traceparent headers from incoming HTTP requests into a context.
func (t *Tracer) ExtractHTTPHeaders(ctx context.Context, req *http.Request) context.Context {
	if req == nil {
		if ctx == nil {
			return context.Background()
		}
		return ctx
	}
	if ctx == nil {
		ctx = req.Context()
		if ctx == nil {
			ctx = context.Background()
		}
	}
	prop := t.propagator
	if prop == nil {
		prop = otel.GetTextMapPropagator()
	}
	return prop.Extract(ctx, propagation.HeaderCarrier(req.Header))
}

// Shutdown flushes all queued spans and stops the TracerProvider.
func (t *Tracer) Shutdown(ctx context.Context) error {
	if t == nil || t.provider == nil {
		return nil
	}
	return t.provider.Shutdown(ctx)
}

// InMemoryExporter returns the underlying in-memory exporter if initialized in memory mode.
func (t *Tracer) InMemoryExporter() *tracetest.InMemoryExporter {
	if t == nil {
		return nil
	}
	return t.inMemory
}

// GetSpans returns all recorded span stubs if configured with an in-memory exporter.
func (t *Tracer) GetSpans() []tracetest.SpanStub {
	if t == nil || t.inMemory == nil {
		return nil
	}
	return t.inMemory.GetSpans()
}

// Reset clears recorded span stubs in the in-memory exporter.
func (t *Tracer) Reset() {
	if t != nil && t.inMemory != nil {
		t.inMemory.Reset()
	}
}

// TraceIDFromContext extracts the trace ID string from the active span in ctx.
func TraceIDFromContext(ctx context.Context) string {
	span := trace.SpanFromContext(ctx)
	if !span.SpanContext().IsValid() {
		return ""
	}
	return span.SpanContext().TraceID().String()
}

// SpanIDFromContext extracts the span ID string from the active span in ctx.
func SpanIDFromContext(ctx context.Context) string {
	span := trace.SpanFromContext(ctx)
	if !span.SpanContext().IsValid() {
		return ""
	}
	return span.SpanContext().SpanID().String()
}

// RecordError marks the span status as Error and records the error event.
func RecordError(span trace.Span, err error) {
	if span == nil || err == nil {
		return
	}
	span.RecordError(err)
	span.SetStatus(codes.Error, err.Error())
}

// NoopTracerProvider provides a zero-allocation TracerProvider implementation for disabled tracing.
type NoopTracerProvider struct{}

// NewNoopTracerProvider returns a no-op implementation of TracerProvider.
func NewNoopTracerProvider() TracerProvider {
	return &NoopTracerProvider{}
}

func (n *NoopTracerProvider) StartSpan(ctx context.Context, name string, opts ...trace.SpanStartOption) (context.Context, trace.Span) {
	if ctx == nil {
		ctx = context.Background()
	}
	return ctx, trace.SpanFromContext(ctx)
}

func (n *NoopTracerProvider) InjectHTTPHeaders(ctx context.Context, req *http.Request) {}

func (n *NoopTracerProvider) ExtractHTTPHeaders(ctx context.Context, req *http.Request) context.Context {
	if ctx == nil {
		if req != nil {
			return req.Context()
		}
		return context.Background()
	}
	return ctx
}

func (n *NoopTracerProvider) Shutdown(ctx context.Context) error {
	return nil
}
