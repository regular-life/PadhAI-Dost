package handlers

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/agent"
	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/memory"
	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// QueryRequest defines the POST /api/v1/query payload.
type QueryRequest struct {
	Question  string `json:"question"`
	DocID     string `json:"doc_id,omitempty"`
	TopK      int    `json:"top_k,omitempty"`
	SessionID string `json:"session_id,omitempty"`
}

// QueryResponse defines the POST /api/v1/query response structure.
type QueryResponse struct {
	Answer       string                    `json:"answer"`
	Confidence   float64                   `json:"confidence"`
	Source       string                    `json:"source"`
	Strategy     string                    `json:"strategy,omitempty"`
	Reasoning    string                    `json:"reasoning,omitempty"`
	PeerReviewed bool                      `json:"peer_reviewed"`
	Reflection   *council.ReflectionResult `json:"reflection,omitempty"`
	Candidates   []council.CandidateAnswer `json:"candidates,omitempty"`
	Latency      string                    `json:"latency"`
	CacheHit     bool                      `json:"cache_hit"`
}

// isSSE checks if the incoming request specifies text/event-stream in Accept headers.
func isSSE(r *http.Request) bool {
	for _, v := range r.Header.Values("Accept") {
		if strings.Contains(strings.ToLower(v), "text/event-stream") {
			return true
		}
	}
	return strings.Contains(strings.ToLower(r.Header.Get("Accept")), "text/event-stream")
}

// writeSSE serializes and writes a single SSE event frame, then flushes.
func writeSSE(w http.ResponseWriter, flusher http.Flusher, eventType string, data interface{}) error {
	payload, err := json.Marshal(data)
	if err != nil {
		return fmt.Errorf("failed to marshal SSE payload: %w", err)
	}
	if _, err := fmt.Fprintf(w, "event: %s\ndata: %s\n\n", eventType, string(payload)); err != nil {
		return err
	}
	if flusher != nil {
		flusher.Flush()
	}
	return nil
}

// HandleQuery processes a user's question, classifying the intent, extracting context, and deliberating.
// Gracefully degrades to direct LLM deliberation on any Redis cache errors, OOM, or circuit breaker trips.
func (h *Handlers) validateQueryRequest(w http.ResponseWriter, r *http.Request, rootSpan trace.Span) (*QueryRequest, bool) {
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if rootSpan != nil {
			rootSpan.RecordError(err)
			rootSpan.SetStatus(codes.Error, "invalid request body")
			rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusBadRequest))
		}
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return nil, false
	}
	if req.Question == "" {
		if rootSpan != nil {
			rootSpan.SetStatus(codes.Error, "question is required")
			rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusBadRequest))
		}
		jsonError(w, "question is required", http.StatusBadRequest)
		return nil, false
	}
	if len(req.Question) > 10000 {
		if rootSpan != nil {
			rootSpan.SetStatus(codes.Error, "question exceeds maximum allowed length of 10000 characters")
			rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusBadRequest))
		}
		jsonError(w, "question exceeds maximum allowed length of 10000 characters", http.StatusBadRequest)
		return nil, false
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}
	return &req, true
}

func (h *Handlers) tryL1CacheLookup(
	ctx context.Context,
	w http.ResponseWriter,
	rootSpan trace.Span,
	req *QueryRequest,
	cacheKey, queryHash, userID string,
	start time.Time,
	sseMode bool,
	flusher http.Flusher,
) bool {
	if h.Cache == nil {
		return false
	}
	var l1Span trace.Span
	l1Ctx := ctx
	if h.Tracer != nil {
		l1Ctx, l1Span = h.Tracer.StartSpan(ctx, "cache.l1_lookup",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("cache.tier", "l1_exact"),
				attribute.String("cache.key", cacheKey),
				attribute.String("cache.doc_id", req.DocID),
			),
		)
	}
	var cachedResponse QueryResponse
	found, err := h.Cache.Get(l1Ctx, cacheKey, &cachedResponse)
	if l1Span != nil {
		if err != nil {
			l1Span.RecordError(err)
			l1Span.SetStatus(codes.Error, err.Error())
		} else {
			l1Span.SetAttributes(attribute.Bool("cache.hit", found))
			l1Span.SetStatus(codes.Ok, "")
		}
		l1Span.End()
	}
	if err != nil {
		if !errors.Is(err, cache.ErrCircuitOpen) {
			log.Printf("[Query] L1 exact cache get warning: %v, proceeding to L2/deliberation", err)
		}
		return false
	}
	if !found {
		return false
	}

	if rootSpan != nil {
		rootSpan.SetAttributes(
			attribute.Bool("query.cache_hit", true),
			attribute.String("query.cache_tier", "l1_exact"),
			attribute.Int("http.status_code", http.StatusOK),
		)
		rootSpan.SetStatus(codes.Ok, "")
	}
	cachedResponse.CacheHit = true
	cachedResponse.Latency = time.Since(start).String()
	if h.Audit != nil {
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "redis_exact_cache_hit")
	}

	if sseMode {
		_ = writeSSE(w, flusher, "final_answer", cachedResponse)
	} else {
		jsonResponse(w, cachedResponse)
	}
	return true
}

func (h *Handlers) tryL2CacheLookup(
	ctx context.Context,
	w http.ResponseWriter,
	rootSpan trace.Span,
	req *QueryRequest,
	queryHash, userID string,
	start time.Time,
	sseMode bool,
	flusher http.Flusher,
) (bool, []float32) {
	if req.DocID == "" || h.SemanticCache == nil {
		return false, nil
	}
	vector, err := h.getEmbedding(ctx, req.Question)
	if err != nil {
		log.Printf("[Query] Failed to get embedding for semantic cache: %v, proceeding to deliberation", err)
		return false, nil
	}
	if len(vector) != 384 {
		log.Printf("[Query] Embedding vector dimension mismatch (got %d, expected 384), proceeding to deliberation", len(vector))
		return false, vector
	}

	var l2Span trace.Span
	l2Ctx := ctx
	if h.Tracer != nil {
		l2Ctx, l2Span = h.Tracer.StartSpan(ctx, "cache.l2_lookup",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("cache.tier", "l2_semantic"),
				attribute.String("cache.doc_id", req.DocID),
				attribute.Float64("cache.threshold", float64(h.SemanticCacheThreshold)),
			),
		)
	}
	var semCachedResponse QueryResponse
	found, semErr := h.SemanticCache.Get(l2Ctx, req.DocID, vector, h.SemanticCacheThreshold, &semCachedResponse)
	if l2Span != nil {
		if semErr != nil {
			l2Span.RecordError(semErr)
			l2Span.SetStatus(codes.Error, semErr.Error())
		} else {
			l2Span.SetAttributes(attribute.Bool("cache.hit", found))
			l2Span.SetStatus(codes.Ok, "")
		}
		l2Span.End()
	}
	if semErr != nil {
		if !errors.Is(semErr, cache.ErrCircuitOpen) {
			log.Printf("[Query] L2 semantic cache get warning: %v, proceeding to deliberation", semErr)
		}
		return false, vector
	}
	if !found {
		return false, vector
	}

	if rootSpan != nil {
		rootSpan.SetAttributes(
			attribute.Bool("query.cache_hit", true),
			attribute.String("query.cache_tier", "l2_semantic"),
			attribute.Int("http.status_code", http.StatusOK),
		)
		rootSpan.SetStatus(codes.Ok, "")
	}
	semCachedResponse.CacheHit = true
	semCachedResponse.Latency = time.Since(start).String()
	if h.Audit != nil {
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "semantic_cache_hit")
	}

	if sseMode {
		_ = writeSSE(w, flusher, "final_answer", semCachedResponse)
	} else {
		jsonResponse(w, semCachedResponse)
	}
	return true, vector
}

func (h *Handlers) getConversationHistory(ctx context.Context, sessionID, userID string) []council.ConversationTurn {
	if sessionID == "" || h.Memory == nil {
		return nil
	}
	turns, err := h.Memory.GetHistory(ctx, userID, sessionID, 5)
	if err != nil {
		log.Printf("[Query] Failed to get conversation history: %v", err)
		return nil
	}
	history := make([]council.ConversationTurn, 0, len(turns))
	for _, t := range turns {
		history = append(history, council.ConversationTurn{
			Role:    t.Role,
			Content: t.Content,
		})
	}
	return history
}

func (h *Handlers) resolveQueryPlan(ctx context.Context, question, docID string) *agent.QueryPlan {
	hasDocument := docID != ""
	var docSummary string
	if hasDocument && h.Cache != nil {
		if found, err := h.Cache.Get(ctx, "doc_summary:"+docID, &docSummary); err != nil {
			if !errors.Is(err, cache.ErrCircuitOpen) {
				log.Printf("[Query] Failed to get doc_summary from cache for %s: %v", docID, err)
			}
		} else if !found {
			log.Printf("[Query] Document summary not found in cache for %s", docID)
		}
	}

	if h.Router != nil {
		plan, err := h.Router.Plan(ctx, question, docSummary)
		if err == nil {
			return plan
		}
		log.Printf("[Query] Router failed: %v, defaulting to council", err)
	}
	return &agent.QueryPlan{Strategy: "council", NeedsDoc: hasDocument}
}

func (h *Handlers) fetchQueryContext(ctx context.Context, r *http.Request, plan *agent.QueryPlan, req *QueryRequest) []string {
	if plan.NeedsDoc && req.DocID != "" {
		chunks, err := h.retrieveChunks(r.WithContext(ctx), *req)
		if err != nil {
			log.Printf("[Query] Retrieval failed: %v, falling back to general knowledge", err)
			return nil
		}
		return chunks
	}
	if req.DocID == "" && h.HTTPClient != nil {
		log.Printf("[Query] General query (no document), executing Web Search fallback...")
		chunks, err := h.retrieveWebSearch(ctx, req.Question)
		if err != nil {
			log.Printf("[Query] Web search fallback failed: %v", err)
		}
		return chunks
	}
	return nil
}

func (h *Handlers) executeStreamingDeliberation(
	ctx context.Context,
	w http.ResponseWriter,
	flusher http.Flusher,
	plan *agent.QueryPlan,
	req *QueryRequest,
	chunks []string,
	history []council.ConversationTurn,
) (*council.CouncilResult, error) {
	eventChan := make(chan council.StreamEvent, 16)
	doneChan := make(chan struct{})

	var (
		result     *council.CouncilResult
		councilErr error
	)

	go func(execCtx context.Context) {
		defer close(doneChan)
		if h.Council == nil {
			close(eventChan)
			councilErr = fmt.Errorf("council orchestrator not initialized")
			return
		}
		switch plan.Strategy {
		case "direct":
			result, councilErr = h.Council.QueryDirectStream(execCtx, req.Question, chunks, history, eventChan)
		case "council", "council_deep":
			result, councilErr = h.Council.QueryStream(execCtx, req.Question, chunks, "", false, plan.Strategy, history, eventChan)
		default:
			result, councilErr = h.Council.QueryStream(execCtx, req.Question, chunks, "", false, "council", history, eventChan)
		}
	}(ctx)

	for {
		select {
		case <-ctx.Done():
			log.Printf("[Query] Client disconnected mid-stream: %v", ctx.Err())
			return nil, ctx.Err()
		case event, ok := <-eventChan:
			if !ok {
				<-doneChan
				return result, councilErr
			}
			if writeErr := writeSSE(w, flusher, string(event.Type), event.Data); writeErr != nil {
				log.Printf("[Query] Failed to write SSE frame: %v", writeErr)
				return nil, nil
			}
		}
	}
}

func (h *Handlers) executeDirectDeliberation(
	ctx context.Context,
	plan *agent.QueryPlan,
	req *QueryRequest,
	chunks []string,
	history []council.ConversationTurn,
) (*council.CouncilResult, error) {
	if h.Council == nil {
		return nil, fmt.Errorf("council orchestrator not initialized")
	}
	switch plan.Strategy {
	case "direct":
		return h.Council.QueryDirect(ctx, req.Question, chunks, history)
	case "council", "council_deep":
		return h.Council.Query(ctx, req.Question, chunks, "", false, plan.Strategy, history)
	default:
		return h.Council.Query(ctx, req.Question, chunks, "", false, "council", history)
	}
}

func (h *Handlers) saveToCaches(
	ctx context.Context,
	req *QueryRequest,
	cacheKey string,
	vector []float32,
	response QueryResponse,
) {
	if h.Cache != nil {
		if err := h.Cache.Set(ctx, cacheKey, response); err != nil && !errors.Is(err, cache.ErrCircuitOpen) {
			log.Printf("[Query] L1 exact cache set failed: %v", err)
		}
	}

	if req.DocID == "" || h.SemanticCache == nil {
		return
	}
	if len(vector) != 384 {
		var err error
		vector, err = h.getEmbedding(ctx, req.Question)
		if err != nil {
			log.Printf("[Query] Failed to get embedding for semantic cache put: %v", err)
			return
		}
	}
	if len(vector) == 384 {
		if err := h.SemanticCache.Put(ctx, req.DocID, vector, response); err != nil && !errors.Is(err, cache.ErrCircuitOpen) {
			log.Printf("[Query] L2 semantic cache put failed: %v", err)
		}
	}
}

func (h *Handlers) saveToMemory(ctx context.Context, sessionID, userID, question, answer string) {
	if sessionID == "" || h.Memory == nil {
		return
	}
	if err := h.Memory.Append(ctx, userID, sessionID, memory.Turn{Role: "user", Content: question}); err != nil {
		log.Printf("[Query] Failed to append user turn to memory: %v", err)
	}
	if err := h.Memory.Append(ctx, userID, sessionID, memory.Turn{Role: "assistant", Content: answer}); err != nil {
		log.Printf("[Query] Failed to append assistant turn to memory: %v", err)
	}
}

func (h *Handlers) initQueryTracing(w http.ResponseWriter, r *http.Request) (context.Context, trace.Span, func()) {
	ctx := r.Context()
	if h.Tracer != nil {
		ctx = h.Tracer.ExtractHTTPHeaders(ctx, r)
	}

	var rootSpan trace.Span
	cleanup := func() {}
	if h.Tracer != nil {
		ctx, rootSpan = h.Tracer.StartSpan(ctx, "HTTP POST /api/v1/query",
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				attribute.String("http.method", r.Method),
				attribute.String("http.route", "/api/v1/query"),
			),
		)
		cleanup = func() { rootSpan.End() }
	}

	if sc := trace.SpanFromContext(ctx).SpanContext(); sc.IsValid() {
		w.Header().Set("X-Trace-ID", sc.TraceID().String())
		w.Header().Set("traceparent", fmt.Sprintf("00-%s-%s-%s", sc.TraceID().String(), sc.SpanID().String(), sc.TraceFlags().String()))
	}
	return ctx, rootSpan, cleanup
}

func (h *Handlers) setupSSEStream(w http.ResponseWriter, rootSpan trace.Span) (http.Flusher, bool) {
	flusher, isFlusher := w.(http.Flusher)
	if !isFlusher {
		if rootSpan != nil {
			rootSpan.SetStatus(codes.Error, "streaming unsupported")
			rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusInternalServerError))
		}
		jsonError(w, "streaming unsupported", http.StatusInternalServerError)
		return nil, false
	}
	w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	return flusher, true
}

func (h *Handlers) handleCouncilFailure(
	w http.ResponseWriter,
	flusher http.Flusher,
	rootSpan trace.Span,
	councilErr error,
	sseMode bool,
	userID, docID, queryHash string,
	start time.Time,
) {
	if rootSpan != nil {
		rootSpan.RecordError(councilErr)
		rootSpan.SetStatus(codes.Error, councilErr.Error())
		rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusInternalServerError))
	}
	log.Printf("[Query] Council failed: %v", councilErr)
	if sseMode {
		_ = writeSSE(w, flusher, "error", map[string]interface{}{
			"code":    http.StatusInternalServerError,
			"message": "LLM council failed",
			"error":   councilErr.Error(),
		})
	} else {
		jsonError(w, "LLM council failed", http.StatusInternalServerError)
	}
	if h.Audit != nil {
		h.Audit.LogQuery(userID, docID, queryHash, time.Since(start), "council_error")
	}
}

func (h *Handlers) buildQueryResponse(result *council.CouncilResult, start time.Time) QueryResponse {
	candidateList := result.CandidateAnswers
	if len(candidateList) == 0 && len(result.Candidates) > 0 {
		candidateList = result.Candidates
	}

	return QueryResponse{
		Answer:       result.FinalAnswer,
		Confidence:   result.Confidence,
		Source:       result.Source,
		Strategy:     result.Strategy,
		Reasoning:    result.Reasoning,
		PeerReviewed: len(result.PeerReviews) > 0 || result.PeerReviewed,
		Reflection:   result.Reflection,
		Candidates:   candidateList,
		Latency:      time.Since(start).String(),
		CacheHit:     false,
	}
}

func (h *Handlers) HandleQuery(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	// 0. Extract incoming W3C traceparent or initialize root HTTP span
	ctx, rootSpan, endSpan := h.initQueryTracing(w, r)
	defer endSpan()
	r = r.WithContext(ctx)

	// 1. Request Validation
	req, ok := h.validateQueryRequest(w, r, rootSpan)
	if !ok {
		return
	}

	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte(req.Question)))[:16]
	sseMode := isSSE(r)
	if rootSpan != nil {
		rootSpan.SetAttributes(
			attribute.String("query.doc_id", req.DocID),
			attribute.String("query.user_id", userID),
			attribute.Int("query.top_k", req.TopK),
			attribute.Bool("query.sse", sseMode),
			attribute.Bool("query.sse_mode", sseMode),
			attribute.Bool("query.has_document", req.DocID != ""),
		)
	}

	var flusher http.Flusher
	if sseMode {
		var sseOk bool
		flusher, sseOk = h.setupSSEStream(w, rootSpan)
		if !sseOk {
			return
		}
	}

	// 2. L1 Exact Match Cache & L2 Semantic Cache
	cacheKey := fmt.Sprintf("cache:%s:%s", req.DocID, req.Question)
	if h.tryL1CacheLookup(ctx, w, rootSpan, req, cacheKey, queryHash, userID, start, sseMode, flusher) {
		return
	}
	hitL2, vector := h.tryL2CacheLookup(ctx, w, rootSpan, req, queryHash, userID, start, sseMode, flusher)
	if hitL2 {
		return
	}

	// 3. Conversation History & Plan
	history := h.getConversationHistory(ctx, req.SessionID, userID)
	plan := h.resolveQueryPlan(ctx, req.Question, req.DocID)
	chunks := h.fetchQueryContext(ctx, r, plan, req)

	// 4. Deliberation Execution
	councilStart := time.Now()
	var (
		result     *council.CouncilResult
		councilErr error
	)

	if sseMode {
		result, councilErr = h.executeStreamingDeliberation(ctx, w, flusher, plan, req, chunks, history)
		if ctx.Err() != nil || (result == nil && councilErr == nil) {
			return
		}
	} else {
		result, councilErr = h.executeDirectDeliberation(ctx, plan, req, chunks, history)
	}

	if councilErr != nil {
		h.handleCouncilFailure(w, flusher, rootSpan, councilErr, sseMode, userID, req.DocID, queryHash, start)
		return
	}

	metrics.CouncilResponseTime.Observe(time.Since(councilStart).Seconds())
	if rootSpan != nil {
		rootSpan.SetAttributes(
			attribute.String("query.strategy", result.Strategy),
			attribute.Float64("query.confidence", result.Confidence),
			attribute.Int("http.status_code", http.StatusOK),
		)
		rootSpan.SetStatus(codes.Ok, "")
	}

	response := h.buildQueryResponse(result, start)

	// 5. Side Effects (Caches, Memory, Audit)
	h.saveToCaches(ctx, req, cacheKey, vector, response)
	h.saveToMemory(ctx, req.SessionID, userID, req.Question, result.FinalAnswer)
	if h.Audit != nil {
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	}

	if !sseMode {
		jsonResponse(w, response)
	}
}
