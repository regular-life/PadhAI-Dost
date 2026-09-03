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
func (h *Handlers) HandleQuery(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	// 0. Extract incoming W3C traceparent or initialize root HTTP span
	ctx := r.Context()
	if h.Tracer != nil {
		ctx = h.Tracer.ExtractHTTPHeaders(ctx, r)
	}

	var rootSpan trace.Span
	if h.Tracer != nil {
		ctx, rootSpan = h.Tracer.StartSpan(ctx, "HTTP POST /api/v1/query",
			trace.WithSpanKind(trace.SpanKindServer),
			trace.WithAttributes(
				attribute.String("http.method", r.Method),
				attribute.String("http.route", "/api/v1/query"),
			),
		)
		defer rootSpan.End()
	}

	// Propagate trace IDs in response headers
	if sc := trace.SpanFromContext(ctx).SpanContext(); sc.IsValid() {
		w.Header().Set("X-Trace-ID", sc.TraceID().String())
		w.Header().Set("traceparent", fmt.Sprintf("00-%s-%s-%s", sc.TraceID().String(), sc.SpanID().String(), sc.TraceFlags().String()))
	}
	r = r.WithContext(ctx)

	// 1. Request Validation (returns HTTP 400 JSON before any stream headers)
	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		if rootSpan != nil {
			rootSpan.RecordError(err)
			rootSpan.SetStatus(codes.Error, "invalid request body")
			rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusBadRequest))
		}
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Question == "" {
		if rootSpan != nil {
			rootSpan.SetStatus(codes.Error, "question is required")
			rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusBadRequest))
		}
		jsonError(w, "question is required", http.StatusBadRequest)
		return
	}
	if len(req.Question) > 10000 {
		if rootSpan != nil {
			rootSpan.SetStatus(codes.Error, "question exceeds maximum allowed length of 10000 characters")
			rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusBadRequest))
		}
		jsonError(w, "question exceeds maximum allowed length of 10000 characters", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}

	hasDocument := req.DocID != ""
	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte(req.Question)))[:16]

	// 2. Content Negotiation
	sseMode := isSSE(r)
	if rootSpan != nil {
		rootSpan.SetAttributes(
			attribute.String("query.doc_id", req.DocID),
			attribute.String("query.user_id", userID),
			attribute.Int("query.top_k", req.TopK),
			attribute.Bool("query.sse", sseMode),
			attribute.Bool("query.sse_mode", sseMode),
			attribute.Bool("query.has_document", hasDocument),
		)
	}

	var flusher http.Flusher
	if sseMode {
		var ok bool
		flusher, ok = w.(http.Flusher)
		if !ok {
			if rootSpan != nil {
				rootSpan.SetStatus(codes.Error, "streaming unsupported")
				rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusInternalServerError))
			}
			jsonError(w, "streaming unsupported", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/event-stream; charset=utf-8")
		w.Header().Set("Cache-Control", "no-cache")
		w.Header().Set("Connection", "keep-alive")
		w.Header().Set("X-Accel-Buffering", "no")
		w.WriteHeader(http.StatusOK)
		flusher.Flush()
	}

	// 3. L1 Exact Match Cache lookup (1ms Redis GET, 0ms embedding overhead).
	cacheKey := fmt.Sprintf("cache:%s:%s", req.DocID, req.Question)
	var cachedResponse QueryResponse
	if h.Cache != nil {
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
		} else if found {
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
				return
			}
			jsonResponse(w, cachedResponse)
			return
		}
	}

	// 4. L2 Semantic Vector Cache lookup (RediSearch VSS).
	var vector []float32
	if hasDocument && h.SemanticCache != nil {
		var err error
		vector, err = h.getEmbedding(ctx, req.Question)
		if err != nil {
			log.Printf("[Query] Failed to get embedding for semantic cache: %v, proceeding to deliberation", err)
		} else if len(vector) == 384 {
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
			} else if found {
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
					return
				}
				jsonResponse(w, semCachedResponse)
				return
			}
		} else {
			log.Printf("[Query] Embedding vector dimension mismatch (got %d, expected 384), proceeding to deliberation", len(vector))
		}
	}

	// 5. Fetch conversation turns.
	var history []council.ConversationTurn
	if req.SessionID != "" && h.Memory != nil {
		turns, err := h.Memory.GetHistory(ctx, userID, req.SessionID, 5)
		if err != nil {
			log.Printf("[Query] Failed to get conversation history: %v", err)
		} else {
			for _, t := range turns {
				history = append(history, council.ConversationTurn{
					Role:    t.Role,
					Content: t.Content,
				})
			}
		}
	}

	// 6. Deliberation Strategy Planning
	var docSummary string
	if hasDocument && h.Cache != nil {
		if found, err := h.Cache.Get(ctx, "doc_summary:"+req.DocID, &docSummary); err != nil {
			if !errors.Is(err, cache.ErrCircuitOpen) {
				log.Printf("[Query] Failed to get doc_summary from cache for %s: %v", req.DocID, err)
			}
		} else if !found {
			log.Printf("[Query] Document summary not found in cache for %s", req.DocID)
		}
	}

	var plan *agent.QueryPlan
	if h.Router != nil {
		var err error
		plan, err = h.Router.Plan(ctx, req.Question, docSummary)
		if err != nil {
			log.Printf("[Query] Router failed: %v, defaulting to council", err)
			plan = &agent.QueryPlan{Strategy: "council", NeedsDoc: hasDocument}
		}
	} else {
		plan = &agent.QueryPlan{Strategy: "council", NeedsDoc: hasDocument}
	}

	// 7. Context Retrieval
	var chunks []string
	if plan.NeedsDoc && hasDocument {
		var err error
		chunks, err = h.retrieveChunks(r.WithContext(ctx), req)
		if err != nil {
			log.Printf("[Query] Retrieval failed: %v, falling back to general knowledge", err)
			chunks = nil
		}
	} else if !hasDocument && h.HTTPClient != nil {
		log.Printf("[Query] General query (no document), executing Web Search fallback...")
		var err error
		chunks, err = h.retrieveWebSearch(ctx, req.Question)
		if err != nil {
			log.Printf("[Query] Web search fallback failed: %v", err)
		}
	}

	// 8. Deliberation Execution
	councilStart := time.Now()
	var (
		result     *council.CouncilResult
		councilErr error
	)

	if sseMode {
		eventChan := make(chan council.StreamEvent, 16)
		doneChan := make(chan struct{})

		go func(execCtx context.Context) {
			defer close(doneChan)
			if h.Council != nil {
				switch plan.Strategy {
				case "direct":
					result, councilErr = h.Council.QueryDirectStream(execCtx, req.Question, chunks, history, eventChan)
				case "council", "council_deep":
					result, councilErr = h.Council.QueryStream(execCtx, req.Question, chunks, "", false, plan.Strategy, history, eventChan)
				default:
					result, councilErr = h.Council.QueryStream(execCtx, req.Question, chunks, "", false, "council", history, eventChan)
				}
			} else {
				close(eventChan)
				councilErr = fmt.Errorf("council orchestrator not initialized")
			}
		}(ctx)

		// Event streaming loop with client disconnect handling
	streamLoop:
		for {
			select {
			case <-ctx.Done():
				log.Printf("[Query] Client disconnected mid-stream: %v", ctx.Err())
				return
			case event, ok := <-eventChan:
				if !ok {
					break streamLoop
				}
				if writeErr := writeSSE(w, flusher, string(event.Type), event.Data); writeErr != nil {
					log.Printf("[Query] Failed to write SSE frame: %v", writeErr)
					return
				}
			}
		}

		<-doneChan

		if councilErr != nil {
			if rootSpan != nil {
				rootSpan.RecordError(councilErr)
				rootSpan.SetStatus(codes.Error, councilErr.Error())
				rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusInternalServerError))
			}
			log.Printf("[Query] Council failed: %v", councilErr)
			_ = writeSSE(w, flusher, "error", map[string]interface{}{
				"code":    http.StatusInternalServerError,
				"message": "LLM council failed",
				"error":   councilErr.Error(),
			})
			if h.Audit != nil {
				h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
			}
			return
		}
	} else {
		if h.Council != nil {
			switch plan.Strategy {
			case "direct":
				result, councilErr = h.Council.QueryDirect(ctx, req.Question, chunks, history)
			case "council", "council_deep":
				result, councilErr = h.Council.Query(ctx, req.Question, chunks, "", false, plan.Strategy, history)
			default:
				result, councilErr = h.Council.Query(ctx, req.Question, chunks, "", false, "council", history)
			}
		} else {
			councilErr = fmt.Errorf("council orchestrator not initialized")
		}

		if councilErr != nil {
			if rootSpan != nil {
				rootSpan.RecordError(councilErr)
				rootSpan.SetStatus(codes.Error, councilErr.Error())
				rootSpan.SetAttributes(attribute.Int("http.status_code", http.StatusInternalServerError))
			}
			log.Printf("[Query] Council failed: %v", councilErr)
			jsonError(w, "LLM council failed", http.StatusInternalServerError)
			if h.Audit != nil {
				h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
			}
			return
		}
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

	candidateList := result.CandidateAnswers
	if len(candidateList) == 0 && len(result.Candidates) > 0 {
		candidateList = result.Candidates
	}

	response := QueryResponse{
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

	// 9. Side Effects (L1 Exact Cache, L2 Semantic Cache, Memory, Audit)
	// Degradation guarantee: Any cache or memory write error must not fail the request.
	if h.Cache != nil {
		if err := h.Cache.Set(ctx, cacheKey, response); err != nil {
			if !errors.Is(err, cache.ErrCircuitOpen) {
				log.Printf("[Query] L1 exact cache set failed: %v", err)
			}
		}
	}

	if hasDocument && h.SemanticCache != nil {
		if len(vector) != 384 {
			var err error
			vector, err = h.getEmbedding(ctx, req.Question)
			if err != nil {
				log.Printf("[Query] Failed to get embedding for semantic cache put: %v", err)
			}
		}
		if len(vector) == 384 {
			if err := h.SemanticCache.Put(ctx, req.DocID, vector, response); err != nil {
				if !errors.Is(err, cache.ErrCircuitOpen) {
					log.Printf("[Query] L2 semantic cache put failed: %v", err)
				}
			}
		}
	}

	if req.SessionID != "" && h.Memory != nil {
		if err := h.Memory.Append(ctx, userID, req.SessionID, memory.Turn{
			Role:    "user",
			Content: req.Question,
		}); err != nil {
			log.Printf("[Query] Failed to append user turn to memory: %v", err)
		}
		if err := h.Memory.Append(ctx, userID, req.SessionID, memory.Turn{
			Role:    "assistant",
			Content: result.FinalAnswer,
		}); err != nil {
			log.Printf("[Query] Failed to append assistant turn to memory: %v", err)
		}
	}

	if h.Audit != nil {
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	}

	// 10. Non-streaming JSON mode final output
	if !sseMode {
		jsonResponse(w, response)
	}
}
