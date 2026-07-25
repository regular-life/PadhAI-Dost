package handlers

import (
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/agent"
	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/memory"
	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
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

// HandleQuery processes a user's question, classifying the intent, extracting context, and deliberating.
func (h *Handlers) HandleQuery(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	var req QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}
	if req.Question == "" {
		jsonError(w, "question is required", http.StatusBadRequest)
		return
	}
	if req.TopK <= 0 {
		req.TopK = 5
	}

	hasDocument := req.DocID != ""
	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte(req.Question)))[:16]

	// L1 Semantic Cache lookup.
	var vector []float32
	if hasDocument {
		var err error
		vector, err = h.getEmbedding(r.Context(), req.Question)
		if err == nil && len(vector) == 384 {
			var semCachedResponse QueryResponse
			if found, _ := h.SemanticCache.Get(r.Context(), req.DocID, vector, h.SemanticCacheThreshold, &semCachedResponse); found {
				semCachedResponse.CacheHit = true
				semCachedResponse.Latency = time.Since(start).String()
				h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "semantic_cache_hit")
				metrics.CacheHits.WithLabelValues("hit", "l1").Inc()
				jsonResponse(w, semCachedResponse)
				return
			}
			metrics.CacheHits.WithLabelValues("miss", "l1").Inc()
		} else if err != nil {
			log.Printf("[Query] Failed to get embedding for semantic cache: %v", err)
		}
	}

	// L2 Redis Cache lookup.
	cacheKey := fmt.Sprintf("cache:%s:%s", req.DocID, req.Question)
	var cachedResponse QueryResponse
	if found, err := h.Cache.Get(r.Context(), cacheKey, &cachedResponse); err == nil && found {
		cachedResponse.CacheHit = true
		cachedResponse.Latency = time.Since(start).String()
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "redis_cache_hit")
		jsonResponse(w, cachedResponse)
		return
	}

	// Fetch conversation turns.
	var history []council.ConversationTurn
	if req.SessionID != "" && h.Memory != nil {
		turns, err := h.Memory.GetHistory(r.Context(), userID, req.SessionID, 5)
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

	// Choose deliberation strategy.
	var docSummary string
	if hasDocument {
		if found, _ := h.Cache.Get(r.Context(), "doc_summary:"+req.DocID, &docSummary); !found {
			log.Printf("[Query] Document summary not found in cache for %s", req.DocID)
		}
	}

	plan, err := h.Router.Plan(r.Context(), req.Question, docSummary)
	if err != nil {
		log.Printf("[Query] Router failed: %v, defaulting to council", err)
		plan = &agent.QueryPlan{Strategy: "council", NeedsDoc: hasDocument}
	}

	// Fetch document context or fall back to web search if no document is provided.
	var chunks []string
	if plan.NeedsDoc && hasDocument {
		chunks, err = h.retrieveChunks(r, req)
		if err != nil {
			log.Printf("[Query] Retrieval failed: %v", err)
			log.Printf("[Query] Falling back to general knowledge mode")
			chunks = nil
		}
	} else if !hasDocument {
		log.Printf("[Query] General query (no document), executing Web Search fallback...")
		chunks, err = h.retrieveWebSearch(r.Context(), req.Question)
		if err != nil {
			log.Printf("[Query] Web search fallback failed: %v", err)
		}
	}

	// Deliberate.
	councilStart := time.Now()
	var result *council.CouncilResult

	switch plan.Strategy {
	case "direct":
		result, err = h.Council.QueryDirect(r.Context(), req.Question, chunks, history)
	case "council", "council_deep":
		result, err = h.Council.Query(r.Context(), req.Question, chunks, "", false, plan.Strategy, history)
	default:
		result, err = h.Council.Query(r.Context(), req.Question, chunks, "", false, "council", history)
	}

	if err != nil {
		log.Printf("[Query] Council failed: %v", err)
		jsonError(w, "LLM council failed", http.StatusInternalServerError)
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
		return
	}
	metrics.CouncilResponseTime.Observe(time.Since(councilStart).Seconds())

	response := QueryResponse{
		Answer:       result.FinalAnswer,
		Confidence:   result.Confidence,
		Source:       result.Source,
		Strategy:     result.Strategy,
		Reasoning:    result.Reasoning,
		PeerReviewed: len(result.PeerReviews) > 0,
		Reflection:   result.Reflection,
		Candidates:   result.CandidateAnswers,
		Latency:      time.Since(start).String(),
		CacheHit:     false,
	}

	// Cache result.
	if err := h.Cache.Set(r.Context(), cacheKey, response); err != nil {
		log.Printf("[Query] Cache set failed: %v", err)
	}
	if hasDocument && len(vector) == 384 {
		if err := h.SemanticCache.Put(r.Context(), req.DocID, vector, response); err != nil {
			log.Printf("[Query] Semantic cache put failed: %v", err)
		}
	}

	// Save turn history.
	if req.SessionID != "" && h.Memory != nil {
		h.Memory.Append(r.Context(), userID, req.SessionID, memory.Turn{
			Role:    "user",
			Content: req.Question,
		})
		h.Memory.Append(r.Context(), userID, req.SessionID, memory.Turn{
			Role:    "assistant",
			Content: result.FinalAnswer,
		})
	}

	h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	jsonResponse(w, response)
}
