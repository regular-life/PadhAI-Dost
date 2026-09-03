package handlers

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
)

// ExplainRequest defines the payload for document explanation options.
type ExplainRequest struct {
	DocID          string   `json:"doc_id"`
	KnowledgeLevel string   `json:"knowledge_level"`
	Level          string   `json:"level,omitempty"`
	Depth          string   `json:"depth"`
	FocusTopics    []string `json:"focus_topics,omitempty"`
}

// UnmarshalJSON custom unmarshals ExplainRequest to support both "knowledge_level" and "level".
func (e *ExplainRequest) UnmarshalJSON(data []byte) error {
	type Alias ExplainRequest
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(e),
	}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if e.KnowledgeLevel == "" && e.Level != "" {
		e.KnowledgeLevel = e.Level
	}
	if e.Level == "" && e.KnowledgeLevel != "" {
		e.Level = e.KnowledgeLevel
	}
	return nil
}

// ExplainSection holds a structured subsection of a document explanation.
type ExplainSection struct {
	Heading  string `json:"heading"`
	Content  string `json:"content"`
	PageRefs []int  `json:"page_refs,omitempty"`
}

// ExplainResponse holds the generated structured document explanation.
type ExplainResponse struct {
	Explanation string           `json:"explanation"`
	Sections    []ExplainSection `json:"sections,omitempty"`
	Confidence  float64          `json:"confidence"`
	Source      string           `json:"source"`
	Latency     string           `json:"latency"`
	CacheHit    bool             `json:"cache_hit"`
}

// HandleExplain generates customized summaries and explanations of documents.
func (h *Handlers) validateExplainRequest(w http.ResponseWriter, r *http.Request) (*ExplainRequest, bool) {
	var req ExplainRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return nil, false
	}
	if req.KnowledgeLevel == "" && req.Level != "" {
		req.KnowledgeLevel = req.Level
	}
	if req.DocID == "" {
		jsonError(w, "doc_id is required", http.StatusBadRequest)
		return nil, false
	}
	if req.KnowledgeLevel == "" {
		req.KnowledgeLevel = "beginner"
	}
	if req.Depth == "" {
		req.Depth = "section-wise"
	}
	return &req, true
}

func (h *Handlers) checkExplainCache(
	ctx context.Context,
	w http.ResponseWriter,
	cacheKey, queryHash, userID, docID string,
	start time.Time,
) bool {
	if h.Cache == nil {
		return false
	}
	var cachedResponse ExplainResponse
	if found, err := h.Cache.Get(ctx, cacheKey, &cachedResponse); err == nil && found {
		cachedResponse.CacheHit = true
		cachedResponse.Latency = time.Since(start).String()
		if h.Audit != nil {
			h.Audit.LogQuery(userID, docID, queryHash, time.Since(start), "cache_hit")
		}
		jsonResponse(w, cachedResponse)
		return true
	}
	return false
}

func buildExplainPrompt(contextText string, req *ExplainRequest) string {
	focusClause := ""
	if len(req.FocusTopics) > 0 {
		focusClause = fmt.Sprintf("\n\nFocus especially on these topics: %s", strings.Join(req.FocusTopics, ", "))
	}

	depthInstruction := map[string]string{
		"brief":        "Provide a concise summary (2-3 paragraphs max).",
		"section-wise": "Organize the explanation into clear sections with headings.",
		"detailed":     "Provide a thorough, detailed explanation covering all major points.",
	}[req.Depth]
	if depthInstruction == "" {
		depthInstruction = "Organize the explanation into clear sections with headings."
	}

	levelInstruction := map[string]string{
		"beginner":     "Explain as if to someone completely new to the subject. Use simple language, analogies, and define all technical terms.",
		"intermediate": "Explain at an intermediate level. Use domain terminology but provide context. Include analysis and comparisons.",
		"advanced":     "Explain at an expert level. Use precise technical language. Include critical analysis, research implications, and connections to related concepts.",
	}[req.KnowledgeLevel]
	if levelInstruction == "" {
		levelInstruction = "Explain as if to someone completely new to the subject."
	}

	return fmt.Sprintf(`You are an expert educator. Based ONLY on the following document excerpts, generate an explanation.

Document Excerpts:
%s

Instructions:
- %s
- %s
- Ground every claim in the provided excerpts — do not invent information
- Use clear formatting with headings and structure%s

Respond with a well-structured explanation.`, contextText, levelInstruction, depthInstruction, focusClause)
}

func (h *Handlers) HandleExplain(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	req, ok := h.validateExplainRequest(w, r)
	if !ok {
		return
	}

	cacheKey := fmt.Sprintf("explain:%s:%s:%s", req.DocID, req.KnowledgeLevel, req.Depth)
	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte(cacheKey)))[:16]

	if h.checkExplainCache(r.Context(), w, cacheKey, queryHash, userID, req.DocID, start) {
		return
	}

	chunks, err := h.retrieveAllChunks(req.DocID)
	if err != nil {
		log.Printf("[Explain] Retrieval failed: %v", err)
		jsonError(w, "failed to retrieve document", http.StatusInternalServerError)
		return
	}
	if len(chunks) == 0 {
		jsonError(w, "document not found or empty", http.StatusNotFound)
		return
	}

	prompt := buildExplainPrompt(strings.Join(chunks, "\n\n---\n\n"), req)
	result, err := h.Council.Query(r.Context(), "explain:"+req.DocID, chunks, prompt, false, "council", nil)
	if err != nil {
		log.Printf("[Explain] Council failed: %v", err)
		jsonError(w, "explanation generation failed", http.StatusInternalServerError)
		if h.Audit != nil {
			h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
		}
		return
	}

	response := ExplainResponse{
		Explanation: result.FinalAnswer,
		Confidence:  result.Confidence,
		Source:      result.Source,
		Latency:     time.Since(start).String(),
		CacheHit:    false,
	}

	if h.Cache != nil {
		if err := h.Cache.Set(r.Context(), cacheKey, response); err != nil {
			log.Printf("[Explain] Cache set failed: %v", err)
		}
	}

	if h.Audit != nil {
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	}
	jsonResponse(w, response)
}
