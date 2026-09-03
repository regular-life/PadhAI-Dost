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

// GenerateQuestionsRequest defines parameters to generate practice questions.
type GenerateQuestionsRequest struct {
	DocID        string `json:"doc_id"`
	NumQuestions int    `json:"num_questions"`
	Count        int    `json:"count,omitempty"`
	Difficulty   int    `json:"difficulty"`
	QuestionType string `json:"question_type"`
	BloomLevel   string `json:"bloom_level,omitempty"`
}

// UnmarshalJSON custom unmarshals GenerateQuestionsRequest to support both "num_questions" and "count".
func (g *GenerateQuestionsRequest) UnmarshalJSON(data []byte) error {
	type Alias GenerateQuestionsRequest
	aux := &struct {
		*Alias
	}{
		Alias: (*Alias)(g),
	}
	if err := json.Unmarshal(data, aux); err != nil {
		return err
	}
	if g.NumQuestions <= 0 && g.Count > 0 {
		g.NumQuestions = g.Count
	}
	if g.Count <= 0 && g.NumQuestions > 0 {
		g.Count = g.NumQuestions
	}
	return nil
}

// GeneratedQuestion represents a single structured question output.
type GeneratedQuestion struct {
	Question    string   `json:"question"`
	Answer      string   `json:"answer"`
	Explanation string   `json:"explanation"`
	SourceChunk string   `json:"source_chunk,omitempty"`
	Options     []string `json:"options,omitempty"`
}

// GenerateQuestionsResponse holds the generated mock assessment.
type GenerateQuestionsResponse struct {
	Questions  []GeneratedQuestion `json:"questions"`
	RawOutput  string              `json:"raw_output,omitempty"`
	Confidence float64             `json:"confidence"`
	Source     string              `json:"source"`
	Latency    string              `json:"latency"`
	CacheHit   bool                `json:"cache_hit"`
}

// HandleGenerateQuestions generates practice questions from document text.
func (h *Handlers) validateGenerateQuestionsRequest(w http.ResponseWriter, r *http.Request) (*GenerateQuestionsRequest, bool) {
	var req GenerateQuestionsRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return nil, false
	}
	if req.NumQuestions <= 0 && req.Count > 0 {
		req.NumQuestions = req.Count
	}
	if req.DocID == "" {
		jsonError(w, "doc_id is required", http.StatusBadRequest)
		return nil, false
	}
	if req.NumQuestions <= 0 {
		req.NumQuestions = 5
	} else if req.NumQuestions > 20 {
		req.NumQuestions = 20
	}
	if req.Difficulty <= 0 || req.Difficulty > 10 {
		req.Difficulty = 5
	}
	if req.QuestionType == "" {
		req.QuestionType = "subjective"
	}
	return &req, true
}

func (h *Handlers) checkQuestionsCache(
	ctx context.Context,
	w http.ResponseWriter,
	cacheKey, queryHash, userID, docID string,
	start time.Time,
) bool {
	if h.Cache == nil {
		return false
	}
	var cachedResponse GenerateQuestionsResponse
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

func buildQuestionsPrompt(contextText string, req *GenerateQuestionsRequest) string {
	bloomClause := ""
	if req.BloomLevel != "" {
		bloomClause = fmt.Sprintf("\n- Target Bloom's taxonomy level: %s", req.BloomLevel)
	}

	questionTypeInstruction := "open-ended subjective questions requiring evidence-based reasoning"
	optionsSchema := ""
	if req.QuestionType == "mcq" {
		questionTypeInstruction = "multiple-choice questions (MCQ) with exactly 4 options each (A, B, C, D) and indicate the correct answer"
		optionsSchema = ",\n  \"options\": [\"A) ...\", \"B) ...\", \"C) ...\", \"D) ...\"]"
	}

	return fmt.Sprintf(`You are an expert assessment designer. Based ONLY on the following document excerpts, generate practice questions.

Document Excerpts:
%s

Generate exactly %d %s.

Requirements:
- Difficulty level: %d/10 (1-3: recall, 4-6: analysis/application, 7-10: synthesis/evaluation)%s
- Ground every question in the provided document content
- Each question must be answerable from the excerpts
- Provide an answer and brief explanation for each question
- Do not repeat similar questions

Respond as a JSON array where each element has:
{
  "question": "The question text",
  "answer": "The correct answer",
  "explanation": "Why this is the answer, citing the source material"%s
}

Respond ONLY with the JSON array.`, contextText, req.NumQuestions, questionTypeInstruction, req.Difficulty, bloomClause, optionsSchema)
}

func parseQuestionsResponse(rawAnswer string, limit int) []GeneratedQuestion {
	var questions []GeneratedQuestion
	jsonStr := extractQuestionsJSON(rawAnswer)
	if err := json.Unmarshal([]byte(jsonStr), &questions); err != nil {
		log.Printf("[GenerateQuestions] JSON parse failed, returning raw output: %v", err)
	}
	if len(questions) > limit {
		questions = questions[:limit]
	}
	return questions
}

func (h *Handlers) HandleGenerateQuestions(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	req, ok := h.validateGenerateQuestionsRequest(w, r)
	if !ok {
		return
	}

	cacheKey := fmt.Sprintf("questions:%s:%d:%d:%s", req.DocID, req.NumQuestions, req.Difficulty, req.QuestionType)
	queryHash := fmt.Sprintf("%x", sha256.Sum256([]byte(cacheKey)))[:16]

	if h.checkQuestionsCache(r.Context(), w, cacheKey, queryHash, userID, req.DocID, start) {
		return
	}

	chunks, err := h.retrieveAllChunks(req.DocID)
	if err != nil {
		log.Printf("[GenerateQuestions] Retrieval failed: %v", err)
		jsonError(w, "failed to retrieve document", http.StatusInternalServerError)
		return
	}
	if len(chunks) == 0 {
		jsonError(w, "document not found or empty", http.StatusNotFound)
		return
	}

	prompt := buildQuestionsPrompt(strings.Join(chunks, "\n\n---\n\n"), req)
	result, err := h.Council.Query(r.Context(), "questions:"+req.DocID, chunks, prompt, true, "council", nil)
	if err != nil {
		log.Printf("[GenerateQuestions] Council failed: %v", err)
		jsonError(w, "question generation failed", http.StatusInternalServerError)
		if h.Audit != nil {
			h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "council_error")
		}
		return
	}

	questions := parseQuestionsResponse(result.FinalAnswer, req.NumQuestions)
	response := GenerateQuestionsResponse{
		Questions:  questions,
		RawOutput:  result.FinalAnswer,
		Confidence: result.Confidence,
		Source:     result.Source,
		Latency:    time.Since(start).String(),
		CacheHit:   false,
	}

	if h.Cache != nil {
		if err := h.Cache.Set(r.Context(), cacheKey, response); err != nil {
			log.Printf("[GenerateQuestions] Cache set failed: %v", err)
		}
	}

	if h.Audit != nil {
		h.Audit.LogQuery(userID, req.DocID, queryHash, time.Since(start), "success")
	}
	jsonResponse(w, response)
}
