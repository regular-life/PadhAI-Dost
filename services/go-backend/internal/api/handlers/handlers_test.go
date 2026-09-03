package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/agent"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

func TestJSONResponse(t *testing.T) {
	w := httptest.NewRecorder()
	payload := map[string]string{"status": "ok", "message": "hello"}

	jsonResponse(w, payload)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", res.StatusCode)
	}

	contentType := res.Header.Get("Content-Type")
	if contentType != "application/json" {
		t.Errorf("expected Content-Type application/json, got %s", contentType)
	}

	var data map[string]string
	if err := json.NewDecoder(res.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode JSON response: %v", err)
	}

	if data["status"] != "ok" || data["message"] != "hello" {
		t.Errorf("unexpected body payload: %v", data)
	}
}

func TestJSONError(t *testing.T) {
	w := httptest.NewRecorder()

	jsonError(w, "invalid request parameter", http.StatusBadRequest)

	res := w.Result()
	if res.StatusCode != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", res.StatusCode)
	}

	var data map[string]string
	if err := json.NewDecoder(res.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode JSON error: %v", err)
	}

	if data["error"] != "invalid request parameter" {
		t.Errorf("expected error message 'invalid request parameter', got %q", data["error"])
	}
}

func TestExplainRequestParameterAliases(t *testing.T) {
	// Task Test Case 1: {"doc_id": "doc1", "level": "advanced"}
	json1 := `{"doc_id": "doc1", "level": "advanced"}`
	var req1 ExplainRequest
	if err := json.Unmarshal([]byte(json1), &req1); err != nil {
		t.Fatalf("failed to unmarshal json1: %v", err)
	}
	if req1.KnowledgeLevel != "advanced" || req1.Level != "advanced" {
		t.Errorf("Test 1 failed: expected KnowledgeLevel and Level to be 'advanced', got KnowledgeLevel=%q, Level=%q", req1.KnowledgeLevel, req1.Level)
	}

	// Task Test Case 2: {"doc_id": "doc1", "knowledge_level": "intermediate"}
	json2 := `{"doc_id": "doc1", "knowledge_level": "intermediate"}`
	var req2 ExplainRequest
	if err := json.Unmarshal([]byte(json2), &req2); err != nil {
		t.Fatalf("failed to unmarshal json2: %v", err)
	}
	if req2.KnowledgeLevel != "intermediate" || req2.Level != "intermediate" {
		t.Errorf("Test 2 failed: expected KnowledgeLevel and Level to be 'intermediate', got KnowledgeLevel=%q, Level=%q", req2.KnowledgeLevel, req2.Level)
	}

	// Task Test Case 3 (Precedence check): {"doc_id": "doc1", "level": "expert", "knowledge_level": "basic"}
	json3 := `{"doc_id": "doc1", "level": "expert", "knowledge_level": "basic"}`
	var req3 ExplainRequest
	if err := json.Unmarshal([]byte(json3), &req3); err != nil {
		t.Fatalf("failed to unmarshal json3: %v", err)
	}
	if req3.KnowledgeLevel != "basic" {
		t.Errorf("Test 3 precedence failed: expected KnowledgeLevel='basic' (primary field), got KnowledgeLevel=%q", req3.KnowledgeLevel)
	}
	if req3.Level != "expert" {
		t.Errorf("Test 3 precedence failed: expected raw Level='expert', got Level=%q", req3.Level)
	}
}

func TestGenerateQuestionsRequestParameterAliases(t *testing.T) {
	// Task Test Case 4: {"doc_id": "doc1", "count": 10}
	json4 := `{"doc_id": "doc1", "count": 10}`
	var req4 GenerateQuestionsRequest
	if err := json.Unmarshal([]byte(json4), &req4); err != nil {
		t.Fatalf("failed to unmarshal json4: %v", err)
	}
	if req4.NumQuestions != 10 || req4.Count != 10 {
		t.Errorf("Test 4 failed: expected NumQuestions and Count to be 10, got NumQuestions=%d, Count=%d", req4.NumQuestions, req4.Count)
	}

	// Task Test Case 5: {"doc_id": "doc1", "num_questions": 7}
	json5 := `{"doc_id": "doc1", "num_questions": 7}`
	var req5 GenerateQuestionsRequest
	if err := json.Unmarshal([]byte(json5), &req5); err != nil {
		t.Fatalf("failed to unmarshal json5: %v", err)
	}
	if req5.NumQuestions != 7 || req5.Count != 7 {
		t.Errorf("Test 5 failed: expected NumQuestions and Count to be 7, got NumQuestions=%d, Count=%d", req5.NumQuestions, req5.Count)
	}

	// Task Test Case 6 (Precedence check): {"doc_id": "doc1", "count": 15, "num_questions": 3}
	json6 := `{"doc_id": "doc1", "count": 15, "num_questions": 3}`
	var req6 GenerateQuestionsRequest
	if err := json.Unmarshal([]byte(json6), &req6); err != nil {
		t.Fatalf("failed to unmarshal json6: %v", err)
	}
	if req6.NumQuestions != 3 {
		t.Errorf("Test 6 precedence failed: expected NumQuestions=3 (primary field), got NumQuestions=%d", req6.NumQuestions)
	}
	if req6.Count != 15 {
		t.Errorf("Test 6 precedence failed: expected raw Count=15, got Count=%d", req6.Count)
	}
}

func TestHandlersPanicSafetyAndValidation(t *testing.T) {
	h := &Handlers{}

	explainCases := []struct {
		name         string
		body         string
		expectedCode int
		expectedErr  string
	}{
		{
			name:         "Invalid JSON body",
			body:         `{invalid_json}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "invalid request body",
		},
		{
			name:         "Missing doc_id with level alias",
			body:         `{"level": "advanced"}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "doc_id is required",
		},
		{
			name:         "Missing doc_id with knowledge_level",
			body:         `{"knowledge_level": "intermediate"}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "doc_id is required",
		},
		{
			name:         "Missing doc_id with precedence conflict",
			body:         `{"level": "expert", "knowledge_level": "basic"}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "doc_id is required",
		},
	}

	for _, tc := range explainCases {
		t.Run("HandleExplain/"+tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/explain", bytes.NewBufferString(tc.body))
			w := httptest.NewRecorder()

			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("HandleExplain panicked on input %s: %v", tc.body, r)
				}
			}()

			h.HandleExplain(w, req)
			res := w.Result()
			if res.StatusCode != tc.expectedCode {
				t.Errorf("expected status %d, got %d", tc.expectedCode, res.StatusCode)
			}
			var resp map[string]string
			if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
				t.Fatalf("failed to decode response: %v", err)
			}
			if resp["error"] != tc.expectedErr {
				t.Errorf("expected error %q, got %q", tc.expectedErr, resp["error"])
			}
		})
	}

	questionCases := []struct {
		name         string
		body         string
		expectedCode int
		expectedErr  string
	}{
		{
			name:         "Invalid JSON body",
			body:         `{invalid_json}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "invalid request body",
		},
		{
			name:         "Missing doc_id with count alias",
			body:         `{"count": 10}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "doc_id is required",
		},
		{
			name:         "Missing doc_id with num_questions",
			body:         `{"num_questions": 7}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "doc_id is required",
		},
		{
			name:         "Missing doc_id with precedence conflict",
			body:         `{"count": 15, "num_questions": 3}`,
			expectedCode: http.StatusBadRequest,
			expectedErr:  "doc_id is required",
		},
	}

	for _, tc := range questionCases {
		t.Run("HandleGenerateQuestions/"+tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/questions", bytes.NewBufferString(tc.body))
			w := httptest.NewRecorder()

			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("HandleGenerateQuestions panicked on input %s: %v", tc.body, r)
				}
			}()

			h.HandleGenerateQuestions(w, req)
			res := w.Result()
			if res.StatusCode != tc.expectedCode {
				t.Errorf("expected status %d, got %d", tc.expectedCode, res.StatusCode)
			}
			var resp map[string]string
			if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
				t.Fatalf("failed to decode response: %v", err)
			}
			if resp["error"] != tc.expectedErr {
				t.Errorf("expected error %q, got %q", tc.expectedErr, resp["error"])
			}
		})
	}
}

func TestExplainRequestFallbackAndDefaultsTable(t *testing.T) {
	tests := []struct {
		name                 string
		payload              string
		expectedLevel        string
		expectedKnowledge    string
		expectedDefaultLevel string
		expectedDefaultDepth string
	}{
		{
			name:                 "Only level set",
			payload:              `{"doc_id": "doc1", "level": "advanced"}`,
			expectedLevel:        "advanced",
			expectedKnowledge:    "advanced",
			expectedDefaultLevel: "advanced",
			expectedDefaultDepth: "section-wise",
		},
		{
			name:                 "Only knowledge_level set",
			payload:              `{"doc_id": "doc1", "knowledge_level": "intermediate"}`,
			expectedLevel:        "intermediate",
			expectedKnowledge:    "intermediate",
			expectedDefaultLevel: "intermediate",
			expectedDefaultDepth: "section-wise",
		},
		{
			name:                 "Both level and knowledge_level set (knowledge_level primary)",
			payload:              `{"doc_id": "doc1", "level": "expert", "knowledge_level": "basic"}`,
			expectedLevel:        "expert",
			expectedKnowledge:    "basic",
			expectedDefaultLevel: "basic",
			expectedDefaultDepth: "section-wise",
		},
		{
			name:                 "Both level and knowledge_level set (knowledge_level expert)",
			payload:              `{"doc_id": "doc1", "level": "basic", "knowledge_level": "expert"}`,
			expectedLevel:        "basic",
			expectedKnowledge:    "expert",
			expectedDefaultLevel: "expert",
			expectedDefaultDepth: "section-wise",
		},
		{
			name:                 "Missing level and knowledge_level (fallback to beginner)",
			payload:              `{"doc_id": "doc1"}`,
			expectedLevel:        "",
			expectedKnowledge:    "",
			expectedDefaultLevel: "beginner",
			expectedDefaultDepth: "section-wise",
		},
		{
			name:                 "Empty strings for level and knowledge_level",
			payload:              `{"doc_id": "doc1", "level": "", "knowledge_level": ""}`,
			expectedLevel:        "",
			expectedKnowledge:    "",
			expectedDefaultLevel: "beginner",
			expectedDefaultDepth: "section-wise",
		},
		{
			name:                 "Custom depth provided",
			payload:              `{"doc_id": "doc1", "depth": "brief"}`,
			expectedLevel:        "",
			expectedKnowledge:    "",
			expectedDefaultLevel: "beginner",
			expectedDefaultDepth: "brief",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req ExplainRequest
			if err := json.Unmarshal([]byte(tt.payload), &req); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}
			if req.Level != tt.expectedLevel {
				t.Errorf("UnmarshalJSON Level = %q; want %q", req.Level, tt.expectedLevel)
			}
			if req.KnowledgeLevel != tt.expectedKnowledge {
				t.Errorf("UnmarshalJSON KnowledgeLevel = %q; want %q", req.KnowledgeLevel, tt.expectedKnowledge)
			}

			// Simulate handler default fallbacks
			if req.KnowledgeLevel == "" && req.Level != "" {
				req.KnowledgeLevel = req.Level
			}
			if req.KnowledgeLevel == "" {
				req.KnowledgeLevel = "beginner"
			}
			if req.Depth == "" {
				req.Depth = "section-wise"
			}

			if req.KnowledgeLevel != tt.expectedDefaultLevel {
				t.Errorf("Handler default KnowledgeLevel = %q; want %q", req.KnowledgeLevel, tt.expectedDefaultLevel)
			}
			if req.Depth != tt.expectedDefaultDepth {
				t.Errorf("Handler default Depth = %q; want %q", req.Depth, tt.expectedDefaultDepth)
			}
		})
	}
}

func TestGenerateQuestionsRequestFallbackAndDefaultsTable(t *testing.T) {
	tests := []struct {
		name                   string
		payload                string
		expectedNumQuestions   int
		expectedCount          int
		expectedFinalQuestions int
		expectedDifficulty     int
		expectedQuestionType   string
	}{
		{
			name:                   "Only count set",
			payload:                `{"doc_id": "doc1", "count": 10}`,
			expectedNumQuestions:   10,
			expectedCount:          10,
			expectedFinalQuestions: 10,
			expectedDifficulty:     5,
			expectedQuestionType:   "subjective",
		},
		{
			name:                   "Only num_questions set",
			payload:                `{"doc_id": "doc1", "num_questions": 7}`,
			expectedNumQuestions:   7,
			expectedCount:          7,
			expectedFinalQuestions: 7,
			expectedDifficulty:     5,
			expectedQuestionType:   "subjective",
		},
		{
			name:                   "Both set (num_questions primary)",
			payload:                `{"doc_id": "doc1", "count": 15, "num_questions": 3}`,
			expectedNumQuestions:   3,
			expectedCount:          15,
			expectedFinalQuestions: 3,
			expectedDifficulty:     5,
			expectedQuestionType:   "subjective",
		},
		{
			name:                   "Both set (num_questions 12, count 4)",
			payload:                `{"doc_id": "doc1", "count": 4, "num_questions": 12}`,
			expectedNumQuestions:   12,
			expectedCount:          4,
			expectedFinalQuestions: 12,
			expectedDifficulty:     5,
			expectedQuestionType:   "subjective",
		},
		{
			name:                   "Missing count and num_questions (defaults to 5)",
			payload:                `{"doc_id": "doc1"}`,
			expectedNumQuestions:   0,
			expectedCount:          0,
			expectedFinalQuestions: 5,
			expectedDifficulty:     5,
			expectedQuestionType:   "subjective",
		},
		{
			name:                   "Zero or negative num_questions and count",
			payload:                `{"doc_id": "doc1", "count": 0, "num_questions": -2}`,
			expectedNumQuestions:   -2,
			expectedCount:          0,
			expectedFinalQuestions: 5,
			expectedDifficulty:     5,
			expectedQuestionType:   "subjective",
		},
		{
			name:                   "Over maximum cap (>20)",
			payload:                `{"doc_id": "doc1", "count": 25}`,
			expectedNumQuestions:   25,
			expectedCount:          25,
			expectedFinalQuestions: 20,
			expectedDifficulty:     5,
			expectedQuestionType:   "subjective",
		},
		{
			name:                   "Custom difficulty and question_type",
			payload:                `{"doc_id": "doc1", "difficulty": 8, "question_type": "mcq"}`,
			expectedNumQuestions:   0,
			expectedCount:          0,
			expectedFinalQuestions: 5,
			expectedDifficulty:     8,
			expectedQuestionType:   "mcq",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var req GenerateQuestionsRequest
			if err := json.Unmarshal([]byte(tt.payload), &req); err != nil {
				t.Fatalf("json.Unmarshal failed: %v", err)
			}
			if req.NumQuestions != tt.expectedNumQuestions {
				t.Errorf("UnmarshalJSON NumQuestions = %d; want %d", req.NumQuestions, tt.expectedNumQuestions)
			}
			if req.Count != tt.expectedCount {
				t.Errorf("UnmarshalJSON Count = %d; want %d", req.Count, tt.expectedCount)
			}

			// Simulate handler default fallbacks
			if req.NumQuestions <= 0 && req.Count > 0 {
				req.NumQuestions = req.Count
			}
			if req.NumQuestions <= 0 {
				req.NumQuestions = 5
			}
			if req.NumQuestions > 20 {
				req.NumQuestions = 20
			}
			if req.Difficulty <= 0 || req.Difficulty > 10 {
				req.Difficulty = 5
			}
			if req.QuestionType == "" {
				req.QuestionType = "subjective"
			}

			if req.NumQuestions != tt.expectedFinalQuestions {
				t.Errorf("Handler final NumQuestions = %d; want %d", req.NumQuestions, tt.expectedFinalQuestions)
			}
			if req.Difficulty != tt.expectedDifficulty {
				t.Errorf("Handler final Difficulty = %d; want %d", req.Difficulty, tt.expectedDifficulty)
			}
			if req.QuestionType != tt.expectedQuestionType {
				t.Errorf("Handler final QuestionType = %q; want %q", req.QuestionType, tt.expectedQuestionType)
			}
		})
	}
}

func TestExhaustiveHTTPMatrix(t *testing.T) {
	h := &Handlers{}

	t.Run("ExplainRequest HTTP Matrix Validation", func(t *testing.T) {
		tests := []struct {
			name         string
			body         string
			expectedCode int
			expectedErr  string
		}{
			{
				name:         "level only without doc_id",
				body:         `{"level": "advanced"}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "knowledge_level only without doc_id",
				body:         `{"knowledge_level": "intermediate"}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "both level and knowledge_level without doc_id",
				body:         `{"level": "expert", "knowledge_level": "basic"}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "neither level nor knowledge_level without doc_id",
				body:         `{"depth": "brief"}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "malformed JSON",
				body:         `{"doc_id": "doc1", "level":`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "invalid request body",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				req := httptest.NewRequest("POST", "/api/explain", bytes.NewBufferString(tt.body))
				w := httptest.NewRecorder()

				h.HandleExplain(w, req)
				res := w.Result()

				if res.StatusCode != tt.expectedCode {
					t.Errorf("expected status %d, got %d", tt.expectedCode, res.StatusCode)
				}

				if ct := res.Header.Get("Content-Type"); ct != "application/json" {
					t.Errorf("expected Content-Type application/json, got %s", ct)
				}
				if nosniff := res.Header.Get("X-Content-Type-Options"); nosniff != "nosniff" {
					t.Errorf("expected X-Content-Type-Options nosniff, got %s", nosniff)
				}

				var resp map[string]string
				if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
					t.Fatalf("failed to decode error body: %v", err)
				}
				if resp["error"] != tt.expectedErr {
					t.Errorf("expected error %q, got %q", tt.expectedErr, resp["error"])
				}
			})
		}
	})

	t.Run("GenerateQuestionsRequest HTTP Matrix Validation", func(t *testing.T) {
		tests := []struct {
			name         string
			body         string
			expectedCode int
			expectedErr  string
		}{
			{
				name:         "count only without doc_id",
				body:         `{"count": 10}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "num_questions only without doc_id",
				body:         `{"num_questions": 7}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "both count and num_questions without doc_id",
				body:         `{"count": 15, "num_questions": 3}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "neither count nor num_questions without doc_id",
				body:         `{"difficulty": 5}`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "doc_id is required",
			},
			{
				name:         "malformed JSON",
				body:         `{"doc_id": "doc1", "count":`,
				expectedCode: http.StatusBadRequest,
				expectedErr:  "invalid request body",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				req := httptest.NewRequest("POST", "/api/questions", bytes.NewBufferString(tt.body))
				w := httptest.NewRecorder()

				h.HandleGenerateQuestions(w, req)
				res := w.Result()

				if res.StatusCode != tt.expectedCode {
					t.Errorf("expected status %d, got %d", tt.expectedCode, res.StatusCode)
				}

				if ct := res.Header.Get("Content-Type"); ct != "application/json" {
					t.Errorf("expected Content-Type application/json, got %s", ct)
				}
				if nosniff := res.Header.Get("X-Content-Type-Options"); nosniff != "nosniff" {
					t.Errorf("expected X-Content-Type-Options nosniff, got %s", nosniff)
				}

				var resp map[string]string
				if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
					t.Fatalf("failed to decode error body: %v", err)
				}
				if resp["error"] != tt.expectedErr {
					t.Errorf("expected error %q, got %q", tt.expectedErr, resp["error"])
				}
			})
		}
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// Milestone 1 SSE Streaming & Content Negotiation Test Doubles & Suites
// ─────────────────────────────────────────────────────────────────────────────

type handlerMockLLMClient struct {
	Name             string
	Delay            time.Duration
	Fail             bool
	GenerateFunc     func(ctx context.Context, prompt string) (*llm.Response, error)
	GenerateChatFunc func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error)
}

func (m *handlerMockLLMClient) ModelName() string {
	if m.Name != "" {
		return m.Name
	}
	return "mock:test-model"
}

func (m *handlerMockLLMClient) Generate(ctx context.Context, prompt string) (*llm.Response, error) {
	if m.Delay > 0 {
		select {
		case <-time.After(m.Delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if m.Fail {
		return nil, context.DeadlineExceeded
	}
	if m.GenerateFunc != nil {
		return m.GenerateFunc(ctx, prompt)
	}
	return &llm.Response{
		Answer:     "RANKING: A, B\nREASONING: Model A is concise and accurate",
		Model:      m.ModelName(),
		Confidence: 0.9,
	}, nil
}

func (m *handlerMockLLMClient) GenerateChat(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
	if m.Delay > 0 {
		select {
		case <-time.After(m.Delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if m.Fail {
		return nil, context.DeadlineExceeded
	}
	if m.GenerateChatFunc != nil {
		return m.GenerateChatFunc(ctx, opts)
	}
	return &llm.Response{
		Answer:     "Candidate draft from " + m.ModelName(),
		Model:      m.ModelName(),
		Confidence: 0.9,
	}, nil
}

type ParsedSSEEvent struct {
	Event string
	Data  string
}

func parseSSEEvents(raw string) []ParsedSSEEvent {
	var events []ParsedSSEEvent
	blocks := strings.Split(raw, "\n\n")
	for _, block := range blocks {
		block = strings.TrimSpace(block)
		if block == "" {
			continue
		}
		var evt ParsedSSEEvent
		lines := strings.Split(block, "\n")
		for _, line := range lines {
			line = strings.TrimSpace(line)
			if strings.HasPrefix(line, "event:") {
				evt.Event = strings.TrimSpace(strings.TrimPrefix(line, "event:"))
			} else if strings.HasPrefix(line, "data:") {
				evt.Data = strings.TrimSpace(strings.TrimPrefix(line, "data:"))
			}
		}
		if evt.Event != "" || evt.Data != "" {
			events = append(events, evt)
		}
	}
	return events
}

type StreamingRecorder struct {
	*httptest.ResponseRecorder
	FlushChan chan struct{}
	mu        sync.RWMutex
}

func NewStreamingRecorder() *StreamingRecorder {
	return &StreamingRecorder{
		ResponseRecorder: httptest.NewRecorder(),
		FlushChan:        make(chan struct{}, 100),
	}
}

func (r *StreamingRecorder) Write(b []byte) (int, error) {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.ResponseRecorder.Write(b)
}

func (r *StreamingRecorder) BodyString() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.ResponseRecorder.Body == nil {
		return ""
	}
	return r.ResponseRecorder.Body.String()
}

func (r *StreamingRecorder) BodyBytes() []byte {
	r.mu.RLock()
	defer r.mu.RUnlock()
	if r.ResponseRecorder.Body == nil {
		return nil
	}
	cp := make([]byte, r.ResponseRecorder.Body.Len())
	copy(cp, r.ResponseRecorder.Body.Bytes())
	return cp
}

func (r *StreamingRecorder) Flush() {
	r.ResponseRecorder.Flush()
	select {
	case r.FlushChan <- struct{}{}:
	default:
	}
}

type nonFlusherResponseWriter struct {
	header     http.Header
	body       *bytes.Buffer
	statusCode int
}

func newNonFlusherResponseWriter() *nonFlusherResponseWriter {
	return &nonFlusherResponseWriter{
		header:     make(http.Header),
		body:       new(bytes.Buffer),
		statusCode: http.StatusOK,
	}
}

func (n *nonFlusherResponseWriter) Header() http.Header {
	return n.header
}

func (n *nonFlusherResponseWriter) Write(b []byte) (int, error) {
	return n.body.Write(b)
}

func (n *nonFlusherResponseWriter) WriteHeader(code int) {
	n.statusCode = code
}

func setupTestHandlersWithCouncil(t *testing.T, clients []llm.LLMClient, chairman llm.LLMClient) *Handlers {
	t.Helper()
	orch := council.NewOrchestrator(clients, chairman, 5*time.Second)
	return &Handlers{
		Council: orch,
	}
}

func TestHandleQuery_SSE_Headers(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	h := setupTestHandlersWithCouncil(t, nil, nil)

	cases := []struct {
		name         string
		acceptHeader string
	}{
		{"Standard SSE Accept", "text/event-stream"},
		{"Case Insensitive Accept", "TEXT/EVENT-STREAM"},
		{"Accept with Charset Param", "text/event-stream; charset=utf-8"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			reqBody := `{"question": "What are the architectural trade-offs of CouncilAI?"}`
			req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", tc.acceptHeader)

			w := httptest.NewRecorder()
			h.HandleQuery(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusOK {
				t.Fatalf("expected status 200, got %d", res.StatusCode)
			}

			expectedHeaders := map[string]string{
				"Content-Type":      "text/event-stream; charset=utf-8",
				"Cache-Control":     "no-cache",
				"Connection":        "keep-alive",
				"X-Accel-Buffering": "no",
			}

			for key, expectedVal := range expectedHeaders {
				got := res.Header.Get(key)
				if got != expectedVal {
					t.Errorf("header %s: expected %q, got %q", key, expectedVal, got)
				}
			}
		})
	}
}

func verifyDeliberationCandidateDraft(t *testing.T, dataStr string) {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
		t.Errorf("candidate_draft data failed to unmarshal: %v", err)
		return
	}
	if data["model"] == nil || data["answer"] == nil {
		t.Errorf("candidate_draft missing model or answer: %v", data)
	}
	if data["model_name"] == nil || data["content"] == nil {
		t.Errorf("candidate_draft missing compatibility aliases: %v", data)
	}
}

func verifyDeliberationPeerReview(t *testing.T, dataStr string) {
	var data map[string]interface{}
	if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
		t.Errorf("peer_review data failed to unmarshal: %v", err)
		return
	}
	if data["reviewer"] == nil || data["review"] == nil {
		t.Errorf("peer_review missing reviewer or review: %v", data)
	}
}

func verifyDeliberationFinalAnswer(t *testing.T, dataStr string) {
	var data QueryResponse
	if err := json.Unmarshal([]byte(dataStr), &data); err != nil {
		t.Errorf("final_answer data failed to unmarshal QueryResponse: %v", err)
		return
	}
	if data.Answer != "Final consensus answer" {
		t.Errorf("expected 'Final consensus answer', got %q", data.Answer)
	}
	if data.Confidence != 0.95 {
		t.Errorf("expected confidence 0.95, got %f", data.Confidence)
	}
}

func verifyDeliberationEventSequence(t *testing.T, events []ParsedSSEEvent) {
	candidateCount := 0
	peerReviewCount := 0
	finalAnswerFound := false

	for _, evt := range events {
		switch evt.Event {
		case "candidate_draft":
			candidateCount++
			verifyDeliberationCandidateDraft(t, evt.Data)
		case "peer_review":
			peerReviewCount++
			verifyDeliberationPeerReview(t, evt.Data)
		case "final_answer":
			finalAnswerFound = true
			verifyDeliberationFinalAnswer(t, evt.Data)
		}
	}

	if candidateCount != 2 {
		t.Errorf("expected 2 candidate_draft events, got %d", candidateCount)
	}
	if peerReviewCount != 2 {
		t.Errorf("expected 2 peer_review events, got %d", peerReviewCount)
	}
	if !finalAnswerFound {
		t.Errorf("final_answer event was not emitted")
	}
}

func TestHandleQuery_SSE_FullDeliberationSequence(t *testing.T) {
	mockClients := []llm.LLMClient{
		&handlerMockLLMClient{
			Name: "openrouter:anthropic/claude-3.5-sonnet",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Draft from Claude", Model: "openrouter:anthropic/claude-3.5-sonnet"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return &llm.Response{Answer: "RANKING: A, B\nREASONING: Claude is comprehensive", Model: "openrouter:anthropic/claude-3.5-sonnet"}, nil
			},
		},
		&handlerMockLLMClient{
			Name: "openrouter:openai/gpt-4o",
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Draft from GPT-4o", Model: "openrouter:openai/gpt-4o"}, nil
			},
			GenerateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
				return &llm.Response{Answer: "RANKING: B, A\nREASONING: GPT-4o has better structure", Model: "openrouter:openai/gpt-4o"}, nil
			},
		},
	}
	mockChairman := &handlerMockLLMClient{
		Name: "chairman:gemini-2.0-flash",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"answer":"Final consensus answer","confidence":0.95,"source":"chairman:gemini-2.0-flash","reasoning":"Synthesized from Claude and GPT-4o"}`,
				Model:  "chairman:gemini-2.0-flash",
			}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, mockClients, mockChairman)

	reqBody := `{"question": "Explain multi-agent deliberation trade-offs"}`
	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	events := parseSSEEvents(w.Body.String())
	if len(events) < 5 {
		t.Fatalf("expected at least 5 SSE events (2 candidate_draft + 2 peer_review + 1 final_answer), got %d: %s", len(events), w.Body.String())
	}

	verifyDeliberationEventSequence(t, events)
}

func TestHandleQuery_SSE_TTFT_Under1500ms(t *testing.T) {
	mockClients := []llm.LLMClient{
		&handlerMockLLMClient{
			Name:  "mock:fast-model",
			Delay: 5 * time.Millisecond,
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Fast draft", Model: "mock:fast-model"}, nil
			},
		},
		&handlerMockLLMClient{
			Name:  "mock:slow-model",
			Delay: 30 * time.Millisecond,
			GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
				return &llm.Response{Answer: "Slow draft", Model: "mock:slow-model"}, nil
			},
		},
	}
	mockChairman := &handlerMockLLMClient{
		Name: "mock:chairman",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"answer":"Final synthesized answer"}`,
				Model:  "mock:chairman",
			}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, mockClients, mockChairman)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "test TTFT"}`)).WithContext(ctx)
	req.Header.Set("Accept", "text/event-stream")

	w := NewStreamingRecorder()
	done := make(chan struct{})

	start := time.Now()
	go func() {
		defer close(done)
		h.HandleQuery(w, req)
	}()

	// First flush is the stream header (status 200 OK)
	select {
	case <-w.FlushChan:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for header flush")
	}

	// Second flush is the first candidate_draft frame
	select {
	case <-w.FlushChan:
		ttft := time.Since(start)
		if ttft > 1500*time.Millisecond {
			t.Errorf("TTFT exceeded 1.5s threshold: took %v", ttft)
		}
		events := parseSSEEvents(w.BodyString())
		if len(events) == 0 || events[0].Event != "candidate_draft" {
			t.Errorf("expected first flushed event to be candidate_draft, got: %s", w.BodyString())
		}
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for first candidate_draft event")
	}

	cancel()
	<-done
}

func TestHandleQuery_JSON_BackwardCompatibility(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	h := setupTestHandlersWithCouncil(t, nil, nil)

	cases := []struct {
		name         string
		acceptHeader string
	}{
		{"Explicit JSON Accept", "application/json"},
		{"Omitted Accept Header", ""},
		{"Wildcard Accept Header", "*/*"},
		{"Quality Weighted JSON Accept", "application/json;q=0.9, text/plain;q=0.8"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "What is CouncilAI?"}`))
			if tc.acceptHeader != "" {
				req.Header.Set("Accept", tc.acceptHeader)
			}

			w := httptest.NewRecorder()
			h.HandleQuery(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusOK {
				t.Fatalf("expected status 200, got %d", res.StatusCode)
			}

			ct := res.Header.Get("Content-Type")
			if !strings.HasPrefix(ct, "application/json") {
				t.Errorf("expected Content-Type application/json, got %q", ct)
			}

			body := w.Body.String()
			if strings.HasPrefix(body, "event:") || strings.Contains(body, "data:") {
				t.Errorf("response body contained SSE frames instead of clean JSON: %s", body)
			}

			var resp QueryResponse
			if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
				t.Fatalf("failed to decode JSON response: %v", err)
			}
			if resp.Answer == "" {
				t.Errorf("expected non-empty answer in QueryResponse")
			}
		})
	}
}

func TestHandleQuery_ValidationErrors(t *testing.T) {
	h := setupTestHandlersWithCouncil(t, nil, nil)

	cases := []struct {
		name         string
		body         string
		acceptHeader string
		expectedCode int
		expectedErr  string
	}{
		{
			name:         "Empty Question with SSE Accept",
			body:         `{"question": ""}`,
			acceptHeader: "text/event-stream",
			expectedCode: http.StatusBadRequest,
			expectedErr:  "question is required",
		},
		{
			name:         "Oversized Question with SSE Accept",
			body:         `{"question": "` + strings.Repeat("a", 10001) + `"}`,
			acceptHeader: "text/event-stream",
			expectedCode: http.StatusBadRequest,
			expectedErr:  "question exceeds maximum allowed length of 10000 characters",
		},
		{
			name:         "Malformed JSON with SSE Accept",
			body:         `{invalid_json}`,
			acceptHeader: "text/event-stream",
			expectedCode: http.StatusBadRequest,
			expectedErr:  "invalid request body",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(tc.body))
			req.Header.Set("Accept", tc.acceptHeader)

			w := httptest.NewRecorder()
			h.HandleQuery(w, req)

			res := w.Result()
			if res.StatusCode != tc.expectedCode {
				t.Errorf("expected status %d, got %d", tc.expectedCode, res.StatusCode)
			}
			if ct := res.Header.Get("Content-Type"); ct != "application/json" {
				t.Errorf("expected Content-Type application/json for validation errors, got %s", ct)
			}

			var resp map[string]string
			if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
				t.Fatalf("failed to decode error body: %v", err)
			}
			if resp["error"] != tc.expectedErr {
				t.Errorf("expected error %q, got %q", tc.expectedErr, resp["error"])
			}
		})
	}
}

func TestHandleQuery_DirectStrategy(t *testing.T) {
	mockChairman := &handlerMockLLMClient{
		Name: "mock:direct-model",
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: "Direct response from model",
				Model:  "mock:direct-model",
			}, nil
		},
	}

	h := setupTestHandlersWithCouncil(t, nil, mockChairman)
	mockRouterClient := &handlerMockLLMClient{
		GenerateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"strategy":"direct","reasoning":"simple question","needs_doc":false}`,
			}, nil
		},
	}
	h.Router = agent.NewRouter(mockRouterClient)

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Direct question"}`))
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	events := parseSSEEvents(w.Body.String())
	if len(events) == 0 {
		t.Fatalf("expected SSE events for direct query")
	}

	var foundDraft, foundFinal bool
	for _, ev := range events {
		if ev.Event == "candidate_draft" {
			foundDraft = true
		}
		if ev.Event == "final_answer" {
			foundFinal = true
		}
	}
	if !foundDraft {
		t.Errorf("expected candidate_draft event in direct strategy stream")
	}
	if !foundFinal {
		t.Errorf("expected final_answer event in direct strategy stream")
	}
}

func TestHandleQuery_AllCandidatesFail(t *testing.T) {
	mockClients := []llm.LLMClient{
		&handlerMockLLMClient{Name: "mock:fail-1", Fail: true},
		&handlerMockLLMClient{Name: "mock:fail-2", Fail: true},
	}

	h := setupTestHandlersWithCouncil(t, mockClients, nil)

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Will fail"}`))
	req.Header.Set("Accept", "text/event-stream")

	w := httptest.NewRecorder()
	h.HandleQuery(w, req)

	events := parseSSEEvents(w.Body.String())
	var foundError bool
	for _, ev := range events {
		if ev.Event == "error" {
			foundError = true
			var errPayload map[string]interface{}
			if err := json.Unmarshal([]byte(ev.Data), &errPayload); err != nil {
				t.Errorf("failed to unmarshal error payload: %v", err)
			}
			if errPayload["message"] == nil && errPayload["error"] == nil {
				t.Errorf("expected error or message field in error payload: %v", errPayload)
			}
		}
	}
	if !foundError {
		t.Errorf("expected error event when all candidates fail: %s", w.Body.String())
	}
}

func TestHandleQuery_NonFlusherError(t *testing.T) {
	h := setupTestHandlersWithCouncil(t, nil, nil)

	req := httptest.NewRequest("POST", "/api/v1/query", strings.NewReader(`{"question": "Non flusher"}`))
	req.Header.Set("Accept", "text/event-stream")

	w := newNonFlusherResponseWriter()
	h.HandleQuery(w, req)

	if w.statusCode != http.StatusInternalServerError {
		t.Errorf("expected status 500 when flusher unsupported, got %d", w.statusCode)
	}
	var resp map[string]string
	if err := json.Unmarshal(w.body.Bytes(), &resp); err != nil {
		t.Fatalf("failed to decode error response: %v", err)
	}
	if resp["error"] != "streaming unsupported" {
		t.Errorf("expected 'streaming unsupported', got %q", resp["error"])
	}
}
