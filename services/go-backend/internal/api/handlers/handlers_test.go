package handlers

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
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


