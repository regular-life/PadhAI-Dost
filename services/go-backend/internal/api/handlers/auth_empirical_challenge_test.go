package handlers_test

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"golang.org/x/crypto/bcrypt"

	"github.com/regular-life/CouncilAI/go-backend/internal/api/handlers"
	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 1: Duplicate User Registration Returning 409 Conflict
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_DuplicateRegistration_409Conflict(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("test-jwt-secret-empirical-12345", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	// 1. Initial registration
	regPayload := `{"username": "duplicate_target", "password": "TargetPassword123!"}`
	req1 := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(regPayload))
	w1 := httptest.NewRecorder()
	handler.HandleRegister(w1, req1)

	if w1.Result().StatusCode != http.StatusCreated {
		t.Fatalf("initial registration failed: %d %s", w1.Result().StatusCode, w1.Body.String())
	}

	// 2. Sequential duplicate registration attempt
	req2 := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(regPayload))
	w2 := httptest.NewRecorder()
	handler.HandleRegister(w2, req2)

	res2 := w2.Result()
	if res2.StatusCode != http.StatusConflict {
		t.Fatalf("expected 409 Conflict on duplicate registration, got %d. Body: %s", res2.StatusCode, w2.Body.String())
	}

	var errBody map[string]string
	if err := json.NewDecoder(res2.Body).Decode(&errBody); err != nil {
		t.Fatalf("failed to decode error JSON: %v", err)
	}
	if errBody["error"] != "user already exists" {
		t.Errorf("expected error 'user already exists', got %q", errBody["error"])
	}

	// 3. Duplicate with surrounding whitespace in username (should be trimmed to same user)
	trimmedPayload := `{"username": "  duplicate_target  ", "password": "DifferentPassword456!"}`
	req3 := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(trimmedPayload))
	w3 := httptest.NewRecorder()
	handler.HandleRegister(w3, req3)

	if w3.Result().StatusCode != http.StatusConflict {
		t.Fatalf("expected 409 Conflict for whitespace-padded duplicate username, got %d", w3.Result().StatusCode)
	}

	// 4. Concurrent duplicate registration race test (Stampede)
	const concurrentAttempts = 50
	const stampedeUser = "stampede_user_409"
	var createdCount int64
	var conflictCount int64
	var otherCount int64

	var wg sync.WaitGroup
	for i := 0; i < concurrentAttempts; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			body := fmt.Sprintf(`{"username": "%s", "password": "Password_%d!"}`, stampedeUser, idx)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(body))
			w := httptest.NewRecorder()
			handler.HandleRegister(w, req)

			switch w.Result().StatusCode {
			case http.StatusCreated:
				atomic.AddInt64(&createdCount, 1)
			case http.StatusConflict:
				atomic.AddInt64(&conflictCount, 1)
			default:
				atomic.AddInt64(&otherCount, 1)
			}
		}(i)
	}
	wg.Wait()

	if createdCount != 1 {
		t.Errorf("expected exactly 1 successful registration under concurrency, got %d", createdCount)
	}
	if conflictCount != concurrentAttempts-1 {
		t.Errorf("expected %d 409 Conflict responses, got %d", concurrentAttempts-1, conflictCount)
	}
	if otherCount != 0 {
		t.Errorf("expected 0 other response codes, got %d", otherCount)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 2: Non-existent User Login Returning 401 Unauthorized
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_NonExistentUser_401Unauthorized(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("test-jwt-secret-empirical-12345", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	testCases := []struct {
		name     string
		username string
		password string
	}{
		{name: "Standard non-existent user", username: "non_existent_user_999", password: "Password123!"},
		{name: "Special characters in username", username: "user' OR '1'='1", password: "Password123!"},
		{name: "Unicode non-existent user", username: "用户不存在_999", password: "Password123!"},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			body := fmt.Sprintf(`{"username": %q, "password": %q}`, tc.username, tc.password)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
			w := httptest.NewRecorder()
			handler.HandleLogin(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusUnauthorized {
				t.Fatalf("expected 401 Unauthorized for non-existent user, got %d. Body: %s", res.StatusCode, w.Body.String())
			}

			var errBody map[string]string
			if err := json.NewDecoder(res.Body).Decode(&errBody); err != nil {
				t.Fatalf("failed to decode error body: %v", err)
			}
			if errBody["error"] != "invalid credentials" {
				t.Errorf("expected error 'invalid credentials', got %q", errBody["error"])
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 3: Wrong Password Returning 401 Unauthorized
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_WrongPassword_401Unauthorized(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("test-jwt-secret-empirical-12345", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	// Register valid user
	regPayload := `{"username": "victim_user", "password": "CorrectHorseBatteryStaple!"}`
	regReq := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(regPayload))
	regW := httptest.NewRecorder()
	handler.HandleRegister(regW, regReq)
	if regW.Result().StatusCode != http.StatusCreated {
		t.Fatalf("registration failed: %d", regW.Result().StatusCode)
	}

	wrongPasswords := []struct {
		name     string
		password string
	}{
		{name: "Completely wrong password", password: "TotallyWrongPassword999!"},
		{name: "Case difference", password: "correcthorsebatterystaple!"},
		{name: "One character off", password: "CorrectHorseBatteryStaple?"},
		{name: "Trailing space", password: "CorrectHorseBatteryStaple! "},
		{name: "Prefix substring", password: "CorrectHorseBattery"},
	}

	for _, wp := range wrongPasswords {
		wp := wp
		t.Run(wp.name, func(t *testing.T) {
			t.Parallel()
			body := fmt.Sprintf(`{"username": "victim_user", "password": %q}`, wp.password)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
			w := httptest.NewRecorder()
			handler.HandleLogin(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusUnauthorized {
				t.Fatalf("expected 401 Unauthorized for wrong password, got %d. Body: %s", res.StatusCode, w.Body.String())
			}

			var errBody map[string]string
			if err := json.NewDecoder(res.Body).Decode(&errBody); err != nil {
				t.Fatalf("failed to decode error body: %v", err)
			}
			if errBody["error"] != "invalid credentials" {
				t.Errorf("expected error 'invalid credentials', got %q", errBody["error"])
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 4: Malformed / Empty Registration Inputs Returning 400 Bad Request
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_MalformedAndEmptyRegistrationInputs_400BadRequest(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("test-jwt-secret-empirical-12345", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	testCases := []struct {
		name        string
		payload     string
		expectedErr string
	}{
		{
			name:        "Empty JSON object",
			payload:     `{}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Empty username string",
			payload:     `{"username": "", "password": "ValidPassword123!"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Whitespace-only username string",
			payload:     `{"username": "    \t\n  ", "password": "ValidPassword123!"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Empty password string",
			payload:     `{"username": "valid_user", "password": ""}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Missing password field",
			payload:     `{"username": "valid_user"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Missing username field",
			payload:     `{"password": "ValidPassword123!"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Malformed JSON truncated",
			payload:     `{"username": "valid_user", "pass`,
			expectedErr: "invalid request body",
		},
		{
			name:        "Malformed JSON invalid token",
			payload:     `{username: "valid_user"}`,
			expectedErr: "invalid request body",
		},
		{
			name:        "Non-JSON raw string",
			payload:     `Hello world this is not JSON`,
			expectedErr: "invalid request body",
		},
		{
			name:        "Array JSON instead of object",
			payload:     `["username", "password"]`,
			expectedErr: "invalid request body",
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(tc.payload))
			w := httptest.NewRecorder()
			handler.HandleRegister(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusBadRequest {
				t.Fatalf("expected 400 Bad Request, got %d. Body: %s", res.StatusCode, w.Body.String())
			}

			var errBody map[string]string
			if err := json.NewDecoder(res.Body).Decode(&errBody); err != nil {
				t.Fatalf("failed to decode error body: %v", err)
			}
			if errBody["error"] != tc.expectedErr {
				t.Errorf("expected error %q, got %q", tc.expectedErr, errBody["error"])
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 5: Security Response Headers on All Endpoint Outcomes
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_SecurityResponseHeaders_Exhaustive(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("test-jwt-secret-empirical-12345", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	// Pre-seed a user for login tests
	_ = repo.SeedDemoUser(context.Background(), "existing_user", "DemoPassword123!")

	testCases := []struct {
		name           string
		endpoint       string
		handlerFunc    func(http.ResponseWriter, *http.Request)
		payload        string
		expectedStatus int
	}{
		{
			name:           "Register: 201 Created",
			endpoint:       "/api/v1/register",
			handlerFunc:    handler.HandleRegister,
			payload:        `{"username": "brand_new_user", "password": "Password123!"}`,
			expectedStatus: http.StatusCreated,
		},
		{
			name:           "Register: 409 Conflict",
			endpoint:       "/api/v1/register",
			handlerFunc:    handler.HandleRegister,
			payload:        `{"username": "existing_user", "password": "Password123!"}`,
			expectedStatus: http.StatusConflict,
		},
		{
			name:           "Register: 400 Bad Request",
			endpoint:       "/api/v1/register",
			handlerFunc:    handler.HandleRegister,
			payload:        `{"username": ""}`,
			expectedStatus: http.StatusBadRequest,
		},
		{
			name:           "Login: 200 OK",
			endpoint:       "/api/v1/login",
			handlerFunc:    handler.HandleLogin,
			payload:        `{"username": "existing_user", "password": "DemoPassword123!"}`,
			expectedStatus: http.StatusOK,
		},
		{
			name:           "Login: 401 Unauthorized (Wrong Password)",
			endpoint:       "/api/v1/login",
			handlerFunc:    handler.HandleLogin,
			payload:        `{"username": "existing_user", "password": "WrongPassword999!"}`,
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "Login: 401 Unauthorized (Non-existent user)",
			endpoint:       "/api/v1/login",
			handlerFunc:    handler.HandleLogin,
			payload:        `{"username": "ghost_user", "password": "Password123!"}`,
			expectedStatus: http.StatusUnauthorized,
		},
		{
			name:           "Login: 400 Bad Request",
			endpoint:       "/api/v1/login",
			handlerFunc:    handler.HandleLogin,
			payload:        `{"password": "missing_username"}`,
			expectedStatus: http.StatusBadRequest,
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, tc.endpoint, strings.NewReader(tc.payload))
			w := httptest.NewRecorder()
			tc.handlerFunc(w, req)

			res := w.Result()
			if res.StatusCode != tc.expectedStatus {
				t.Fatalf("expected status %d, got %d. Body: %s", tc.expectedStatus, res.StatusCode, w.Body.String())
			}

			// Validate security headers
			if ct := res.Header.Get("Content-Type"); ct != "application/json" {
				t.Errorf("missing or invalid Content-Type: want 'application/json', got %q", ct)
			}
			if nosniff := res.Header.Get("X-Content-Type-Options"); nosniff != "nosniff" {
				t.Errorf("missing or invalid X-Content-Type-Options: want 'nosniff', got %q", nosniff)
			}
			if frame := res.Header.Get("X-Frame-Options"); frame != "DENY" {
				t.Errorf("missing or invalid X-Frame-Options: want 'DENY', got %q", frame)
			}
			if xss := res.Header.Get("X-XSS-Protection"); xss != "1; mode=block" {
				t.Errorf("missing or invalid X-XSS-Protection: want '1; mode=block', got %q", xss)
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 6: Bcrypt Hashing Behavior & Execution Latency
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_BcryptHashingBehaviorAndLatency(t *testing.T) {
	t.Parallel()

	password := "CouncilAIPasswordStressTest!@#456"

	// 1. Measure Bcrypt Generation Latency
	startGen := time.Now()
	hashBytes, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	genDuration := time.Since(startGen)

	if err != nil {
		t.Fatalf("bcrypt generation failed: %v", err)
	}

	hashStr := string(hashBytes)
	if !strings.HasPrefix(hashStr, "$2a$10$") && !strings.HasPrefix(hashStr, "$2b$10$") {
		t.Errorf("expected bcrypt cost 10 prefix ($2a$10$ or $2b$10$), got: %s", hashStr[:8])
	}

	cost, err := bcrypt.Cost(hashBytes)
	if err != nil {
		t.Fatalf("failed to read bcrypt cost: %v", err)
	}
	if cost != bcrypt.DefaultCost {
		t.Errorf("expected cost %d, got %d", bcrypt.DefaultCost, cost)
	}

	t.Logf("Empirical Bcrypt GenerateFromPassword (cost %d) latency: %v", cost, genDuration)

	// 2. Measure Bcrypt Verification Latency (Matching Password)
	startVer := time.Now()
	err = bcrypt.CompareHashAndPassword(hashBytes, []byte(password))
	verDuration := time.Since(startVer)
	if err != nil {
		t.Fatalf("bcrypt compare failed for valid password: %v", err)
	}

	t.Logf("Empirical Bcrypt CompareHashAndPassword (matching) latency: %v", verDuration)

	// 3. Measure Bcrypt Verification Latency (Mismatch Password)
	startMis := time.Now()
	err = bcrypt.CompareHashAndPassword(hashBytes, []byte("WrongPassword123!"))
	misDuration := time.Since(startMis)
	if err == nil {
		t.Fatal("expected bcrypt compare to fail for wrong password, got nil")
	}

	t.Logf("Empirical Bcrypt CompareHashAndPassword (mismatch) latency: %v", misDuration)

	// Bcrypt cost 10 latency should be between 10ms and 500ms on any modern server CPU
	if genDuration < 5*time.Millisecond || genDuration > 1000*time.Millisecond {
		t.Logf("Warning: Bcrypt generation latency %v is outside standard range (5ms-1000ms)", genDuration)
	}

	// 4. Bcrypt 72-byte password boundary test
	// Bcrypt historically has a 72-byte truncation limit.
	base72 := strings.Repeat("A", 72)
	extended80 := base72 + "EXTRA8BYTES"

	h72, err := bcrypt.GenerateFromPassword([]byte(base72), bcrypt.DefaultCost)
	if err != nil {
		t.Fatalf("failed to hash 72-byte password: %v", err)
	}

	// In standard bcrypt, comparing 72-byte hash against extended 80-byte password succeeds due to 72-byte truncation limit!
	truncationCheck := bcrypt.CompareHashAndPassword(h72, []byte(extended80))
	if truncationCheck == nil {
		t.Logf("Observed: Standard Bcrypt 72-byte password truncation behavior (passwords > 72 bytes match on first 72 bytes). Documenting finding.")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario 7: Timing Discrepancy Analysis (User Exists vs User Not Found)
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_TimingDiscrepancy_UserExistsVsNotFound(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("test-jwt-secret-empirical-12345", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	// Register existing user
	_ = repo.SeedDemoUser(context.Background(), "timing_test_user", "SecretPass123!")

	// 1. Measure latency for login with existing user + wrong password (triggers bcrypt.CompareHashAndPassword)
	wrongPassBody := `{"username": "timing_test_user", "password": "WrongPassword999!"}`
	startWrongPass := time.Now()
	req1 := httptest.NewRequest(http.MethodPost, "/api/v1/login", bytes.NewBufferString(wrongPassBody))
	w1 := httptest.NewRecorder()
	handler.HandleLogin(w1, req1)
	latencyWrongPass := time.Since(startWrongPass)

	if w1.Result().StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", w1.Result().StatusCode)
	}

	// 2. Measure latency for login with non-existent user (does not trigger bcrypt)
	notFoundBody := `{"username": "ghost_timing_user", "password": "WrongPassword999!"}`
	startNotFound := time.Now()
	req2 := httptest.NewRequest(http.MethodPost, "/api/v1/login", bytes.NewBufferString(notFoundBody))
	w2 := httptest.NewRecorder()
	handler.HandleLogin(w2, req2)
	latencyNotFound := time.Since(startNotFound)

	if w2.Result().StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected 401, got %d", w2.Result().StatusCode)
	}

	t.Logf("Empirical Login Timing: Non-existent user latency = %v, Wrong password latency = %v", latencyNotFound, latencyWrongPass)
	t.Logf("Timing Delta: %v (Bcrypt difference provides username enumeration oracle if unprotected by rate limiting)", latencyWrongPass-latencyNotFound)
}
