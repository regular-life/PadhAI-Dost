package handlers_test

import (
	"bytes"
	"context"
	"encoding/base64"
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
	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

const (
	testJWTSecret = "test-jwt-secret-key-councilai-unit-tests"
	testJWTTTL    = 1 * time.Hour
)

type authTestFixture struct {
	jwtManager *auth.JWTManager
	repo       auth.UserRepository
	handler    *handlers.AuthHandler
}

func newAuthTestFixture(t *testing.T) *authTestFixture {
	t.Helper()
	jwtMgr := auth.NewJWTManager(testJWTSecret, testJWTTTL)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)
	return &authTestFixture{
		jwtManager: jwtMgr,
		repo:       repo,
		handler:    handler,
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 1. HandleRegister Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestHandleRegister_Success(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	reqBody := `{"username": "alice", "password": "SuperSecretPassword123!"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	f.handler.HandleRegister(w, req)

	res := w.Result()
	if res.StatusCode != http.StatusCreated {
		t.Fatalf("expected status 201 Created, got %d. Body: %s", res.StatusCode, w.Body.String())
	}

	var resp map[string]string
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode JSON response: %v", err)
	}

	if resp["user_id"] != "alice" {
		t.Errorf("expected user_id 'alice', got %q", resp["user_id"])
	}
	if resp["message"] != "user created successfully" {
		t.Errorf("expected message 'user created successfully', got %q", resp["message"])
	}
	token := resp["token"]
	if token == "" {
		t.Fatal("expected non-empty JWT token string")
	}

	// Validate JWT Token and claims
	claims, err := f.jwtManager.ValidateToken(token)
	if err != nil {
		t.Fatalf("failed to validate issued JWT token: %v", err)
	}
	if claims.UserID != "alice" {
		t.Errorf("expected claims.UserID 'alice', got %q", claims.UserID)
	}
	if claims.Issuer != "council-ai" {
		t.Errorf("expected claims.Issuer 'council-ai', got %q", claims.Issuer)
	}

	// Verify User persistence in repository
	ctx := context.Background()
	user, err := f.repo.GetUserByUsername(ctx, "alice")
	if err != nil {
		t.Fatalf("expected user 'alice' in repository, got error: %v", err)
	}
	if user.Username != "alice" {
		t.Errorf("expected user.Username 'alice', got %q", user.Username)
	}
	if user.ID == "" {
		t.Error("expected non-empty user ID")
	}
	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte("SuperSecretPassword123!")); err != nil {
		t.Errorf("stored password hash does not match original password: %v", err)
	}
}

func TestHandleRegister_DuplicateUsername_409Conflict(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	// Register initial user
	reqBody := `{"username": "bob", "password": "Password123!"}`
	req1 := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(reqBody))
	req1.Header.Set("Content-Type", "application/json")
	w1 := httptest.NewRecorder()
	f.handler.HandleRegister(w1, req1)
	if w1.Result().StatusCode != http.StatusCreated {
		t.Fatalf("initial registration failed: %s", w1.Body.String())
	}

	// Attempt duplicate registration
	req2 := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(reqBody))
	req2.Header.Set("Content-Type", "application/json")
	w2 := httptest.NewRecorder()
	f.handler.HandleRegister(w2, req2)

	res := w2.Result()
	if res.StatusCode != http.StatusConflict {
		t.Fatalf("expected status 409 Conflict for duplicate username, got %d. Body: %s", res.StatusCode, w2.Body.String())
	}

	var errResp map[string]string
	if err := json.NewDecoder(res.Body).Decode(&errResp); err != nil {
		t.Fatalf("failed to decode error body: %v", err)
	}
	if errResp["error"] != "user already exists" {
		t.Errorf("expected error 'user already exists', got %q", errResp["error"])
	}
}

func TestHandleRegister_InvalidInput_400BadRequest(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	testCases := []struct {
		name        string
		body        string
		expectedErr string
	}{
		{
			name:        "Missing username",
			body:        `{"password": "Password123!"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Missing password",
			body:        `{"username": "carol"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Empty username string",
			body:        `{"username": "", "password": "Password123!"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Empty password string",
			body:        `{"username": "carol", "password": ""}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Whitespace-only username",
			body:        `{"username": "   ", "password": "Password123!"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Empty JSON object",
			body:        `{}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Malformed JSON syntax",
			body:        `{"username": "carol", "password":`,
			expectedErr: "invalid request body",
		},
		{
			name:        "Plain text non-JSON payload",
			body:        `not_a_json_payload`,
			expectedErr: "invalid request body",
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			f.handler.HandleRegister(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusBadRequest {
				t.Errorf("expected status 400 Bad Request, got %d", res.StatusCode)
			}

			var errResp map[string]string
			if err := json.NewDecoder(res.Body).Decode(&errResp); err != nil {
				t.Fatalf("failed to decode error body: %v", err)
			}
			if errResp["error"] != tc.expectedErr {
				t.Errorf("expected error %q, got %q", tc.expectedErr, errResp["error"])
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 2. HandleLogin Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestHandleLogin_Success(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	// Register user
	regBody := `{"username": "dan", "password": "MySecurePassword789!"}`
	regReq := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(regBody))
	regW := httptest.NewRecorder()
	f.handler.HandleRegister(regW, regReq)

	// Perform login
	loginBody := `{"username": "dan", "password": "MySecurePassword789!"}`
	loginReq := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(loginBody))
	loginReq.Header.Set("Content-Type", "application/json")
	loginW := httptest.NewRecorder()

	f.handler.HandleLogin(loginW, loginReq)

	res := loginW.Result()
	if res.StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 OK, got %d. Body: %s", res.StatusCode, loginW.Body.String())
	}

	var resp map[string]string
	if err := json.NewDecoder(res.Body).Decode(&resp); err != nil {
		t.Fatalf("failed to decode login response: %v", err)
	}
	if resp["user_id"] != "dan" {
		t.Errorf("expected user_id 'dan', got %q", resp["user_id"])
	}
	if resp["token"] == "" {
		t.Fatal("expected non-empty token")
	}

	claims, err := f.jwtManager.ValidateToken(resp["token"])
	if err != nil {
		t.Fatalf("failed to validate login token: %v", err)
	}
	if claims.UserID != "dan" {
		t.Errorf("expected claims.UserID 'dan', got %q", claims.UserID)
	}
}

func TestHandleLogin_IncorrectPassword_401Unauthorized(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	// Register user
	regBody := `{"username": "eve", "password": "CorrectPassword123"}`
	regReq := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(regBody))
	regW := httptest.NewRecorder()
	f.handler.HandleRegister(regW, regReq)

	// Attempt login with wrong password
	loginBody := `{"username": "eve", "password": "IncorrectPassword999"}`
	loginReq := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(loginBody))
	loginW := httptest.NewRecorder()

	f.handler.HandleLogin(loginW, loginReq)

	res := loginW.Result()
	if res.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected status 401 Unauthorized for incorrect password, got %d", res.StatusCode)
	}

	var errResp map[string]string
	if err := json.NewDecoder(res.Body).Decode(&errResp); err != nil {
		t.Fatalf("failed to decode error body: %v", err)
	}
	if errResp["error"] != "invalid credentials" {
		t.Errorf("expected error 'invalid credentials', got %q", errResp["error"])
	}
}

func TestHandleLogin_NonExistentUser_401Unauthorized(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	loginBody := `{"username": "ghost_user", "password": "AnyPassword123"}`
	loginReq := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(loginBody))
	loginW := httptest.NewRecorder()

	f.handler.HandleLogin(loginW, loginReq)

	res := loginW.Result()
	if res.StatusCode != http.StatusUnauthorized {
		t.Fatalf("expected status 401 Unauthorized for non-existent user, got %d", res.StatusCode)
	}

	var errResp map[string]string
	if err := json.NewDecoder(res.Body).Decode(&errResp); err != nil {
		t.Fatalf("failed to decode error body: %v", err)
	}
	if errResp["error"] != "invalid credentials" {
		t.Errorf("expected error 'invalid credentials', got %q", errResp["error"])
	}
}

func TestHandleLogin_InvalidInput_400BadRequest(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	testCases := []struct {
		name        string
		body        string
		expectedErr string
	}{
		{
			name:        "Missing username",
			body:        `{"password": "Password123!"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Missing password",
			body:        `{"username": "frank"}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Empty JSON body",
			body:        `{}`,
			expectedErr: "username and password are required",
		},
		{
			name:        "Malformed JSON syntax",
			body:        `{"username": "frank", "password":`,
			expectedErr: "invalid request body",
		},
	}

	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()
			req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(tc.body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			f.handler.HandleLogin(w, req)

			res := w.Result()
			if res.StatusCode != http.StatusBadRequest {
				t.Errorf("expected status 400 Bad Request, got %d", res.StatusCode)
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 3. Security Headers & Protection Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestAuth_SecurityHeaders(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	endpoints := []struct {
		name    string
		handler func(http.ResponseWriter, *http.Request)
		body    string
	}{
		{
			name:    "HandleRegister",
			handler: f.handler.HandleRegister,
			body:    `{"username": "sec_user", "password": "Password123!"}`,
		},
		{
			name:    "HandleLogin",
			handler: f.handler.HandleLogin,
			body:    `{"username": "sec_user", "password": "Password123!"}`,
		},
	}

	for _, ep := range endpoints {
		t.Run(ep.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/test", strings.NewReader(ep.body))
			w := httptest.NewRecorder()
			ep.handler(w, req)

			res := w.Result()
			if ct := res.Header.Get("Content-Type"); ct != "application/json" {
				t.Errorf("expected Content-Type application/json, got %q", ct)
			}
			if nosniff := res.Header.Get("X-Content-Type-Options"); nosniff != "nosniff" {
				t.Errorf("expected X-Content-Type-Options nosniff, got %q", nosniff)
			}
			if frame := res.Header.Get("X-Frame-Options"); frame != "DENY" {
				t.Errorf("expected X-Frame-Options DENY, got %q", frame)
			}
			if xss := res.Header.Get("X-XSS-Protection"); xss != "1; mode=block" {
				t.Errorf("expected X-XSS-Protection 1; mode=block, got %q", xss)
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// 4. JWT Claim Propagation via Middleware Integration
// ─────────────────────────────────────────────────────────────────────────────

func TestAuth_JWTClaimExtraction_ThroughMiddleware(t *testing.T) {
	t.Parallel()
	f := newAuthTestFixture(t)

	// Register user and obtain token
	regBody := `{"username": "grace", "password": "Password123!"}`
	regReq := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(regBody))
	regW := httptest.NewRecorder()
	f.handler.HandleRegister(regW, regReq)

	var regResp map[string]string
	_ = json.NewDecoder(regW.Body).Decode(&regResp)
	token := regResp["token"]

	// Create protected test handler wrapped with AuthMiddleware
	var capturedUserID string
	protectedHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedUserID = middleware.GetUserID(r.Context())
		w.WriteHeader(http.StatusOK)
	})

	authMw := middleware.AuthMiddleware(f.jwtManager)
	server := authMw(protectedHandler)

	// Issue request with Bearer token
	testReq := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
	testReq.Header.Set("Authorization", "Bearer "+token)
	testW := httptest.NewRecorder()

	server.ServeHTTP(testW, testReq)

	if testW.Result().StatusCode != http.StatusOK {
		t.Fatalf("expected status 200 OK from protected endpoint, got %d", testW.Result().StatusCode)
	}
	if capturedUserID != "grace" {
		t.Errorf("expected captured UserID in context to be 'grace', got %q", capturedUserID)
	}
}

// ── Concurrency & Data Race Safety Tests ───────────────────

// ─────────────────────────────────────────────────────────────────────────────
// Challenger 2: Scenario 1 - High Concurrency HTTP Handlers & Data Race Safety
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalC2_HTTP_HighConcurrencyRegistrationsAndLogins_200Goroutines(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("concurrency-stress-jwt-secret-key-32b", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	const numUsers = 200
	var wg sync.WaitGroup

	// Phase 1: 200 Concurrent Registrations
	regSuccessCount := int64(0)
	for i := 0; i < numUsers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uname := fmt.Sprintf("concurrent_user_%d", idx)
			pass := fmt.Sprintf("StrongPassword#%d!", idx)

			body := fmt.Sprintf(`{"username": "%s", "password": "%s"}`, uname, pass)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			handler.HandleRegister(w, req)

			res := w.Result()
			if res.StatusCode == http.StatusCreated {
				atomic.AddInt64(&regSuccessCount, 1)

				var resp map[string]string
				if err := json.NewDecoder(res.Body).Decode(&resp); err == nil {
					if resp["user_id"] != uname || resp["token"] == "" {
						t.Errorf("invalid register response payload for %s: %+v", uname, resp)
					}
				}
			} else {
				t.Errorf("registration failed for %s with status %d: %s", uname, res.StatusCode, w.Body.String())
			}
		}(i)
	}
	wg.Wait()

	if regSuccessCount != numUsers {
		t.Fatalf("expected %d successful registrations, got %d", numUsers, regSuccessCount)
	}

	// Phase 2: 200 Concurrent Logins
	loginSuccessCount := int64(0)
	for i := 0; i < numUsers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uname := fmt.Sprintf("concurrent_user_%d", idx)
			pass := fmt.Sprintf("StrongPassword#%d!", idx)

			body := fmt.Sprintf(`{"username": "%s", "password": "%s"}`, uname, pass)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			handler.HandleLogin(w, req)

			res := w.Result()
			if res.StatusCode == http.StatusOK {
				atomic.AddInt64(&loginSuccessCount, 1)

				var resp map[string]string
				if err := json.NewDecoder(res.Body).Decode(&resp); err == nil {
					if resp["user_id"] != uname || resp["token"] == "" {
						t.Errorf("invalid login response payload for %s: %+v", uname, resp)
					}
					// Validate issued token
					claims, err := jwtMgr.ValidateToken(resp["token"])
					if err != nil || claims.UserID != uname {
						t.Errorf("invalid token issued during concurrent login for %s: %v", uname, err)
					}
				}
			} else {
				t.Errorf("login failed for %s with status %d: %s", uname, res.StatusCode, w.Body.String())
			}
		}(i)
	}
	wg.Wait()

	if loginSuccessCount != numUsers {
		t.Fatalf("expected %d successful logins, got %d", numUsers, loginSuccessCount)
	}
}

func TestEmpiricalC2_HTTP_MassiveStampedeDuplicateRegistration_100Goroutines(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("stampede-http-jwt-secret-key-32b", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	const numRequests = 100
	const contendedUsername = "stampede_http_user"
	var (
		wg            sync.WaitGroup
		createdCount  int64
		conflictCount int64
		otherCount    int64
	)

	for i := 0; i < numRequests; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			body := fmt.Sprintf(`{"username": "%s", "password": "Password_%d!"}`, contendedUsername, idx)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(body))
			req.Header.Set("Content-Type", "application/json")
			w := httptest.NewRecorder()

			handler.HandleRegister(w, req)

			res := w.Result()
			switch res.StatusCode {
			case http.StatusCreated:
				atomic.AddInt64(&createdCount, 1)
			case http.StatusConflict:
				atomic.AddInt64(&conflictCount, 1)
				var errResp map[string]string
				if err := json.NewDecoder(res.Body).Decode(&errResp); err == nil {
					if errResp["error"] != "user already exists" {
						t.Errorf("expected error 'user already exists', got %q", errResp["error"])
					}
				}
			default:
				atomic.AddInt64(&otherCount, 1)
				t.Errorf("unexpected status %d: %s", res.StatusCode, w.Body.String())
			}
		}(i)
	}
	wg.Wait()

	if createdCount != 1 {
		t.Fatalf("expected exactly 1 winner for HTTP stampede registration, got %d", createdCount)
	}
	if conflictCount != numRequests-1 {
		t.Fatalf("expected %d 409 Conflict responses, got %d", numRequests-1, conflictCount)
	}
	if otherCount != 0 {
		t.Fatalf("expected 0 other response codes, got %d", otherCount)
	}
}

func runMixedTrafficAuthAction(t *testing.T, workerID int, handler *handlers.AuthHandler) {
	switch workerID % 4 {
	case 0:
		uname := fmt.Sprintf("mixed_reg_%d", workerID)
		body := fmt.Sprintf(`{"username": "%s", "password": "MixedPassword123!"}`, uname)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(body))
		w := httptest.NewRecorder()
		handler.HandleRegister(w, req)
		if w.Result().StatusCode != http.StatusCreated {
			t.Errorf("mixed reg failed for %s: %d", uname, w.Result().StatusCode)
		}
	case 1:
		uname := fmt.Sprintf("preseed_user_%d", workerID%20)
		body := fmt.Sprintf(`{"username": "%s", "password": "PreseedPass123!"}`, uname)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
		w := httptest.NewRecorder()
		handler.HandleLogin(w, req)
		if w.Result().StatusCode != http.StatusOK {
			t.Errorf("mixed login failed for %s: %d", uname, w.Result().StatusCode)
		}
	case 2:
		uname := fmt.Sprintf("preseed_user_%d", workerID%20)
		body := fmt.Sprintf(`{"username": "%s", "password": "WrongPassword999!"}`, uname)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
		w := httptest.NewRecorder()
		handler.HandleLogin(w, req)
		if w.Result().StatusCode != http.StatusUnauthorized {
			t.Errorf("expected 401 for wrong pass, got %d", w.Result().StatusCode)
		}
	case 3:
		uname := fmt.Sprintf("ghost_worker_user_%d", workerID)
		body := fmt.Sprintf(`{"username": "%s", "password": "Password123!"}`, uname)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
		w := httptest.NewRecorder()
		handler.HandleLogin(w, req)
		if w.Result().StatusCode != http.StatusUnauthorized {
			t.Errorf("expected 401 for ghost user, got %d", w.Result().StatusCode)
		}
	}
}

func runMixedTrafficProtectedAction(t *testing.T, workerID int, protectedEndpoint http.Handler, jwtMgr *auth.JWTManager) {
	switch workerID % 4 {
	case 0:
		token, _ := jwtMgr.GenerateToken(fmt.Sprintf("preseed_user_%d", workerID%20))
		req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
		req.Header.Set("Authorization", "Bearer "+token)
		w := httptest.NewRecorder()
		protectedEndpoint.ServeHTTP(w, req)
		if w.Result().StatusCode != http.StatusOK {
			t.Errorf("expected 200 OK from protected endpoint, got %d", w.Result().StatusCode)
		}
	case 1:
		token, _ := jwtMgr.GenerateToken("valid_user")
		req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
		req.Header.Set("Authorization", "Bearer "+token+"corrupted_sig")
		w := httptest.NewRecorder()
		protectedEndpoint.ServeHTTP(w, req)
		if w.Result().StatusCode != http.StatusUnauthorized {
			t.Errorf("expected 401 for tampered token, got %d", w.Result().StatusCode)
		}
	case 2:
		expiredMgr := auth.NewJWTManager("mixed-traffic-jwt-secret-key-32b", -5*time.Minute)
		expiredToken, _ := expiredMgr.GenerateToken("expired_user")
		req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
		req.Header.Set("Authorization", "Bearer "+expiredToken)
		w := httptest.NewRecorder()
		protectedEndpoint.ServeHTTP(w, req)
		if w.Result().StatusCode != http.StatusUnauthorized {
			t.Errorf("expected 401 for expired token, got %d", w.Result().StatusCode)
		}
	case 3:
		req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
		req.Header.Set("Authorization", "InvalidScheme xyz")
		w := httptest.NewRecorder()
		protectedEndpoint.ServeHTTP(w, req)
		if w.Result().StatusCode != http.StatusUnauthorized {
			t.Errorf("expected 401 for invalid scheme, got %d", w.Result().StatusCode)
		}
	}
}

func runMixedTrafficWorker(t *testing.T, workerID int, handler *handlers.AuthHandler, protectedEndpoint http.Handler, jwtMgr *auth.JWTManager) {
	if workerID%2 == 0 {
		runMixedTrafficAuthAction(t, workerID, handler)
	} else {
		runMixedTrafficProtectedAction(t, workerID, protectedEndpoint, jwtMgr)
	}
}

func TestEmpiricalC2_HTTP_MixedConcurrentTraffic_Interleaved(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("mixed-traffic-jwt-secret-key-32b", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	for i := 0; i < 20; i++ {
		_ = repo.SeedDemoUser(context.Background(), fmt.Sprintf("preseed_user_%d", i), "PreseedPass123!")
	}

	protectedHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		uid := middleware.GetUserID(r.Context())
		if uid == "anonymous" {
			http.Error(w, "forbidden", http.StatusForbidden)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_ = json.NewEncoder(w).Encode(map[string]string{"status": "ok", "user": uid})
	})
	authMw := middleware.AuthMiddleware(jwtMgr)
	protectedEndpoint := authMw(protectedHandler)

	const totalWorkers = 160
	var wg sync.WaitGroup

	for i := 0; i < totalWorkers; i++ {
		wg.Add(1)
		workerID := i
		go func() {
			defer wg.Done()
			runMixedTrafficWorker(t, workerID, handler, protectedEndpoint, jwtMgr)
		}()
	}
	wg.Wait()
}

// ─────────────────────────────────────────────────────────────────────────────
// Challenger 2: Scenario 2 - Middleware Header & Claim Extraction Security
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalC2_Middleware_AuthorizationHeaderExhaustiveMatrix(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("middleware-test-secret-32b-length", 1*time.Hour)
	validToken, err := jwtMgr.GenerateToken("auth_matrix_user")
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	expiredMgr := auth.NewJWTManager("middleware-test-secret-32b-length", -10*time.Minute)
	expiredToken, _ := expiredMgr.GenerateToken("expired_user")

	var capturedUser string
	protectedHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedUser = middleware.GetUserID(r.Context())
		w.WriteHeader(http.StatusOK)
	})
	authMw := middleware.AuthMiddleware(jwtMgr)
	server := authMw(protectedHandler)

	matrix := []struct {
		name           string
		authHeader     string
		setHeader      bool
		expectedStatus int
		expectedError  string
		expectedUser   string
	}{
		{
			name:           "Valid Bearer token standard case",
			authHeader:     "Bearer " + validToken,
			setHeader:      true,
			expectedStatus: http.StatusOK,
			expectedUser:   "auth_matrix_user",
		},
		{
			name:           "Valid bearer token lowercase case",
			authHeader:     "bearer " + validToken,
			setHeader:      true,
			expectedStatus: http.StatusOK,
			expectedUser:   "auth_matrix_user",
		},
		{
			name:           "Valid BEARER token uppercase case",
			authHeader:     "BEARER " + validToken,
			setHeader:      true,
			expectedStatus: http.StatusOK,
			expectedUser:   "auth_matrix_user",
		},
		{
			name:           "Missing Authorization header",
			authHeader:     "",
			setHeader:      false,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "missing authorization header",
		},
		{
			name:           "Empty string Authorization header",
			authHeader:     "",
			setHeader:      true,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "missing authorization header",
		},
		{
			name:           "Single word 'Bearer' (no token)",
			authHeader:     "Bearer",
			setHeader:      true,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "invalid authorization format",
		},
		{
			name:           "Single word 'Token' (unknown scheme)",
			authHeader:     "Token",
			setHeader:      true,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "invalid authorization format",
		},
		{
			name:           "Basic auth scheme instead of Bearer",
			authHeader:     "Basic " + base64.StdEncoding.EncodeToString([]byte("user:pass")),
			setHeader:      true,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "invalid authorization format",
		},
		{
			name:           "Expired JWT token",
			authHeader:     "Bearer " + expiredToken,
			setHeader:      true,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "invalid or expired token",
		},
		{
			name:           "Tampered JWT signature",
			authHeader:     "Bearer " + validToken + "_tampered",
			setHeader:      true,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "invalid or expired token",
		},
		{
			name:           "Three parts in header (extra tokens)",
			authHeader:     "Bearer token1 token2",
			setHeader:      true,
			expectedStatus: http.StatusUnauthorized,
			expectedError:  "invalid or expired token",
		},
	}

	for _, tc := range matrix {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			capturedUser = ""
			req := httptest.NewRequest(http.MethodGet, "/api/v1/resource", nil)
			if tc.setHeader {
				req.Header.Set("Authorization", tc.authHeader)
			}
			w := httptest.NewRecorder()

			server.ServeHTTP(w, req)

			res := w.Result()
			if res.StatusCode != tc.expectedStatus {
				t.Fatalf("expected status %d, got %d. Body: %s", tc.expectedStatus, res.StatusCode, w.Body.String())
			}

			if tc.expectedStatus == http.StatusOK {
				if capturedUser != tc.expectedUser {
					t.Errorf("expected captured UserID %q, got %q", tc.expectedUser, capturedUser)
				}
			} else {
				var errResp map[string]string
				if err := json.NewDecoder(res.Body).Decode(&errResp); err != nil {
					t.Fatalf("failed to decode error body: %v", err)
				}
				if errResp["error"] != tc.expectedError {
					t.Errorf("expected error %q, got %q", tc.expectedError, errResp["error"])
				}
			}
		})
	}
}

func TestEmpiricalC2_Middleware_GetUserID_UnauthenticatedContext(t *testing.T) {
	t.Parallel()
	// An empty context without UserIDKey must safely return "anonymous" without panic
	ctx := context.Background()
	uid := middleware.GetUserID(ctx)
	if uid != "anonymous" {
		t.Errorf("expected 'anonymous' for empty context, got %q", uid)
	}
}

// ── Empirical Security & Edge Invariants ──────────────────

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
