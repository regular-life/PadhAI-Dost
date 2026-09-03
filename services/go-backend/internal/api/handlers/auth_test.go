package handlers_test

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
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
