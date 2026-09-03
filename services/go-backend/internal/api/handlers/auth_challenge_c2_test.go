package handlers_test

import (
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

	"github.com/regular-life/CouncilAI/go-backend/internal/api/handlers"
	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

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

func TestEmpiricalC2_HTTP_MixedConcurrentTraffic_Interleaved(t *testing.T) {
	t.Parallel()
	jwtMgr := auth.NewJWTManager("mixed-traffic-jwt-secret-key-32b", 1*time.Hour)
	repo := auth.NewMemoryUserRepository()
	handler := handlers.NewAuthHandler(jwtMgr, repo)

	// Pre-seed some existing users
	for i := 0; i < 20; i++ {
		_ = repo.SeedDemoUser(context.Background(), fmt.Sprintf("preseed_user_%d", i), "PreseedPass123!")
	}

	// Protected handler behind AuthMiddleware
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
			switch workerID % 8 {
			case 0:
				// Action: Register new user
				uname := fmt.Sprintf("mixed_reg_%d", workerID)
				body := fmt.Sprintf(`{"username": "%s", "password": "MixedPassword123!"}`, uname)
				req := httptest.NewRequest(http.MethodPost, "/api/v1/register", strings.NewReader(body))
				w := httptest.NewRecorder()
				handler.HandleRegister(w, req)
				if w.Result().StatusCode != http.StatusCreated {
					t.Errorf("mixed reg failed for %s: %d", uname, w.Result().StatusCode)
				}

			case 1:
				// Action: Login with preseeded user
				uname := fmt.Sprintf("preseed_user_%d", workerID%20)
				body := fmt.Sprintf(`{"username": "%s", "password": "PreseedPass123!"}`, uname)
				req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
				w := httptest.NewRecorder()
				handler.HandleLogin(w, req)
				if w.Result().StatusCode != http.StatusOK {
					t.Errorf("mixed login failed for %s: %d", uname, w.Result().StatusCode)
				}

			case 2:
				// Action: Login with wrong password (expect 401)
				uname := fmt.Sprintf("preseed_user_%d", workerID%20)
				body := fmt.Sprintf(`{"username": "%s", "password": "WrongPassword999!"}`, uname)
				req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
				w := httptest.NewRecorder()
				handler.HandleLogin(w, req)
				if w.Result().StatusCode != http.StatusUnauthorized {
					t.Errorf("expected 401 for wrong pass, got %d", w.Result().StatusCode)
				}

			case 3:
				// Action: Login with non-existent user (expect 401)
				uname := fmt.Sprintf("ghost_worker_user_%d", workerID)
				body := fmt.Sprintf(`{"username": "%s", "password": "Password123!"}`, uname)
				req := httptest.NewRequest(http.MethodPost, "/api/v1/login", strings.NewReader(body))
				w := httptest.NewRecorder()
				handler.HandleLogin(w, req)
				if w.Result().StatusCode != http.StatusUnauthorized {
					t.Errorf("expected 401 for ghost user, got %d", w.Result().StatusCode)
				}

			case 4:
				// Action: Access protected endpoint with valid JWT token
				token, _ := jwtMgr.GenerateToken(fmt.Sprintf("preseed_user_%d", workerID%20))
				req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
				req.Header.Set("Authorization", "Bearer "+token)
				w := httptest.NewRecorder()
				protectedEndpoint.ServeHTTP(w, req)
				if w.Result().StatusCode != http.StatusOK {
					t.Errorf("expected 200 OK from protected endpoint, got %d", w.Result().StatusCode)
				}

			case 5:
				// Action: Access protected endpoint with tampered token (expect 401)
				token, _ := jwtMgr.GenerateToken("valid_user")
				tamperedToken := token + "corrupted_sig"
				req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
				req.Header.Set("Authorization", "Bearer "+tamperedToken)
				w := httptest.NewRecorder()
				protectedEndpoint.ServeHTTP(w, req)
				if w.Result().StatusCode != http.StatusUnauthorized {
					t.Errorf("expected 401 for tampered token, got %d", w.Result().StatusCode)
				}

			case 6:
				// Action: Access protected endpoint with expired token (expect 401)
				expiredMgr := auth.NewJWTManager("mixed-traffic-jwt-secret-key-32b", -5*time.Minute)
				expiredToken, _ := expiredMgr.GenerateToken("expired_user")
				req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
				req.Header.Set("Authorization", "Bearer "+expiredToken)
				w := httptest.NewRecorder()
				protectedEndpoint.ServeHTTP(w, req)
				if w.Result().StatusCode != http.StatusUnauthorized {
					t.Errorf("expected 401 for expired token, got %d", w.Result().StatusCode)
				}

			case 7:
				// Action: Access protected endpoint with missing/malformed auth header (expect 401)
				req := httptest.NewRequest(http.MethodGet, "/api/v1/protected", nil)
				req.Header.Set("Authorization", "InvalidScheme xyz")
				w := httptest.NewRecorder()
				protectedEndpoint.ServeHTTP(w, req)
				if w.Result().StatusCode != http.StatusUnauthorized {
					t.Errorf("expected 401 for invalid scheme, got %d", w.Result().StatusCode)
				}
			}
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
