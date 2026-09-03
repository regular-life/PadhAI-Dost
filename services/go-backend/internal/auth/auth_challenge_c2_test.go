package auth_test

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

// ─────────────────────────────────────────────────────────────────────────────
// Challenger 2: Scenario 1 - High Concurrency Memory Repository Stress Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalC2_MemoryRepo_HighConcurrencyReadWriteStress(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	const workers = 300
	var wg sync.WaitGroup
	ctx := context.Background()

	// 1. Concurrent writes (300 distinct users)
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uname := fmt.Sprintf("stress_user_%d", idx)
			u, err := repo.CreateUser(ctx, uname, fmt.Sprintf("hash_%d", idx))
			if err != nil {
				t.Errorf("failed CreateUser for %s: %v", uname, err)
				return
			}
			if u.Username != uname {
				t.Errorf("username mismatch: got %s, want %s", u.Username, uname)
			}
		}(i)
	}
	wg.Wait()

	// 2. Concurrent reads and struct mutation isolation verification
	for i := 0; i < workers; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uname := fmt.Sprintf("stress_user_%d", idx)
			u, err := repo.GetUserByUsername(ctx, uname)
			if err != nil {
				t.Errorf("failed GetUserByUsername for %s: %v", uname, err)
				return
			}
			if u.Username != uname {
				t.Errorf("username mismatch: got %s, want %s", u.Username, uname)
			}

			// Mutate returned user pointer to verify Copy-on-Read immutability
			u.Username = "corrupted_username"
			u.PasswordHash = "corrupted_hash"
		}(i)
	}
	wg.Wait()

	// 3. Verify internal state was NOT corrupted by external mutations
	for i := 0; i < workers; i++ {
		uname := fmt.Sprintf("stress_user_%d", i)
		u, err := repo.GetUserByUsername(ctx, uname)
		if err != nil {
			t.Fatalf("failed to retrieve %s after mutation check: %v", uname, err)
		}
		if u.Username != uname {
			t.Fatalf("CRITICAL: MemoryUserRepository internal map was corrupted! Got %q, want %q", u.Username, uname)
		}
		if u.PasswordHash != fmt.Sprintf("hash_%d", i) {
			t.Fatalf("CRITICAL: MemoryUserRepository internal password hash corrupted! Got %q", u.PasswordHash)
		}
	}
}

func TestEmpiricalC2_MemoryRepo_MassiveStampedeRegistration(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	const concurrentGoroutines = 200
	const targetUsername = "stampede_contended_user"
	var (
		wg            sync.WaitGroup
		successCount  int64
		conflictCount int64
		otherCount    int64
	)
	ctx := context.Background()

	for i := 0; i < concurrentGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, err := repo.CreateUser(ctx, targetUsername, fmt.Sprintf("hash_%d", idx))
			if err == nil {
				atomic.AddInt64(&successCount, 1)
			} else if errors.Is(err, auth.ErrUserAlreadyExists) {
				atomic.AddInt64(&conflictCount, 1)
			} else {
				atomic.AddInt64(&otherCount, 1)
			}
		}(i)
	}
	wg.Wait()

	if successCount != 1 {
		t.Fatalf("expected exactly 1 winner in stampede, got %d", successCount)
	}
	if conflictCount != concurrentGoroutines-1 {
		t.Fatalf("expected %d conflicts, got %d", concurrentGoroutines-1, conflictCount)
	}
	if otherCount != 0 {
		t.Fatalf("expected 0 unexpected errors, got %d", otherCount)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Challenger 2: Scenario 2 - JWT Lifecycle, Claims, Expiration & Tampering
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalC2_JWT_TokenGeneration_ValidClaimsExtraction(t *testing.T) {
	t.Parallel()
	secret := "super-secure-production-jwt-secret-key-32b"
	ttl := 30 * time.Minute
	mgr := auth.NewJWTManager(secret, ttl)

	usernames := []string{
		"standard_user",
		"user@domain.com",
		"user-with-dashes_and.dots",
		"admin_1234567890",
		"código_usuario_123",
		"00000000-0000-0000-0000-000000000001",
	}

	for _, uname := range usernames {
		uname := uname
		t.Run("User_"+uname, func(t *testing.T) {
			t.Parallel()
			now := time.Now()
			token, err := mgr.GenerateToken(uname)
			if err != nil {
				t.Fatalf("GenerateToken failed for %s: %v", uname, err)
			}

			claims, err := mgr.ValidateToken(token)
			if err != nil {
				t.Fatalf("ValidateToken failed for valid token: %v", err)
			}

			if claims.UserID != uname {
				t.Errorf("claims.UserID mismatch: got %q, want %q", claims.UserID, uname)
			}
			if claims.Issuer != "council-ai" {
				t.Errorf("claims.Issuer mismatch: got %q, want 'council-ai'", claims.Issuer)
			}

			// Validate expiration timestamp bounds
			if claims.ExpiresAt == nil {
				t.Fatal("claims.ExpiresAt is nil")
			}
			expTime := claims.ExpiresAt.Time
			expectedMinExp := now.Add(ttl - 5*time.Second)
			expectedMaxExp := now.Add(ttl + 5*time.Second)
			if expTime.Before(expectedMinExp) || expTime.After(expectedMaxExp) {
				t.Errorf("ExpiresAt %v out of expected range [%v, %v]", expTime, expectedMinExp, expectedMaxExp)
			}
		})
	}
}

func TestEmpiricalC2_JWT_ExpirationLifecycle(t *testing.T) {
	t.Parallel()
	secret := "jwt-expiration-test-secret-key"

	// 1. Negative expiration duration (expired immediately)
	mgrExpired := auth.NewJWTManager(secret, -10*time.Minute)
	tokenExpired, err := mgrExpired.GenerateToken("expired_alice")
	if err != nil {
		t.Fatalf("failed to generate expired token: %v", err)
	}

	_, err = mgrExpired.ValidateToken(tokenExpired)
	if err == nil {
		t.Fatal("expected error validating expired token, got nil")
	}

	// 2. Short 1-second TTL token (RFC 7519 second resolution) that expires after 1.1s
	mgrShort := auth.NewJWTManager(secret, 1*time.Second)
	tokenShort, err := mgrShort.GenerateToken("short_ttl_bob")
	if err != nil {
		t.Fatalf("failed to generate short ttl token: %v", err)
	}

	// Immediate validation should succeed
	claims, err := mgrShort.ValidateToken(tokenShort)
	if err != nil {
		t.Fatalf("immediate validation of short TTL token failed: %v", err)
	}
	if claims.UserID != "short_ttl_bob" {
		t.Errorf("UserID mismatch: got %s", claims.UserID)
	}

	// Wait for expiration beyond 1s boundary
	time.Sleep(1100 * time.Millisecond)

	// Validation after expiration must fail
	_, err = mgrShort.ValidateToken(tokenShort)
	if err == nil {
		t.Fatal("expected expired token error after sleep, got nil")
	}
}

func TestEmpiricalC2_JWT_TamperScenarios(t *testing.T) {
	t.Parallel()
	secret := "tamper-defense-secret-key-abcdef123"
	mgr := auth.NewJWTManager(secret, 1*time.Hour)

	validToken, err := mgr.GenerateToken("legitimate_user")
	if err != nil {
		t.Fatalf("failed to generate base valid token: %v", err)
	}

	parts := strings.Split(validToken, ".")
	if len(parts) != 3 {
		t.Fatalf("expected 3 JWT parts, got %d", len(parts))
	}
	headerB64, payloadB64, sigB64 := parts[0], parts[1], parts[2]

	t.Run("Tamper_Signature_BitFlip", func(t *testing.T) {
		t.Parallel()
		tamperedSigBytes := []byte(sigB64)
		if len(tamperedSigBytes) > 5 {
			if tamperedSigBytes[5] == 'X' {
				tamperedSigBytes[5] = 'Y'
			} else {
				tamperedSigBytes[5] = 'X'
			}
		}
		tamperedToken := fmt.Sprintf("%s.%s.%s", headerB64, payloadB64, string(tamperedSigBytes))

		_, err := mgr.ValidateToken(tamperedToken)
		if err == nil {
			t.Fatal("SECURITY FAILURE: Tampered signature was accepted by ValidateToken!")
		}
	})

	t.Run("Tamper_Payload_PrivilegeEscalation", func(t *testing.T) {
		t.Parallel()
		rawPayload, err := base64.RawURLEncoding.DecodeString(payloadB64)
		if err != nil {
			t.Fatalf("failed to decode payload base64: %v", err)
		}

		var payloadMap map[string]interface{}
		if err := json.Unmarshal(rawPayload, &payloadMap); err != nil {
			t.Fatalf("failed to unmarshal payload JSON: %v", err)
		}

		payloadMap["user_id"] = "admin_super_user"
		tamperedPayloadBytes, _ := json.Marshal(payloadMap)
		tamperedPayloadB64 := base64.RawURLEncoding.EncodeToString(tamperedPayloadBytes)

		tamperedToken := fmt.Sprintf("%s.%s.%s", headerB64, tamperedPayloadB64, sigB64)

		_, err = mgr.ValidateToken(tamperedToken)
		if err == nil {
			t.Fatal("SECURITY FAILURE: Payload privilege escalation accepted without valid signature!")
		}
	})

	t.Run("Tamper_AlgorithmConfusion_NoneAlg", func(t *testing.T) {
		t.Parallel()
		noneHeader := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
		nonePayload := base64.RawURLEncoding.EncodeToString([]byte(`{"user_id":"admin","iss":"council-ai"}`))
		noneToken := fmt.Sprintf("%s.%s.", noneHeader, nonePayload)

		_, err := mgr.ValidateToken(noneToken)
		if err == nil {
			t.Fatal("SECURITY FAILURE: alg=none token was accepted by ValidateToken!")
		}
		if !strings.Contains(err.Error(), "unexpected signing method") && !strings.Contains(err.Error(), "token is unverifiable") && !strings.Contains(err.Error(), "invalid token") {
			t.Logf("Observed none alg rejection error: %v", err)
		}
	})

	t.Run("Tamper_KeyConfusion_DifferentSecret", func(t *testing.T) {
		t.Parallel()
		attackerMgr := auth.NewJWTManager("attacker-controlled-secret-key", 1*time.Hour)
		attackerToken, err := attackerMgr.GenerateToken("impersonated_admin")
		if err != nil {
			t.Fatalf("failed to generate attacker token: %v", err)
		}

		_, err = mgr.ValidateToken(attackerToken)
		if err == nil {
			t.Fatal("SECURITY FAILURE: Token signed with foreign secret key was accepted!")
		}
	})

	t.Run("Tamper_Malformed_TruncatedAndGarbageTokens", func(t *testing.T) {
		t.Parallel()
		malformedCases := []struct {
			name  string
			token string
		}{
			{name: "Empty string", token: ""},
			{name: "Single part", token: "headeronlywithoutdot"},
			{name: "Two parts (missing signature)", token: headerB64 + "." + payloadB64},
			{name: "Four parts", token: validToken + ".extrapart"},
			{name: "Garbage ASCII", token: "not.a.valid.jwt.at.all"},
			{name: "Non-base64 characters", token: "header!.payload@.signature#"},
			{name: "Trailing garbage", token: validToken + "  "},
			{name: "Leading garbage", token: "  " + validToken},
		}

		for _, tc := range malformedCases {
			tc := tc
			t.Run(tc.name, func(t *testing.T) {
				t.Parallel()
				_, err := mgr.ValidateToken(tc.token)
				if err == nil {
					t.Fatalf("SECURITY FAILURE: Malformed token %q was accepted!", tc.token)
				}
			})
		}
	})
}

func TestEmpiricalC2_JWT_ConcurrentGenerationAndValidation(t *testing.T) {
	t.Parallel()
	mgr := auth.NewJWTManager("concurrency-stress-test-jwt-secret", 1*time.Hour)

	const totalOperations = 500
	var wg sync.WaitGroup

	for i := 0; i < totalOperations; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uname := fmt.Sprintf("concurrent_jwt_user_%d", idx)
			token, err := mgr.GenerateToken(uname)
			if err != nil {
				t.Errorf("concurrent GenerateToken error for %s: %v", uname, err)
				return
			}

			claims, err := mgr.ValidateToken(token)
			if err != nil {
				t.Errorf("concurrent ValidateToken error for %s: %v", uname, err)
				return
			}

			if claims.UserID != uname {
				t.Errorf("concurrent claims.UserID mismatch: got %s, want %s", claims.UserID, uname)
			}
		}(i)
	}
	wg.Wait()
}

// ─────────────────────────────────────────────────────────────────────────────
// Challenger 2: Scenario 3 - PostgresUserRepository Isolation & Config Tests
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalC2_PostgresRepo_ConnectionStringGeneration(t *testing.T) {
	t.Parallel()

	// 1. Direct URL takes precedence
	cfg1 := auth.PostgresConfig{
		URL: "postgres://custom:pass@customhost:9999/customdb?sslmode=require",
	}
	if cfg1.ConnectionString() != cfg1.URL {
		t.Errorf("expected direct URL, got %q", cfg1.ConnectionString())
	}

	// 2. Discrete fields with URL escaping
	cfg2 := auth.PostgresConfig{
		Host:     "db.internal",
		Port:     "5432",
		User:     "user@special#name",
		Password: "p@ss:word/with?special=chars",
		Database: "council_db",
		SSLMode:  "disable",
	}
	connStr2 := cfg2.ConnectionString()
	if !strings.Contains(connStr2, "db.internal:5432/council_db?sslmode=disable") {
		t.Errorf("connection string missing expected host/db: %q", connStr2)
	}
	if !strings.Contains(connStr2, "user%40special%23name") {
		t.Errorf("user was not URL-escaped properly in connection string: %q", connStr2)
	}

	// 3. Default fallback values
	cfg3 := auth.PostgresConfig{}
	connStr3 := cfg3.ConnectionString()
	expectedDefault := "postgres://council_user:council_pass@localhost:5432/council_db?sslmode=disable"
	if connStr3 != expectedDefault {
		t.Errorf("expected default connection string %q, got %q", expectedDefault, connStr3)
	}
}

func TestEmpiricalC2_PostgresRepo_OfflineConnectionTimeout(t *testing.T) {
	t.Parallel()

	cfg := auth.PostgresConfig{
		URL: "postgres://user:pass@127.0.0.1:59999/db?sslmode=disable&connect_timeout=1",
	}

	ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
	defer cancel()

	start := time.Now()
	repo, err := auth.NewPostgresUserRepository(ctx, cfg)
	elapsed := time.Since(start)

	if err == nil {
		_ = repo.Close()
		t.Fatal("expected connection error for closed port, got nil")
	}

	if elapsed > 1*time.Second {
		t.Errorf("connection attempt took too long: %v (expected < 1s)", elapsed)
	}
}

func TestEmpiricalC2_PostgresRepo_GracefulSkipWhenOffline(t *testing.T) {
	t.Parallel()
	start := time.Now()

	setupPostgresRepo(t)

	elapsed := time.Since(start)
	t.Logf("PostgresRepo isolation test executed in %v", elapsed)
	if elapsed > 500*time.Millisecond {
		t.Logf("Warning: Postgres skip took %v", elapsed)
	}
}
