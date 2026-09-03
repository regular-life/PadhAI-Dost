package auth_test

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

func TestJWTManager_GenerateAndValidateToken(t *testing.T) {
	secret := "test-secret-key-12345"
	expiration := 1 * time.Hour

	manager := auth.NewJWTManager(secret, expiration)

	userID := "user-test-789"
	token, err := manager.GenerateToken(userID)
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	if token == "" {
		t.Fatal("expected non-empty token string")
	}

	claims, err := manager.ValidateToken(token)
	if err != nil {
		t.Fatalf("failed to validate valid token: %v", err)
	}

	if claims.UserID != userID {
		t.Errorf("expected UserID %q, got %q", userID, claims.UserID)
	}

	if claims.Issuer != "council-ai" {
		t.Errorf("expected Issuer 'council-ai', got %q", claims.Issuer)
	}
}

func TestJWTManager_ExpiredToken(t *testing.T) {
	secret := "test-secret-key-12345"
	expiration := -1 * time.Second // Expired immediately

	manager := auth.NewJWTManager(secret, expiration)

	token, err := manager.GenerateToken("user-expired")
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	_, err = manager.ValidateToken(token)
	if err == nil {
		t.Fatal("expected error when validating expired token, got nil")
	}
}

func TestJWTManager_InvalidSecret(t *testing.T) {
	manager1 := auth.NewJWTManager("secret-1", 1*time.Hour)
	manager2 := auth.NewJWTManager("secret-2", 1*time.Hour)

	token, err := manager1.GenerateToken("user-123")
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	_, err = manager2.ValidateToken(token)
	if err == nil {
		t.Fatal("expected error validating token signed with different secret, got nil")
	}
}

// ── Adversarial & Concurrency Tests ──────────────────────────

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

func verifyTamperSignatureBitFlip(t *testing.T, mgr *auth.JWTManager, headerB64, payloadB64, sigB64 string) {
	tamperedSigBytes := []byte(sigB64)
	if len(tamperedSigBytes) > 5 {
		if tamperedSigBytes[5] == 'X' {
			tamperedSigBytes[5] = 'Y'
		} else {
			tamperedSigBytes[5] = 'X'
		}
	}
	tamperedToken := fmt.Sprintf("%s.%s.%s", headerB64, payloadB64, string(tamperedSigBytes))
	if _, err := mgr.ValidateToken(tamperedToken); err == nil {
		t.Fatal("SECURITY FAILURE: Tampered signature was accepted by ValidateToken!")
	}
}

func verifyTamperPayloadPrivilegeEscalation(t *testing.T, mgr *auth.JWTManager, headerB64, payloadB64, sigB64 string) {
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

	if _, err = mgr.ValidateToken(tamperedToken); err == nil {
		t.Fatal("SECURITY FAILURE: Payload privilege escalation accepted without valid signature!")
	}
}

func verifyTamperAlgorithmConfusion(t *testing.T, mgr *auth.JWTManager) {
	noneHeader := base64.RawURLEncoding.EncodeToString([]byte(`{"alg":"none","typ":"JWT"}`))
	nonePayload := base64.RawURLEncoding.EncodeToString([]byte(`{"user_id":"admin","iss":"council-ai"}`))
	noneToken := fmt.Sprintf("%s.%s.", noneHeader, nonePayload)

	if _, err := mgr.ValidateToken(noneToken); err == nil {
		t.Fatal("SECURITY FAILURE: alg=none token was accepted by ValidateToken!")
	}
}

func verifyTamperForeignSecret(t *testing.T, mgr *auth.JWTManager) {
	attackerMgr := auth.NewJWTManager("attacker-controlled-secret-key", 1*time.Hour)
	attackerToken, err := attackerMgr.GenerateToken("impersonated_admin")
	if err != nil {
		t.Fatalf("failed to generate attacker token: %v", err)
	}
	if _, err = mgr.ValidateToken(attackerToken); err == nil {
		t.Fatal("SECURITY FAILURE: Token signed with foreign secret key was accepted!")
	}
}

func verifyMalformedTokens(t *testing.T, mgr *auth.JWTManager, headerB64, payloadB64, validToken string) {
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
		t.Run(tc.name, func(t *testing.T) {
			if _, err := mgr.ValidateToken(tc.token); err == nil {
				t.Fatalf("SECURITY FAILURE: Malformed token %q was accepted!", tc.token)
			}
		})
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
		verifyTamperSignatureBitFlip(t, mgr, headerB64, payloadB64, sigB64)
	})

	t.Run("Tamper_Payload_PrivilegeEscalation", func(t *testing.T) {
		t.Parallel()
		verifyTamperPayloadPrivilegeEscalation(t, mgr, headerB64, payloadB64, sigB64)
	})

	t.Run("Tamper_AlgorithmConfusion_NoneAlg", func(t *testing.T) {
		t.Parallel()
		verifyTamperAlgorithmConfusion(t, mgr)
	})

	t.Run("Tamper_KeyConfusion_DifferentSecret", func(t *testing.T) {
		t.Parallel()
		verifyTamperForeignSecret(t, mgr)
	})

	t.Run("Tamper_Malformed_TruncatedAndGarbageTokens", func(t *testing.T) {
		t.Parallel()
		verifyMalformedTokens(t, mgr, headerB64, payloadB64, validToken)
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
