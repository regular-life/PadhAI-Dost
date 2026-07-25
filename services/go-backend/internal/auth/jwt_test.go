package auth

import (
	"testing"
	"time"
)

func TestJWTManager_GenerateAndValidateToken(t *testing.T) {
	secret := "test-secret-key-12345"
	expiration := 1 * time.Hour

	manager := NewJWTManager(secret, expiration)

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

	manager := NewJWTManager(secret, expiration)

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
	manager1 := NewJWTManager("secret-1", 1*time.Hour)
	manager2 := NewJWTManager("secret-2", 1*time.Hour)

	token, err := manager1.GenerateToken("user-123")
	if err != nil {
		t.Fatalf("failed to generate token: %v", err)
	}

	_, err = manager2.ValidateToken(token)
	if err == nil {
		t.Fatal("expected error validating token signed with different secret, got nil")
	}
}
