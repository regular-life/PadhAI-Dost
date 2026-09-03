package auth_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

func TestMemoryUserRepository_CreateAndGet(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	ctx := context.Background()
	user, err := repo.CreateUser(ctx, "testuser", "$2a$10$fakebcryptpasswordhash")
	if err != nil {
		t.Fatalf("CreateUser failed: %v", err)
	}

	if user.Username != "testuser" {
		t.Errorf("expected username 'testuser', got %q", user.Username)
	}
	if user.PasswordHash != "$2a$10$fakebcryptpasswordhash" {
		t.Errorf("expected password hash match, got %q", user.PasswordHash)
	}
	if user.ID == "" {
		t.Error("expected non-empty user ID")
	}
	if user.CreatedAt.IsZero() {
		t.Error("expected valid CreatedAt timestamp")
	}

	// Retrieve user
	got, err := repo.GetUserByUsername(ctx, "testuser")
	if err != nil {
		t.Fatalf("GetUserByUsername failed: %v", err)
	}
	if got.ID != user.ID || got.Username != user.Username || got.PasswordHash != user.PasswordHash {
		t.Errorf("retrieved user does not match created user: got %+v, want %+v", got, user)
	}
}

func TestMemoryUserRepository_DuplicateUser(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	ctx := context.Background()
	_, err := repo.CreateUser(ctx, "duplicate_user", "hash1")
	if err != nil {
		t.Fatalf("first CreateUser failed: %v", err)
	}

	_, err = repo.CreateUser(ctx, "duplicate_user", "hash2")
	if err == nil {
		t.Fatal("expected error on duplicate CreateUser, got nil")
	}
	if !errors.Is(err, auth.ErrUserAlreadyExists) {
		t.Errorf("expected ErrUserAlreadyExists, got %v", err)
	}
}

func TestMemoryUserRepository_GetNonExistentUser(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	ctx := context.Background()
	user, err := repo.GetUserByUsername(ctx, "non_existent")
	if err == nil {
		t.Fatal("expected error for non-existent user, got nil")
	}
	if !errors.Is(err, auth.ErrUserNotFound) {
		t.Errorf("expected ErrUserNotFound, got %v", err)
	}
	if user != nil {
		t.Errorf("expected nil user pointer, got %+v", user)
	}
}

func TestMemoryUserRepository_InputValidation(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	ctx := context.Background()

	// Empty username
	_, err := repo.CreateUser(ctx, "", "hash")
	if err == nil || !errors.Is(err, auth.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for empty username, got %v", err)
	}

	// Empty password hash
	_, err = repo.CreateUser(ctx, "validuser", "")
	if err == nil || !errors.Is(err, auth.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for empty password hash, got %v", err)
	}

	// Empty username lookup
	_, err = repo.GetUserByUsername(ctx, "")
	if err == nil || !errors.Is(err, auth.ErrInvalidInput) {
		t.Errorf("expected ErrInvalidInput for empty username lookup, got %v", err)
	}
}

func TestMemoryUserRepository_ContextCancellation(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := repo.CreateUser(ctx, "cancelled_user", "hash")
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled error, got %v", err)
	}

	_, err = repo.GetUserByUsername(ctx, "cancelled_user")
	if err == nil || !errors.Is(err, context.Canceled) {
		t.Errorf("expected context.Canceled error on GetUser, got %v", err)
	}
}

func TestMemoryUserRepository_SeedDemoUser(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	ctx := context.Background()
	if err := repo.SeedDemoUser(ctx, "demo", "demo123"); err != nil {
		t.Fatalf("SeedDemoUser failed: %v", err)
	}

	// Seeding again should be idempotent
	if err := repo.SeedDemoUser(ctx, "demo", "demo123"); err != nil {
		t.Fatalf("Idempotent SeedDemoUser failed: %v", err)
	}

	user, err := repo.GetUserByUsername(ctx, "demo")
	if err != nil {
		t.Fatalf("GetUserByUsername for demo failed: %v", err)
	}
	if user.Username != "demo" {
		t.Errorf("expected username 'demo', got %q", user.Username)
	}
}

func TestMemoryUserRepository_PingAndClose(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	if err := repo.Ping(context.Background()); err != nil {
		t.Errorf("expected nil error on Ping, got %v", err)
	}
	if err := repo.Close(); err != nil {
		t.Errorf("expected nil error on Close, got %v", err)
	}
}

func TestMemoryUserRepository_ConcurrentAccess(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	const numGoroutines = 100
	var wg sync.WaitGroup
	ctx := context.Background()

	// Concurrently create unique users
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uname := fmt.Sprintf("user_%d", idx)
			_, err := repo.CreateUser(ctx, uname, fmt.Sprintf("hash_%d", idx))
			if err != nil {
				t.Errorf("concurrent CreateUser failed for %s: %v", uname, err)
			}
		}(i)
	}
	wg.Wait()

	// Concurrently read created users
	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uname := fmt.Sprintf("user_%d", idx)
			u, err := repo.GetUserByUsername(ctx, uname)
			if err != nil {
				t.Errorf("concurrent GetUserByUsername failed for %s: %v", uname, err)
				return
			}
			if u.Username != uname {
				t.Errorf("concurrent GetUserByUsername username mismatch: got %s, want %s", u.Username, uname)
			}
		}(i)
	}
	wg.Wait()
}

func TestMemoryUserRepository_ConcurrentDuplicateRegistration(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	const numAttempts = 50
	const contendedUsername = "race_contended_user"
	var (
		wg            sync.WaitGroup
		successCount  int64
		conflictCount int64
	)
	ctx := context.Background()

	for i := 0; i < numAttempts; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			_, err := repo.CreateUser(ctx, contendedUsername, fmt.Sprintf("hash_%d", idx))
			if err == nil {
				atomic.AddInt64(&successCount, 1)
			} else if errors.Is(err, auth.ErrUserAlreadyExists) {
				atomic.AddInt64(&conflictCount, 1)
			} else {
				t.Errorf("unexpected error on concurrent register: %v", err)
			}
		}(i)
	}
	wg.Wait()

	if successCount != 1 {
		t.Fatalf("expected exactly 1 successful registration under contention, got %d", successCount)
	}
	if conflictCount != numAttempts-1 {
		t.Fatalf("expected %d conflict errors, got %d", numAttempts-1, conflictCount)
	}
}
