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

// ── Concurrency & Stress Tests ──────────────────────────────

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

func TestEmpiricalChallenge_Repository_EdgeCasesAndInvariants(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	ctx := context.Background()

	// 1. Unicode & Special Character Handling in Usernames and Hashes
	specialCases := []struct {
		name     string
		username string
		hash     string
	}{
		{name: "Emoji Username", username: "user_🚀_council", hash: "$2a$10$emojihash123456789012345678"},
		{name: "CJK Characters", username: "用户_999", hash: "$2a$10$cjkhash12345678901234567890"},
		{name: "SQL Injection Probe Username", username: "admin' OR 1=1; --", hash: "$2a$10$sqlprobehash12345678901"},
		{name: "Whitespace in middle of username", username: "user with spaces", hash: "$2a$10$spaceshash12345678901234"},
		{name: "Symbols and quotes", username: "user<script>alert(1)</script>", hash: "$2a$10$xsshash1234567890123456"},
	}

	for _, sc := range specialCases {
		sc := sc
		t.Run(sc.name, func(t *testing.T) {
			user, err := repo.CreateUser(ctx, sc.username, sc.hash)
			if err != nil {
				t.Fatalf("failed to create user with %s: %v", sc.name, err)
			}
			if user.Username != sc.username {
				t.Errorf("username mismatch: got %q, want %q", user.Username, sc.username)
			}

			// Lookup
			fetched, err := repo.GetUserByUsername(ctx, sc.username)
			if err != nil {
				t.Fatalf("failed to get user with %s: %v", sc.name, err)
			}
			if fetched.Username != sc.username || fetched.PasswordHash != sc.hash {
				t.Errorf("retrieved user mismatch: got %+v, want %+v", fetched, user)
			}
		})
	}

	// 2. Immutability of returned User copy
	created, err := repo.CreateUser(ctx, "immutable_check", "$2a$10$originalhash123456789012")
	if err != nil {
		t.Fatalf("CreateUser failed: %v", err)
	}

	// Mutate local struct
	created.Username = "mutated_locally"
	created.PasswordHash = "mutated_hash"

	// Fetch fresh copy and verify unchanged
	fresh, err := repo.GetUserByUsername(ctx, "immutable_check")
	if err != nil {
		t.Fatalf("failed to get user: %v", err)
	}
	if fresh.Username != "immutable_check" || fresh.PasswordHash != "$2a$10$originalhash123456789012" {
		t.Errorf("internal repository state was mutated through external struct pointer: %+v", fresh)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Scenario: High-Volume Concurrent Mixed Read/Write Stress Test
// ─────────────────────────────────────────────────────────────────────────────

func TestEmpiricalChallenge_Repository_HighConcurrencyMixedStress(t *testing.T) {
	t.Parallel()
	repo := auth.NewMemoryUserRepository()
	defer repo.Close()

	const numWriters = 50
	const numReaders = 50
	const iterations = 20

	var (
		wg           sync.WaitGroup
		writeErrors  int64
		readMisses   int64
		conflictHits int64
	)
	ctx := context.Background()

	// Writers concurrently create users
	for w := 0; w < numWriters; w++ {
		wg.Add(1)
		go func(writerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				uname := fmt.Sprintf("stress_user_%d_%d", writerID, i)
				_, err := repo.CreateUser(ctx, uname, fmt.Sprintf("$2a$10$hash_%d_%d", writerID, i))
				if err != nil {
					if errors.Is(err, auth.ErrUserAlreadyExists) {
						atomic.AddInt64(&conflictHits, 1)
					} else {
						atomic.AddInt64(&writeErrors, 1)
					}
				}
			}
		}(w)
	}

	// Readers concurrently attempt to read users
	for r := 0; r < numReaders; r++ {
		wg.Add(1)
		go func(readerID int) {
			defer wg.Done()
			for i := 0; i < iterations; i++ {
				uname := fmt.Sprintf("stress_user_%d_%d", readerID, i)
				_, err := repo.GetUserByUsername(ctx, uname)
				if err != nil && errors.Is(err, auth.ErrUserNotFound) {
					atomic.AddInt64(&readMisses, 1)
				}
			}
		}(r)
	}

	wg.Wait()

	if writeErrors != 0 {
		t.Errorf("encountered %d unexpected write errors during mixed stress test", writeErrors)
	}
	t.Logf("Stress test complete: %d writes, read misses during async ramp-up: %d, conflicts: %d",
		numWriters*iterations, readMisses, conflictHits)
}
