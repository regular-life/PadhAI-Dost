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

// ─────────────────────────────────────────────────────────────────────────────
// Scenario: Memory & Postgres Repository Empirical Invariants & Edge Cases
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
