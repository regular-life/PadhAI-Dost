package auth_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

func getTestDatabaseURL() string {
	if url := os.Getenv("TEST_DATABASE_URL"); url != "" {
		return url
	}
	return os.Getenv("DATABASE_URL")
}

func setupPostgresRepo(t *testing.T) (*auth.PostgresUserRepository, context.Context) {
	t.Helper()
	dbURL := getTestDatabaseURL()
	if dbURL == "" {
		t.Skip("Skipping Postgres integration test; TEST_DATABASE_URL / DATABASE_URL not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	t.Cleanup(cancel)

	repo, err := auth.NewPostgresUserRepository(ctx, auth.PostgresConfig{
		URL: dbURL,
	})
	if err != nil {
		t.Skipf("Skipping Postgres integration test; failed to connect to database: %v", err)
	}

	if err := repo.Ping(ctx); err != nil {
		_ = repo.Close()
		t.Skipf("Skipping Postgres integration test; database ping failed: %v", err)
	}

	if err := repo.InitSchema(ctx); err != nil {
		_ = repo.Close()
		t.Fatalf("failed to initialize schema in PostgreSQL: %v", err)
	}

	t.Cleanup(func() {
		_ = repo.Close()
	})

	return repo, ctx
}

func TestPostgresUserRepository_CreateAndGet(t *testing.T) {
	repo, ctx := setupPostgresRepo(t)

	username := fmt.Sprintf("pg_user_%d", time.Now().UnixNano())
	passwordHash := "$2a$10$samplepgbcryptpasswordhashvalue"

	user, err := repo.CreateUser(ctx, username, passwordHash)
	if err != nil {
		t.Fatalf("CreateUser failed in PostgreSQL: %v", err)
	}

	if user.Username != username {
		t.Errorf("expected username %q, got %q", username, user.Username)
	}
	if user.PasswordHash != passwordHash {
		t.Errorf("expected password hash %q, got %q", passwordHash, user.PasswordHash)
	}
	if user.ID == "" {
		t.Error("expected non-empty UUID user ID from PostgreSQL")
	}
	if user.CreatedAt.IsZero() {
		t.Error("expected non-zero CreatedAt timestamp")
	}

	// Lookup user
	got, err := repo.GetUserByUsername(ctx, username)
	if err != nil {
		t.Fatalf("GetUserByUsername failed: %v", err)
	}
	if got.ID != user.ID || got.Username != user.Username || got.PasswordHash != user.PasswordHash {
		t.Errorf("retrieved user mismatch: got %+v, want %+v", got, user)
	}
}

func TestPostgresUserRepository_UniqueConstraint_ErrUserAlreadyExists(t *testing.T) {
	repo, ctx := setupPostgresRepo(t)

	username := fmt.Sprintf("pg_unique_%d", time.Now().UnixNano())
	_, err := repo.CreateUser(ctx, username, "hash1")
	if err != nil {
		t.Fatalf("initial CreateUser failed: %v", err)
	}

	// Second insert with exact same username
	_, err = repo.CreateUser(ctx, username, "hash2")
	if err == nil {
		t.Fatal("expected error on duplicate username in PostgreSQL, got nil")
	}
	if !errors.Is(err, auth.ErrUserAlreadyExists) {
		t.Errorf("expected ErrUserAlreadyExists error code mapping (23505), got %v", err)
	}
}

func TestPostgresUserRepository_NotFound_ErrUserNotFound(t *testing.T) {
	repo, ctx := setupPostgresRepo(t)

	missingUsername := fmt.Sprintf("pg_missing_%d", time.Now().UnixNano())
	user, err := repo.GetUserByUsername(ctx, missingUsername)
	if err == nil {
		t.Fatal("expected error for non-existent user lookup, got nil")
	}
	if !errors.Is(err, auth.ErrUserNotFound) {
		t.Errorf("expected ErrUserNotFound, got %v", err)
	}
	if user != nil {
		t.Errorf("expected nil user, got %+v", user)
	}
}

func TestPostgresUserRepository_SeedDemoUser(t *testing.T) {
	repo, ctx := setupPostgresRepo(t)

	demoUser := fmt.Sprintf("demo_%d", time.Now().UnixNano())
	if err := repo.SeedDemoUser(ctx, demoUser, "demo_password"); err != nil {
		t.Fatalf("SeedDemoUser failed: %v", err)
	}

	// Seed again (idempotent)
	if err := repo.SeedDemoUser(ctx, demoUser, "demo_password"); err != nil {
		t.Fatalf("SeedDemoUser idempotent call failed: %v", err)
	}
}

func TestPostgresUserRepository_PoolAndClose(t *testing.T) {
	dbURL := getTestDatabaseURL()
	if dbURL == "" {
		t.Skip("Skipping Postgres integration test; DATABASE_URL not set")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()

	repo, err := auth.NewPostgresUserRepository(ctx, auth.PostgresConfig{
		URL: dbURL,
	})
	if err != nil {
		t.Skipf("Skipping Postgres integration test; failed to connect: %v", err)
	}

	if err := repo.Ping(ctx); err != nil {
		_ = repo.Close()
		t.Skipf("Skipping Postgres integration test; ping failed: %v", err)
	}

	// Close pool
	if err := repo.Close(); err != nil {
		t.Fatalf("failed to close repo pool: %v", err)
	}

	// Subsequent operations on closed pool must fail
	if err := repo.Ping(ctx); err == nil {
		t.Error("expected ping to fail on closed repository, got nil")
	}
}
