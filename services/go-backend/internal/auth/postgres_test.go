package auth_test

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"
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

// ── Connection & Offline Resilience Tests ────────────────────

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
