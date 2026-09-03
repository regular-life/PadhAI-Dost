package auth

import (
	"context"
	"errors"
	"fmt"
	"net/url"
	"time"

	"github.com/jackc/pgx/v5"
	"github.com/jackc/pgx/v5/pgconn"
	"github.com/jackc/pgx/v5/pgxpool"
	"golang.org/x/crypto/bcrypt"
)

// PostgresConfig encapsulates connection and pool tuning parameters.
type PostgresConfig struct {
	URL               string        // Direct connection URI, e.g. "postgres://user:pass@host:5432/dbname?sslmode=disable"
	Host              string        // default: "localhost"
	Port              string        // default: "5432"
	User              string        // default: "council_user"
	Password          string        // default: "council_pass"
	Database          string        // default: "council_db"
	SSLMode           string        // default: "disable"
	MaxConns          int32         // default: 25
	MinConns          int32         // default: 5
	MaxConnIdleTime   time.Duration // default: 30m
	MaxConnLifetime   time.Duration // default: 1h
	HealthCheckPeriod time.Duration // default: 1m
}

// ConnectionString constructs a valid PostgreSQL connection URI from configuration.
func (c PostgresConfig) ConnectionString() string {
	if c.URL != "" {
		return c.URL
	}
	host := c.Host
	if host == "" {
		host = "localhost"
	}
	port := c.Port
	if port == "" {
		port = "5432"
	}
	user := c.User
	if user == "" {
		user = "council_user"
	}
	password := c.Password
	if password == "" {
		password = "council_pass"
	}
	database := c.Database
	if database == "" {
		database = "council_db"
	}
	sslMode := c.SSLMode
	if sslMode == "" {
		sslMode = "disable"
	}
	return fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=%s",
		url.QueryEscape(user),
		url.QueryEscape(password),
		host,
		port,
		database,
		sslMode,
	)
}

// PostgresUserRepository is a production-grade PostgreSQL implementation of UserRepository.
type PostgresUserRepository struct {
	pool *pgxpool.Pool
}

const usersTableSchema = `
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
`

// NewPostgresUserRepository initializes a pgx connection pool with configuration,
// verifies connection health via Ping, and returns a ready PostgresUserRepository.
func NewPostgresUserRepository(ctx context.Context, cfg PostgresConfig) (*PostgresUserRepository, error) {
	connStr := cfg.ConnectionString()
	poolConfig, err := pgxpool.ParseConfig(connStr)
	if err != nil {
		return nil, fmt.Errorf("failed to parse postgres configuration: %w", err)
	}

	// Apply connection pool sizing and lifecycle policies
	if cfg.MaxConns > 0 {
		poolConfig.MaxConns = cfg.MaxConns
	} else {
		poolConfig.MaxConns = 25
	}

	if cfg.MinConns > 0 {
		poolConfig.MinConns = cfg.MinConns
	} else {
		poolConfig.MinConns = 5
	}

	if cfg.MaxConnIdleTime > 0 {
		poolConfig.MaxConnIdleTime = cfg.MaxConnIdleTime
	} else {
		poolConfig.MaxConnIdleTime = 30 * time.Minute
	}

	if cfg.MaxConnLifetime > 0 {
		poolConfig.MaxConnLifetime = cfg.MaxConnLifetime
	} else {
		poolConfig.MaxConnLifetime = 1 * time.Hour
	}

	if cfg.HealthCheckPeriod > 0 {
		poolConfig.HealthCheckPeriod = cfg.HealthCheckPeriod
	} else {
		poolConfig.HealthCheckPeriod = 1 * time.Minute
	}

	pool, err := pgxpool.NewWithConfig(ctx, poolConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to initialize pgx connection pool: %w", err)
	}

	// Verify immediate connection health
	if err := pool.Ping(ctx); err != nil {
		pool.Close()
		return nil, fmt.Errorf("failed to ping postgres database: %w", err)
	}

	return &PostgresUserRepository{pool: pool}, nil
}

// InitSchema creates the users table and indexes idempotently.
func (p *PostgresUserRepository) InitSchema(ctx context.Context) error {
	if _, err := p.pool.Exec(ctx, usersTableSchema); err != nil {
		return fmt.Errorf("failed to execute schema migration: %w", err)
	}
	return nil
}

// SeedDemoUser guarantees demo credentials exist in the database.
func (p *PostgresUserRepository) SeedDemoUser(ctx context.Context, username, password string) error {
	_, err := p.GetUserByUsername(ctx, username)
	if err == nil {
		return nil // Demo user already exists
	}
	if !errors.Is(err, ErrUserNotFound) {
		return fmt.Errorf("error checking demo user existence: %w", err)
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return fmt.Errorf("failed to generate demo password hash: %w", err)
	}

	_, err = p.CreateUser(ctx, username, string(hash))
	if err != nil && !errors.Is(err, ErrUserAlreadyExists) {
		return fmt.Errorf("failed to seed demo user: %w", err)
	}
	return nil
}

// CreateUser persists a new user entity to PostgreSQL.
func (p *PostgresUserRepository) CreateUser(ctx context.Context, username, passwordHash string) (*User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if username == "" || passwordHash == "" {
		return nil, ErrInvalidInput
	}

	query := `
		INSERT INTO users (username, password_hash)
		VALUES ($1, $2)
		RETURNING id::text, username, password_hash, created_at;
	`

	var u User
	err := p.pool.QueryRow(ctx, query, username, passwordHash).Scan(
		&u.ID,
		&u.Username,
		&u.PasswordHash,
		&u.CreatedAt,
	)
	if err != nil {
		var pgErr *pgconn.PgError
		if errors.As(err, &pgErr) && pgErr.Code == "23505" { // unique_violation
			return nil, ErrUserAlreadyExists
		}
		return nil, fmt.Errorf("failed to insert user into database: %w", err)
	}

	return &u, nil
}

// GetUserByUsername retrieves a user entity by username from PostgreSQL.
func (p *PostgresUserRepository) GetUserByUsername(ctx context.Context, username string) (*User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if username == "" {
		return nil, ErrInvalidInput
	}

	query := `
		SELECT id::text, username, password_hash, created_at
		FROM users
		WHERE username = $1;
	`

	var u User
	err := p.pool.QueryRow(ctx, query, username).Scan(
		&u.ID,
		&u.Username,
		&u.PasswordHash,
		&u.CreatedAt,
	)
	if err != nil {
		if errors.Is(err, pgx.ErrNoRows) {
			return nil, ErrUserNotFound
		}
		return nil, fmt.Errorf("failed to query user from database: %w", err)
	}

	return &u, nil
}

// Ping checks whether the PostgreSQL connection pool is alive and healthy.
func (p *PostgresUserRepository) Ping(ctx context.Context) error {
	return p.pool.Ping(ctx)
}

// Close gracefully closes the pgx connection pool.
func (p *PostgresUserRepository) Close() error {
	p.pool.Close()
	return nil
}
