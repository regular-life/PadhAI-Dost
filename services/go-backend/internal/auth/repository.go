package auth

import (
	"context"
	"errors"
	"time"
)

// Domain error definitions.
var (
	// ErrUserNotFound is returned when looking up a non-existent user.
	ErrUserNotFound = errors.New("user not found")

	// ErrUserAlreadyExists is returned when attempting to register an existing username.
	ErrUserAlreadyExists = errors.New("user already exists")

	// ErrInvalidInput is returned when supplied credentials or parameters are empty or malformed.
	ErrInvalidInput = errors.New("invalid user input")
)

// User represents a persistent user entity in the CouncilAI system.
type User struct {
	ID           string    `json:"id"`
	Username     string    `json:"username"`
	PasswordHash string    `json:"-"`
	CreatedAt    time.Time `json:"created_at"`
}

// UserRepository defines the persistence and lookup contract for user identities.
type UserRepository interface {
	// CreateUser persists a new user with the given username and pre-hashed password.
	CreateUser(ctx context.Context, username, passwordHash string) (*User, error)

	// GetUserByUsername retrieves a user entity by unique username.
	GetUserByUsername(ctx context.Context, username string) (*User, error)

	// Ping checks the health and responsiveness of the underlying storage backend.
	Ping(ctx context.Context) error

	// Close terminates any active connections or pools held by the repository.
	Close() error
}
