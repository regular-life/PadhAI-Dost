package auth

import (
	"context"
	"crypto/rand"
	"errors"
	"fmt"
	"sync"
	"time"

	"golang.org/x/crypto/bcrypt"
)

// MemoryUserRepository is an in-memory, thread-safe implementation of UserRepository
// designed for fast unit testing and zero-dependency local fallback execution.
type MemoryUserRepository struct {
	mu    sync.RWMutex
	users map[string]*User // key: username
}

// NewMemoryUserRepository constructs an empty MemoryUserRepository.
func NewMemoryUserRepository() *MemoryUserRepository {
	return &MemoryUserRepository{
		users: make(map[string]*User),
	}
}

// generateUUID creates an RFC 4122 v4 compliant UUID string without external dependencies.
func generateUUID() string {
	var uuid [16]byte
	_, _ = rand.Read(uuid[:])
	uuid[6] = (uuid[6] & 0x0f) | 0x40 // Version 4
	uuid[8] = (uuid[8] & 0x3f) | 0x80 // Variant RFC 4122
	return fmt.Sprintf("%08x-%04x-%04x-%04x-%012x",
		uuid[0:4], uuid[4:6], uuid[6:8], uuid[8:10], uuid[10:16])
}

// CreateUser persists a new user record in the in-memory map.
func (m *MemoryUserRepository) CreateUser(ctx context.Context, username, passwordHash string) (*User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if username == "" || passwordHash == "" {
		return nil, ErrInvalidInput
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if _, exists := m.users[username]; exists {
		return nil, ErrUserAlreadyExists
	}

	user := &User{
		ID:           generateUUID(),
		Username:     username,
		PasswordHash: passwordHash,
		CreatedAt:    time.Now().UTC(),
	}

	m.users[username] = user

	userCopy := *user
	return &userCopy, nil
}

// GetUserByUsername retrieves a user entity by username from the in-memory map.
func (m *MemoryUserRepository) GetUserByUsername(ctx context.Context, username string) (*User, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if username == "" {
		return nil, ErrInvalidInput
	}

	m.mu.RLock()
	defer m.mu.RUnlock()

	user, exists := m.users[username]
	if !exists {
		return nil, ErrUserNotFound
	}

	userCopy := *user
	return &userCopy, nil
}

// SeedDemoUser seeds a default user into the in-memory repository if not already existing.
func (m *MemoryUserRepository) SeedDemoUser(ctx context.Context, username, password string) error {
	hash, err := bcrypt.GenerateFromPassword([]byte(password), bcrypt.DefaultCost)
	if err != nil {
		return fmt.Errorf("failed to generate demo password hash: %w", err)
	}

	_, err = m.CreateUser(ctx, username, string(hash))
	if err != nil && !errors.Is(err, ErrUserAlreadyExists) {
		return err
	}
	return nil
}

// Ping verifies context liveness for in-memory repository.
func (m *MemoryUserRepository) Ping(ctx context.Context) error {
	return ctx.Err()
}

// Close is a no-op for the in-memory repository.
func (m *MemoryUserRepository) Close() error {
	return nil
}
