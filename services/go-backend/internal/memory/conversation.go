// Package memory manages multi-turn conversation session history backed by Redis.
package memory

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"time"

	"github.com/redis/go-redis/v9"
)

// Turn represents a single message in a conversation.
type Turn struct {
	Role      string    `json:"role"` // "user" | "assistant"
	Content   string    `json:"content"`
	Timestamp time.Time `json:"timestamp"`
}

// ConversationStore manages multi-turn conversation history in Redis.
type ConversationStore struct {
	client   *redis.Client
	maxTurns int
	ttl      time.Duration
}

// NewConversationStore creates a new conversation store backed by Redis.
func NewConversationStore(addr, password string, db int, maxTurns int, ttl time.Duration) *ConversationStore {
	if maxTurns <= 0 {
		maxTurns = 10
	}
	if ttl <= 0 {
		ttl = 24 * time.Hour
	}

	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})

	return &ConversationStore{
		client:   client,
		maxTurns: maxTurns,
		ttl:      ttl,
	}
}

// key generates the Redis key for a conversation session.
func key(userID, sessionID string) string {
	return fmt.Sprintf("conv:%s:%s", userID, sessionID)
}

// Append adds a turn to the conversation history.
// Automatically trims to maxTurns and refreshes TTL.
func (s *ConversationStore) Append(ctx context.Context, userID, sessionID string, turn Turn) error {
	if turn.Timestamp.IsZero() {
		turn.Timestamp = time.Now()
	}

	data, err := json.Marshal(turn)
	if err != nil {
		return fmt.Errorf("failed to marshal turn: %w", err)
	}

	k := key(userID, sessionID)

	// Push to the end of the list
	if err := s.client.RPush(ctx, k, string(data)).Err(); err != nil {
		return fmt.Errorf("failed to append turn: %w", err)
	}

	// Trim to keep only the last maxTurns entries
	// We store pairs (user + assistant), so keep maxTurns * 2 entries
	maxEntries := int64(s.maxTurns * 2)
	if err := s.client.LTrim(ctx, k, -maxEntries, -1).Err(); err != nil {
		log.Printf("[Memory] Failed to trim conversation: %v", err)
	}

	// Refresh TTL
	s.client.Expire(ctx, k, s.ttl)

	return nil
}

// GetHistory retrieves the conversation history for a session.
// Returns up to limit turns (0 = all available).
func (s *ConversationStore) GetHistory(ctx context.Context, userID, sessionID string, limit int) ([]Turn, error) {
	k := key(userID, sessionID)

	var start int64 = 0
	if limit > 0 {
		// Get the last `limit * 2` entries (user + assistant pairs)
		total, err := s.client.LLen(ctx, k).Result()
		if err != nil {
			return nil, fmt.Errorf("failed to get list length: %w", err)
		}
		maxEntries := int64(limit * 2)
		if total > maxEntries {
			start = total - maxEntries
		}
	}

	results, err := s.client.LRange(ctx, k, start, -1).Result()
	if err != nil {
		return nil, fmt.Errorf("failed to get conversation history: %w", err)
	}

	turns := make([]Turn, 0, len(results))
	for _, raw := range results {
		var turn Turn
		if err := json.Unmarshal([]byte(raw), &turn); err != nil {
			log.Printf("[Memory] Failed to unmarshal turn, skipping: %v", err)
			continue
		}
		turns = append(turns, turn)
	}

	return turns, nil
}

// Clear deletes all conversation history for a session.
func (s *ConversationStore) Clear(ctx context.Context, userID, sessionID string) error {
	k := key(userID, sessionID)
	return s.client.Del(ctx, k).Err()
}

// Close closes the Redis connection.
func (s *ConversationStore) Close() error {
	return s.client.Close()
}
