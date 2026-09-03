package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/redis/go-redis/v9"
	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
)

// RedisCache manages L1 exact match caching backed by Redis with Circuit Breaker protection.
type RedisCache struct {
	client     *redis.Client
	ttl        time.Duration
	cb         CircuitBreaker
	mockStore  map[string]string
	mockGetErr error
	mockSetErr error
	mu         sync.RWMutex
}

// NewRedisCache constructs a new RedisCache instance with a default CircuitBreaker.
func NewRedisCache(addr, password string, db int) *RedisCache {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})
	cb := NewCircuitBreaker(DefaultConfig())
	return &RedisCache{
		client: client,
		ttl:    1 * time.Hour,
		cb:     cb,
	}
}

// NewRedisCacheWithBreaker constructs a RedisCache with custom client, breaker, and TTL.
func NewRedisCacheWithBreaker(client *redis.Client, cb CircuitBreaker, ttl time.Duration) *RedisCache {
	if ttl <= 0 {
		ttl = 1 * time.Hour
	}
	if cb == nil {
		cb = NewCircuitBreaker(DefaultConfig())
	}
	return &RedisCache{
		client: client,
		ttl:    ttl,
		cb:     cb,
	}
}

// NewMockRedisCache creates an in-memory RedisCache for deterministic testing without external services.
func NewMockRedisCache(cb CircuitBreaker) *RedisCache {
	if cb == nil {
		cb = NewCircuitBreaker(DefaultConfig())
	}
	return &RedisCache{
		ttl:       1 * time.Hour,
		cb:        cb,
		mockStore: make(map[string]string),
	}
}

// CircuitBreaker returns the active circuit breaker.
func (c *RedisCache) CircuitBreaker() CircuitBreaker {
	return c.cb
}

// SetCircuitBreaker updates the active circuit breaker.
func (c *RedisCache) SetCircuitBreaker(cb CircuitBreaker) {
	c.cb = cb
}

// SetMockGetErr injects a mock error for subsequent Get/Ping operations.
func (c *RedisCache) SetMockGetErr(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.mockGetErr = err
}

// SetMockSetErr injects a mock error for subsequent Set operations.
func (c *RedisCache) SetMockSetErr(err error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.mockSetErr = err
}

// SetMockData seeds an in-memory key/value pair for testing.
func (c *RedisCache) SetMockData(key string, val string) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.mockStore == nil {
		c.mockStore = make(map[string]string)
	}
	c.mockStore[key] = val
}

// Ping checks Redis connectivity protected by the circuit breaker.
func (c *RedisCache) Ping(ctx context.Context) error {
	return c.cb.Execute(ctx, func() error {
		if c.client == nil {
			c.mu.RLock()
			defer c.mu.RUnlock()
			if c.mockGetErr != nil {
				return c.mockGetErr
			}
			return nil
		}
		return c.client.Ping(ctx).Err()
	})
}

// CacheKey generates a deterministic SHA-256 cache key for document queries.
func CacheKey(docID, question string) string {
	hash := sha256.Sum256([]byte(question))
	return fmt.Sprintf("query:%s:%x", docID, hash[:8])
}

// Get fetches and unmarshals a cached entry, treating redis.Nil as a non-failing miss.
func (c *RedisCache) Get(ctx context.Context, key string, dest interface{}) (bool, error) {
	var val string
	var isNil bool

	err := c.cb.Execute(ctx, func() error {
		if c.client == nil {
			c.mu.RLock()
			defer c.mu.RUnlock()
			if c.mockGetErr != nil {
				return c.mockGetErr
			}
			if c.mockStore != nil {
				v, ok := c.mockStore[key]
				if !ok {
					isNil = true
					return nil
				}
				val = v
				return nil
			}
			isNil = true
			return nil
		}

		var gErr error
		val, gErr = c.client.Get(ctx, key).Result()
		if gErr == redis.Nil {
			isNil = true
			return nil // Key miss is not a Redis failure; circuit remains intact
		}
		return gErr
	})

	if errors.Is(err, ErrCircuitOpen) {
		metrics.CacheHits.WithLabelValues("circuit_open", "l1_exact").Inc()
		return false, ErrCircuitOpen
	}

	if isNil {
		metrics.CacheHits.WithLabelValues("miss", "l1_exact").Inc()
		return false, nil
	}

	if err != nil {
		metrics.CacheHits.WithLabelValues("error", "l1_exact").Inc()
		return false, fmt.Errorf("redis get failed: %w", err)
	}

	metrics.CacheHits.WithLabelValues("hit", "l1_exact").Inc()
	if err := json.Unmarshal([]byte(val), dest); err != nil {
		return false, fmt.Errorf("cache unmarshal failed: %w", err)
	}
	return true, nil
}

// Set writes an entry to Redis with configured TTL protected by the circuit breaker.
func (c *RedisCache) Set(ctx context.Context, key string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("cache marshal failed: %w", err)
	}

	err = c.cb.Execute(ctx, func() error {
		if c.client == nil {
			c.mu.Lock()
			defer c.mu.Unlock()
			if c.mockSetErr != nil {
				return c.mockSetErr
			}
			if c.mockStore != nil {
				c.mockStore[key] = string(data)
			}
			return nil
		}
		return c.client.Set(ctx, key, string(data), c.ttl).Err()
	})

	if errors.Is(err, ErrCircuitOpen) {
		log.Printf("[Cache] Circuit breaker open, skipping L1 set for key: %s", key)
		return ErrCircuitOpen
	}
	if err != nil {
		return fmt.Errorf("redis set failed: %w", err)
	}

	log.Printf("[Cache] Set key: %s", key)
	return nil
}

// Close closes the underlying Redis connection.
func (c *RedisCache) Close() error {
	if c.client != nil {
		return c.client.Close()
	}
	return nil
}
