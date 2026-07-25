package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log"
	"strconv"
	"strings"
	"time"
	"unsafe"

	"github.com/redis/go-redis/v9"
	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
)

const (
	SemanticIndexName = "idx:semantic_cache"
	SemanticKeyPrefix = "semcache:"
	VectorDim         = 384
	DefaultTTL        = 24 * time.Hour
)

// SemanticCache defines the interface for vector similarity caching.
type SemanticCache interface {
	EnsureIndex(ctx context.Context) error
	Put(ctx context.Context, docID string, vector []float32, response interface{}) error
	Get(ctx context.Context, docID string, vector []float32, threshold float32, dest interface{}) (bool, error)
	Close() error
}

// RedisSemanticCache implements SemanticCache using RediSearch VSS on Redis Stack.
type RedisSemanticCache struct {
	client *redis.Client
	ttl    time.Duration
}

// NewRedisSemanticCache constructs a new RedisSemanticCache instance.
func NewRedisSemanticCache(addr, password string, db int) *RedisSemanticCache {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})
	return &RedisSemanticCache{
		client: client,
		ttl:    DefaultTTL,
	}
}

// Float32ToBytes converts a []float32 slice to IEEE 754 little-endian binary bytes for RediSearch (Zero-Copy).
func Float32ToBytes(vec []float32) []byte {
	if len(vec) == 0 {
		return nil
	}
	return unsafe.Slice((*byte)(unsafe.Pointer(&vec[0])), len(vec)*4)
}

// SanitizeTag escapes special characters in RediSearch tag query values to prevent syntax errors/injection.
func SanitizeTag(tag string) string {
	replacer := strings.NewReplacer(
		"-", "\\-",
		".", "\\.",
		"@", "\\@",
		"{", "\\{",
		"}", "\\}",
		":", "\\:",
		"/", "\\/",
		"\\", "\\\\",
	)
	return replacer.Replace(tag)
}

// EnsureIndex creates the RediSearch index schema if it does not already exist.
func (c *RedisSemanticCache) EnsureIndex(ctx context.Context) error {
	err := c.client.Do(ctx,
		"FT.CREATE", SemanticIndexName,
		"ON", "HASH",
		"PREFIX", "1", SemanticKeyPrefix,
		"SCHEMA",
		"doc_id", "TAG",
		"response", "TEXT",
		"vector", "VECTOR", "FLAT", "6",
		"TYPE", "FLOAT32",
		"DIM", strconv.Itoa(VectorDim),
		"DISTANCE_METRIC", "COSINE",
	).Err()

	if err != nil {
		if strings.Contains(err.Error(), "Index already exists") || strings.Contains(err.Error(), "BUSY") {
			log.Printf("[RedisSemanticCache] Index %s already exists", SemanticIndexName)
			return nil
		}
		return fmt.Errorf("failed to create RediSearch index: %w", err)
	}

	log.Printf("[RedisSemanticCache] Successfully created index %s", SemanticIndexName)
	return nil
}

// Put stores a response and vector embedding in Redis Stack HASH with a 24h TTL.
func (c *RedisSemanticCache) Put(ctx context.Context, docID string, vector []float32, response interface{}) error {
	if len(vector) != VectorDim {
		return fmt.Errorf("vector dimension mismatch: expected %d, got %d", VectorDim, len(vector))
	}

	data, err := json.Marshal(response)
	if err != nil {
		return fmt.Errorf("failed to marshal cache response: %w", err)
	}

	vecBytes := Float32ToBytes(vector)
	hash := sha256.Sum256(data)
	key := fmt.Sprintf("%s%s:%x", SemanticKeyPrefix, docID, hash[:8])

	pipe := c.client.Pipeline()
	pipe.HSet(ctx, key, map[string]interface{}{
		"doc_id":   docID,
		"response": string(data),
		"vector":   vecBytes,
	})
	pipe.Expire(ctx, key, c.ttl)

	if _, err := pipe.Exec(ctx); err != nil {
		return fmt.Errorf("failed to store semantic cache entry: %w", err)
	}

	log.Printf("[RedisSemanticCache] Put key=%s doc_id=%s dim=%d", key, docID, len(vector))
	return nil
}

// Get performs a KNN vector search via RediSearch FT.SEARCH.
func (c *RedisSemanticCache) Get(ctx context.Context, docID string, vector []float32, threshold float32, dest interface{}) (bool, error) {
	if len(vector) != VectorDim {
		metrics.CacheHits.WithLabelValues("miss", "l1").Inc()
		return false, nil
	}

	vecBytes := Float32ToBytes(vector)
	safeDocID := SanitizeTag(docID)
	queryStr := fmt.Sprintf("(@doc_id:{%s})=>[KNN 1 @vector $vec AS score]", safeDocID)
	maxDistance := float64(1.0 - threshold)

	res, err := c.client.Do(ctx,
		"FT.SEARCH", SemanticIndexName, queryStr,
		"PARAMS", "2", "vec", vecBytes,
		"SORTBY", "score", "ASC",
		"RETURN", "2", "response", "score",
		"DIALECT", "2",
	).Result()

	if err != nil {
		log.Printf("[RedisSemanticCache] FT.SEARCH query failed: %v", err)
		metrics.CacheHits.WithLabelValues("miss", "l1").Inc()
		return false, nil
	}

	responseJSON, score, ok := parseSearchResult(res)
	if !ok || responseJSON == "" || score > maxDistance {
		if ok && score > maxDistance {
			log.Printf("[RedisSemanticCache] Miss: score %.4f > maxDistance %.4f", score, maxDistance)
		}
		metrics.CacheHits.WithLabelValues("miss", "l1").Inc()
		return false, nil
	}

	if err := json.Unmarshal([]byte(responseJSON), dest); err != nil {
		log.Printf("[RedisSemanticCache] Failed to unmarshal response: %v", err)
		metrics.CacheHits.WithLabelValues("miss", "l1").Inc()
		return false, nil
	}

	metrics.CacheHits.WithLabelValues("hit", "l1").Inc()
	log.Printf("[RedisSemanticCache] Hit: doc_id=%s, score=%.4f (sim=%.4f)", docID, score, 1.0-score)
	return true, nil
}

// parseSearchResult extracts the response JSON string and distance score from an FT.SEARCH result slice.
func parseSearchResult(res interface{}) (string, float64, bool) {
	resSlice, ok := res.([]interface{})
	if !ok || len(resSlice) < 3 {
		return "", 0, false
	}

	totalHits, _ := resSlice[0].(int64)
	if totalHits == 0 {
		return "", 0, false
	}

	docFields, ok := resSlice[2].([]interface{})
	if !ok {
		return "", 0, false
	}

	var responseJSON string
	var score float64

	for i := 0; i < len(docFields)-1; i += 2 {
		fieldName, _ := docFields[i].(string)
		fieldVal, _ := docFields[i+1].(string)

		switch fieldName {
		case "response":
			responseJSON = fieldVal
		case "score":
			score, _ = strconv.ParseFloat(fieldVal, 64)
		}
	}

	return responseJSON, score, true
}

// Close shuts down the Redis client.
func (c *RedisSemanticCache) Close() error {
	return c.client.Close()
}
