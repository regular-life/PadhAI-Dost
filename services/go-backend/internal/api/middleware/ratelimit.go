package middleware

import (
	"context"
	"fmt"
	"net/http"
	"time"

	"github.com/redis/go-redis/v9"
)

// RateLimiter implements Redis-backed per-user rate limiting using a sliding window.
type RateLimiter struct {
	client    *redis.Client
	rps       int
	burst     int
	windowSec int
}

// NewRateLimiter creates a Redis-backed sliding-window rate limiter with specified RPS and burst limits.
func NewRateLimiter(addr, password string, db, rps, burst int) *RateLimiter {
	client := redis.NewClient(&redis.Options{
		Addr:     addr,
		Password: password,
		DB:       db,
	})
	return &RateLimiter{
		client:    client,
		rps:       rps,
		burst:     burst,
		windowSec: 60,
	}
}

// Middleware returns an HTTP middleware handler enforcing per-user rate limits.
func (rl *RateLimiter) Middleware() func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			userID := GetUserID(r.Context())
			key := fmt.Sprintf("ratelimit:%s", userID)

			allowed, err := rl.allow(r.Context(), key)
			if err != nil {
				// Fail-open on Redis errors
				next.ServeHTTP(w, r)
				return
			}
			if !allowed {
				w.Header().Set("Retry-After", "60")
				http.Error(w, `{"error":"rate limit exceeded"}`, http.StatusTooManyRequests)
				return
			}

			next.ServeHTTP(w, r)
		})
	}
}

func (rl *RateLimiter) allow(ctx context.Context, key string) (bool, error) {
	pipe := rl.client.Pipeline()
	incr := pipe.Incr(ctx, key)
	pipe.Expire(ctx, key, time.Duration(rl.windowSec)*time.Second)

	if _, err := pipe.Exec(ctx); err != nil {
		return false, err
	}

	return incr.Val() <= int64(rl.rps*rl.windowSec), nil
}
