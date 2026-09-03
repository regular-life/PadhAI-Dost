package cache

import (
	"context"
	"errors"
	"fmt"
	"log"
	"sync"
	"time"
)

// State represents the operational state of the circuit breaker.
type State int

const (
	StateClosed State = iota
	StateHalfOpen
	StateOpen
)

func (s State) String() string {
	switch s {
	case StateClosed:
		return "CLOSED"
	case StateHalfOpen:
		return "HALF_OPEN"
	case StateOpen:
		return "OPEN"
	default:
		return "UNKNOWN"
	}
}

var (
	// ErrCircuitOpen is returned immediately when the circuit breaker is in open state.
	ErrCircuitOpen = errors.New("circuit breaker is open: cache operation rejected")
	// ErrTooManyRequests is returned when half-open max concurrent probes are exceeded.
	ErrTooManyRequests = errors.New("circuit breaker: too many requests in half-open state")
	// ErrHalfOpenRateLimit is an alias for ErrTooManyRequests.
	ErrHalfOpenRateLimit = ErrTooManyRequests
)

// CircuitBreaker defines the public interface for the circuit breaker.
type CircuitBreaker interface {
	Execute(ctx context.Context, op func() error) error
	State() State
	Reset()
	Trip()
	Counts() (consecutiveSuccesses, consecutiveFailures int, totalSuccesses, totalFailures uint64)
	Allow() bool
}

// Config specifies the operational parameters for the circuit breaker.
type Config struct {
	FailureThreshold int           // Consecutive failures to transition Closed -> Open (default: 3)
	SuccessThreshold int           // Consecutive successes in Half-Open to transition -> Closed (default: 2)
	Timeout          time.Duration // Cooldown duration before transitioning Open -> Half-Open (default: 10s)
	ResetTimeout     time.Duration // Alias for Timeout if specified
	HalfOpenMaxCalls int           // Maximum concurrent probe calls in Half-Open state (default: 1)
	NowFunc          func() time.Time // Clock override for deterministic unit testing
}

// CircuitBreakerConfig is a type alias for Config.
type CircuitBreakerConfig = Config

// DefaultConfig returns production default settings.
func DefaultConfig() Config {
	return Config{
		FailureThreshold: 3,
		SuccessThreshold: 2,
		Timeout:          10 * time.Second,
		ResetTimeout:     10 * time.Second,
		HalfOpenMaxCalls: 1,
	}
}

// Breaker implements the CircuitBreaker interface with thread-safe atomic state transitions.
type Breaker struct {
	mu                   sync.RWMutex
	name                 string
	config               Config
	state                State
	consecutiveFailures  int
	consecutiveSuccesses int
	totalSuccesses       uint64
	totalFailures        uint64
	lastStateChange      time.Time
	halfOpenCalls        int
	nowFunc              func() time.Time
}

// NewCircuitBreaker constructs a new Breaker with flexible arguments:
// Accepts NewCircuitBreaker(cfg), NewCircuitBreaker(name, cfg), or NewCircuitBreaker().
func NewCircuitBreaker(args ...any) *Breaker {
	cfg := DefaultConfig()
	var name string

	for _, arg := range args {
		switch v := arg.(type) {
		case string:
			name = v
		case Config:
			cfg = v
		}
	}

	if cfg.ResetTimeout > 0 && cfg.Timeout == 0 {
		cfg.Timeout = cfg.ResetTimeout
	}
	if cfg.Timeout > 0 && cfg.ResetTimeout == 0 {
		cfg.ResetTimeout = cfg.Timeout
	}
	if cfg.FailureThreshold <= 0 {
		cfg.FailureThreshold = 3
	}
	if cfg.SuccessThreshold <= 0 {
		cfg.SuccessThreshold = 2
	}
	if cfg.Timeout <= 0 {
		cfg.Timeout = 10 * time.Second
		cfg.ResetTimeout = 10 * time.Second
	}
	if cfg.HalfOpenMaxCalls <= 0 {
		cfg.HalfOpenMaxCalls = 1
	}

	b := &Breaker{
		name:            name,
		config:          cfg,
		state:           StateClosed,
		nowFunc:         cfg.NowFunc,
	}
	b.lastStateChange = b.now()
	return b
}

func (cb *Breaker) now() time.Time {
	if cb.nowFunc != nil {
		return cb.nowFunc()
	}
	if cb.config.NowFunc != nil {
		return cb.config.NowFunc()
	}
	return time.Now()
}

// SetNowFunc overrides the clock for unit testing.
func (cb *Breaker) SetNowFunc(f func() time.Time) {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.nowFunc = f
}

// Name returns the identifier of the circuit breaker.
func (cb *Breaker) Name() string {
	return cb.name
}

// State returns the current circuit breaker state, taking timeout into account.
func (cb *Breaker) State() State {
	cb.mu.RLock()
	defer cb.mu.RUnlock()

	if cb.state == StateOpen && cb.now().Sub(cb.lastStateChange) >= cb.config.Timeout {
		return StateHalfOpen
	}
	return cb.state
}

// Counts returns operational telemetry.
func (cb *Breaker) Counts() (consecutiveSuccesses, consecutiveFailures int, totalSuccesses, totalFailures uint64) {
	cb.mu.RLock()
	defer cb.mu.RUnlock()
	return cb.consecutiveSuccesses, cb.consecutiveFailures, cb.totalSuccesses, cb.totalFailures
}

// Reset restores the circuit breaker to closed state with clean counters.
func (cb *Breaker) Reset() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = StateClosed
	cb.consecutiveFailures = 0
	cb.consecutiveSuccesses = 0
	cb.halfOpenCalls = 0
	cb.lastStateChange = cb.now()
}

// Trip forces the circuit breaker into open state.
func (cb *Breaker) Trip() {
	cb.mu.Lock()
	defer cb.mu.Unlock()
	cb.state = StateOpen
	cb.consecutiveFailures = 0
	cb.consecutiveSuccesses = 0
	cb.halfOpenCalls = 0
	cb.lastStateChange = cb.now()
}

// Allow checks whether an execution is permitted right now.
func (cb *Breaker) Allow() bool {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := cb.now()
	switch cb.state {
	case StateClosed:
		return true
	case StateOpen:
		if now.Sub(cb.lastStateChange) >= cb.config.Timeout {
			cb.state = StateHalfOpen
			cb.consecutiveFailures = 0
			cb.consecutiveSuccesses = 0
			cb.halfOpenCalls = 1
			cb.lastStateChange = now
			return true
		}
		return false
	case StateHalfOpen:
		if cb.halfOpenCalls >= cb.config.HalfOpenMaxCalls {
			return false
		}
		cb.halfOpenCalls++
		return true
	default:
		return true
	}
}

// RecordSuccess records a successful execution.
func (cb *Breaker) RecordSuccess() {
	cb.afterExecution(nil)
}

// RecordFailure records a failed execution.
func (cb *Breaker) RecordFailure(err error) {
	if err == nil {
		return
	}
	cb.afterExecution(err)
}

func (cb *Breaker) beforeExecution(ctx context.Context) error {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return err
		}
	}

	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := cb.now()
	switch cb.state {
	case StateOpen:
		if now.Sub(cb.lastStateChange) >= cb.config.Timeout {
			cb.state = StateHalfOpen
			cb.consecutiveFailures = 0
			cb.consecutiveSuccesses = 0
			cb.halfOpenCalls = 1
			cb.lastStateChange = now
			log.Printf("[CircuitBreaker] Transition: OPEN -> HALF-OPEN (timeout %.1fs elapsed, probing)", cb.config.Timeout.Seconds())
			return nil
		}
		return ErrCircuitOpen

	case StateHalfOpen:
		if cb.halfOpenCalls >= cb.config.HalfOpenMaxCalls {
			return ErrCircuitOpen
		}
		cb.halfOpenCalls++
		return nil

	case StateClosed:
		return nil

	default:
		return nil
	}
}

func (cb *Breaker) afterExecution(err error) {
	cb.mu.Lock()
	defer cb.mu.Unlock()

	now := cb.now()
	if err != nil {
		cb.totalFailures++
		switch cb.state {
		case StateClosed:
			cb.consecutiveSuccesses = 0
			cb.consecutiveFailures++
			if cb.consecutiveFailures >= cb.config.FailureThreshold {
				cb.state = StateOpen
				cb.lastStateChange = now
				cb.consecutiveFailures = 0
				cb.consecutiveSuccesses = 0
				log.Printf("[CircuitBreaker] Transition: CLOSED -> OPEN (failures >= %d)", cb.config.FailureThreshold)
			}
		case StateHalfOpen:
			cb.state = StateOpen
			cb.lastStateChange = now
			cb.consecutiveFailures = 0
			cb.consecutiveSuccesses = 0
			cb.halfOpenCalls = 0
			log.Printf("[CircuitBreaker] Transition: HALF-OPEN -> OPEN (probe failed: %v)", err)
		case StateOpen:
			// Already open
		}
	} else {
		cb.totalSuccesses++
		switch cb.state {
		case StateClosed:
			cb.consecutiveFailures = 0
			cb.consecutiveSuccesses++
		case StateHalfOpen:
			if cb.halfOpenCalls > 0 {
				cb.halfOpenCalls--
			}
			cb.consecutiveFailures = 0
			cb.consecutiveSuccesses++
			if cb.consecutiveSuccesses >= cb.config.SuccessThreshold {
				cb.state = StateClosed
				cb.lastStateChange = now
				cb.consecutiveFailures = 0
				cb.consecutiveSuccesses = 0
				cb.halfOpenCalls = 0
				log.Printf("[CircuitBreaker] Transition: HALF-OPEN -> CLOSED (probe successes >= %d)", cb.config.SuccessThreshold)
			}
		case StateOpen:
			// No action
		}
	}
}

// Execute wraps an operation with circuit breaker state management.
// op is executed without holding any locks to avoid blocking concurrent requests.
func (cb *Breaker) Execute(ctx context.Context, op func() error) (err error) {
	if err := cb.beforeExecution(ctx); err != nil {
		return err
	}

	defer func() {
		if r := recover(); r != nil {
			cb.afterExecution(fmt.Errorf("panic: %v", r))
			panic(r)
		}
	}()

	err = op()
	cb.afterExecution(err)
	return err
}
