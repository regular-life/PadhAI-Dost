package cache_test

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
)

var (
	errMockRedisDown = errors.New("dial tcp 127.0.0.1:6379: connect: connection refused")
	errMockRedisOOM  = errors.New("OOM command not allowed when used memory > 'maxmemory'")
)

func TestCircuitBreaker_InitialState(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())

	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed, got %s", cb.State())
	}
	if !cb.Allow() {
		t.Fatal("expected Allow() == true in initial closed state")
	}

	succ, fail, totSucc, totFail := cb.Counts()
	if succ != 0 || fail != 0 || totSucc != 0 || totFail != 0 {
		t.Fatalf("expected zero counts, got succ=%d fail=%d totSucc=%d totFail=%d", succ, fail, totSucc, totFail)
	}
}

func TestCircuitBreaker_StateTransitions_FullLifecycle(t *testing.T) {
	t.Parallel()

	now := time.Now()
	nowFunc := func() time.Time { return now }

	cfg := cache.Config{
		FailureThreshold: 3,
		SuccessThreshold: 2,
		Timeout:          100 * time.Millisecond,
		HalfOpenMaxCalls: 1,
		NowFunc:          nowFunc,
	}

	cb := cache.NewCircuitBreaker("test-breaker", cfg)
	ctx := context.Background()

	// 1. Initial State: Closed
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed, got %s", cb.State())
	}
	if !cb.Allow() {
		t.Fatal("expected Allow() == true in Closed state")
	}

	// 2. Closed -> Closed (Sub-threshold failures)
	for i := 0; i < 2; i++ {
		err := cb.Execute(ctx, func() error { return errMockRedisDown })
		if !errors.Is(err, errMockRedisDown) {
			t.Fatalf("expected errMockRedisDown, got %v", err)
		}
		if cb.State() != cache.StateClosed {
			t.Fatalf("expected StateClosed at failure %d, got %s", i+1, cb.State())
		}
	}

	// Intermittent success resets failure count
	err := cb.Execute(ctx, func() error { return nil })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed, got %s", cb.State())
	}
	_, failCount, _, _ := cb.Counts()
	if failCount != 0 {
		t.Fatalf("expected consecutive failures reset to 0, got %d", failCount)
	}

	// 3. Closed -> Open (3 consecutive failures)
	for i := 0; i < 3; i++ {
		_ = cb.Execute(ctx, func() error { return errMockRedisDown })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen after 3 failures, got %s", cb.State())
	}

	// 4. Open Fast-Fail (Within Timeout)
	invoked := false
	err = cb.Execute(ctx, func() error {
		invoked = true
		return nil
	})
	if !errors.Is(err, cache.ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen, got %v", err)
	}
	if invoked {
		t.Fatal("expected closure NOT to be invoked in StateOpen")
	}
	if cb.Allow() {
		t.Fatal("expected Allow() == false in StateOpen within timeout")
	}

	// 5. Open -> Half-Open (Advance mock time by 150ms past 100ms Timeout)
	now = now.Add(150 * time.Millisecond)

	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen after timeout elapsed, got %s", cb.State())
	}

	// Next call executes as probe
	probeInvoked := false
	err = cb.Execute(ctx, func() error {
		probeInvoked = true
		return nil // Probe 1 success
	})
	if err != nil {
		t.Fatalf("unexpected probe error: %v", err)
	}
	if !probeInvoked {
		t.Fatal("expected probe closure to be invoked")
	}
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen during probing (1/2 successes), got %s", cb.State())
	}

	// 6. Half-Open -> Closed (2nd consecutive success reaches SuccessThreshold)
	err = cb.Execute(ctx, func() error { return nil })
	if err != nil {
		t.Fatalf("unexpected second success error: %v", err)
	}
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed after reaching success threshold, got %s", cb.State())
	}

	// 7. Half-Open -> Open (Probe Failure)
	// Trip to Open again
	for i := 0; i < 3; i++ {
		_ = cb.Execute(ctx, func() error { return errMockRedisDown })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen, got %s", cb.State())
	}

	// Advance time past Timeout to enter Half-Open
	now = now.Add(150 * time.Millisecond)

	// Probe fails -> must immediately trip back to StateOpen
	err = cb.Execute(ctx, func() error { return errMockRedisOOM })
	if !errors.Is(err, errMockRedisOOM) {
		t.Fatalf("expected errMockRedisOOM, got %v", err)
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected immediate transition back to StateOpen on probe failure, got %s", cb.State())
	}
}

func TestCircuitBreaker_HalfOpen_MaxCallsLimiting(t *testing.T) {
	t.Parallel()

	now := time.Now()
	nowFunc := func() time.Time { return now }

	cfg := cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 2,
		Timeout:          50 * time.Millisecond,
		HalfOpenMaxCalls: 1,
		NowFunc:          nowFunc,
	}
	cb := cache.NewCircuitBreaker(cfg)
	ctx := context.Background()

	// Trip breaker to Open
	for i := 0; i < 2; i++ {
		_ = cb.Execute(ctx, func() error { return errMockRedisDown })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen, got %s", cb.State())
	}

	// Advance time to Half-Open
	now = now.Add(100 * time.Millisecond)

	// In Half-Open, Allow() once succeeds (probe registered), second Allow() returns false
	if !cb.Allow() {
		t.Fatal("expected first Allow() in Half-Open to succeed")
	}
	if cb.Allow() {
		t.Fatal("expected second Allow() in Half-Open to fail due to HalfOpenMaxCalls=1")
	}
}

func TestCircuitBreaker_ContextCancellation(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // cancel immediately

	invoked := false
	err := cb.Execute(ctx, func() error {
		invoked = true
		return nil
	})

	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
	if invoked {
		t.Fatal("expected closure not to be invoked with cancelled context")
	}
	// Verify failure was NOT counted against circuit breaker
	_, failures, _, totFail := cb.Counts()
	if failures != 0 || totFail != 0 {
		t.Fatalf("expected 0 failures on cancelled context, got failures=%d totFail=%d", failures, totFail)
	}
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed, got %s", cb.State())
	}
}

func TestCircuitBreaker_PanicRecovery(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 1,
		Timeout:          10 * time.Second,
	})
	ctx := context.Background()

	panicked := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()
		_ = cb.Execute(ctx, func() error {
			panic("simulated redis driver panic")
		})
	}()

	if !panicked {
		t.Fatal("expected Execute to re-panic")
	}

	// Verify panic was recorded as a failure
	_, failCount, _, totFail := cb.Counts()
	if failCount != 1 || totFail != 1 {
		t.Fatalf("expected failure recorded from panic, got failCount=%d totFail=%d", failCount, totFail)
	}
}

func TestCircuitBreaker_ResetAndTrip(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	ctx := context.Background()

	// Manual Trip
	cb.Trip()
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen after Trip(), got %s", cb.State())
	}

	err := cb.Execute(ctx, func() error { return nil })
	if !errors.Is(err, cache.ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen after Trip(), got %v", err)
	}

	// Manual Reset
	cb.Reset()
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed after Reset(), got %s", cb.State())
	}

	succ, fail, totSucc, totFail := cb.Counts()
	if succ != 0 || fail != 0 {
		t.Fatalf("expected consecutive counts reset to 0, got succ=%d fail=%d", succ, fail)
	}
	_ = totSucc
	_ = totFail

	// Operations should succeed now
	err = cb.Execute(ctx, func() error { return nil })
	if err != nil {
		t.Fatalf("expected success after Reset(), got %v", err)
	}
}

func TestCircuitBreaker_CacheMissNotFailure(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	ctx := context.Background()

	// Simulate 10 cache misses (returning nil error from Execute closure)
	for i := 0; i < 10; i++ {
		err := cb.Execute(ctx, func() error {
			// Cache miss returns nil in RedisCache
			return nil
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed, got %s", cb.State())
	}
	_, failures, totalSucc, totalFail := cb.Counts()
	if failures != 0 || totalFail != 0 {
		t.Fatalf("expected 0 failures, got failures=%d totalFail=%d", failures, totalFail)
	}
	if totalSucc != 10 {
		t.Fatalf("expected 10 total successes, got %d", totalSucc)
	}
}

func TestCircuitBreaker_RecordSuccessAndFailure(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 2,
		Timeout:          10 * time.Second,
	})

	cb.RecordFailure(errMockRedisDown)
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed at 1 failure, got %s", cb.State())
	}

	cb.RecordFailure(errMockRedisDown)
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen at 2 failures, got %s", cb.State())
	}

	cb.Reset()
	cb.RecordSuccess()
	succ, _, _, _ := cb.Counts()
	if succ != 1 {
		t.Fatalf("expected 1 consecutive success, got %d", succ)
	}
}

func TestCircuitBreaker_HighConcurrency_200Goroutines(t *testing.T) {
	t.Parallel()

	cfg := cache.Config{
		FailureThreshold: 10,
		SuccessThreshold: 5,
		Timeout:          20 * time.Millisecond,
		HalfOpenMaxCalls: 2,
	}
	cb := cache.NewCircuitBreaker("concurrent-breaker", cfg)

	const goroutines = 200
	var wg sync.WaitGroup
	var executedCount int64
	var rejectedCount int64

	ctx := context.Background()

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			err := cb.Execute(ctx, func() error {
				atomic.AddInt64(&executedCount, 1)
				if idx%3 == 0 {
					return errMockRedisDown
				}
				return nil
			})
			if errors.Is(err, cache.ErrCircuitOpen) {
				atomic.AddInt64(&rejectedCount, 1)
			}
		}(i)
	}
	wg.Wait()

	total := atomic.LoadInt64(&executedCount) + atomic.LoadInt64(&rejectedCount)
	if total != goroutines {
		t.Fatalf("expected total executed + rejected == %d, got %d (executed=%d, rejected=%d)",
			goroutines, total, executedCount, rejectedCount)
	}
}
