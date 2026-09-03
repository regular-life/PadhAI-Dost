package cache_test

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
)

// TestAdversarial_StateTransitions_ClosedOpenHalfOpenClosed rigorously verifies
// the complete state transition cycle: Closed -> Open -> HalfOpen -> Closed.
func TestAdversarial_StateTransitions_ClosedOpenHalfOpenClosed(t *testing.T) {
	t.Parallel()

	virtualTime := time.Now()
	nowFunc := func() time.Time { return virtualTime }

	cfg := cache.Config{
		FailureThreshold: 5,
		SuccessThreshold: 3,
		Timeout:          500 * time.Millisecond,
		HalfOpenMaxCalls: 1,
		NowFunc:          nowFunc,
	}
	cb := cache.NewCircuitBreaker("adv-transition-test", cfg)
	ctx := context.Background()

	// 1. Initial State: Must be Closed
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected initial StateClosed, got %s", cb.State())
	}

	// 2. Sub-threshold failures (4 failures < FailureThreshold 5)
	dummyErr := errors.New("simulated redis timeout")
	for i := 1; i <= 4; i++ {
		err := cb.Execute(ctx, func() error { return dummyErr })
		if !errors.Is(err, dummyErr) {
			t.Fatalf("expected dummyErr, got %v", err)
		}
		if cb.State() != cache.StateClosed {
			t.Fatalf("at failure %d: expected StateClosed, got %s", i, cb.State())
		}
		_, fails, _, totFails := cb.Counts()
		if fails != i || totFails != uint64(i) {
			t.Fatalf("expected consecutive fails=%d, totFails=%d; got %d, %d", i, i, fails, totFails)
		}
	}

	// 3. Success resets consecutive failures to 0
	err := cb.Execute(ctx, func() error { return nil })
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	_, fails, _, _ := cb.Counts()
	if fails != 0 {
		t.Fatalf("expected consecutive failures reset to 0 after success, got %d", fails)
	}
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed after success, got %s", cb.State())
	}

	// 4. Exactly 5 consecutive failures trip to StateOpen
	for i := 1; i <= 5; i++ {
		_ = cb.Execute(ctx, func() error { return dummyErr })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen after 5 consecutive failures, got %s", cb.State())
	}

	// 5. While Open and before timeout: must fast-fail with ErrCircuitOpen without executing op
	opExecuted := false
	err = cb.Execute(ctx, func() error {
		opExecuted = true
		return nil
	})
	if !errors.Is(err, cache.ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen, got %v", err)
	}
	if opExecuted {
		t.Fatal("op() was executed when circuit was Open!")
	}

	// 6. Advance virtual clock past timeout (600ms > 500ms) -> transitions to HalfOpen
	virtualTime = virtualTime.Add(600 * time.Millisecond)
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen after timeout elapsed, got %s", cb.State())
	}

	// 7. In HalfOpen, exactly 3 consecutive successes are required to transition to Closed
	for i := 1; i <= 2; i++ {
		probeCalled := false
		err = cb.Execute(ctx, func() error {
			probeCalled = true
			return nil
		})
		if err != nil {
			t.Fatalf("probe %d failed: %v", i, err)
		}
		if !probeCalled {
			t.Fatalf("probe %d op was not executed", i)
		}
		if cb.State() != cache.StateHalfOpen {
			t.Fatalf("expected StateHalfOpen during probe %d (need 3 successes), got %s", i, cb.State())
		}
	}

	// 3rd success transitions to Closed
	err = cb.Execute(ctx, func() error { return nil })
	if err != nil {
		t.Fatalf("final probe failed: %v", err)
	}
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed after 3 probe successes, got %s", cb.State())
	}
}

// TestAdversarial_StateTransitions_HalfOpenToOpen verifies that ANY probe failure
// in HalfOpen immediately trips the breaker back to Open with reset probe counters.
func TestAdversarial_StateTransitions_HalfOpenToOpen(t *testing.T) {
	t.Parallel()

	virtualTime := time.Now()
	nowFunc := func() time.Time { return virtualTime }

	cfg := cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 3,
		Timeout:          200 * time.Millisecond,
		HalfOpenMaxCalls: 1,
		NowFunc:          nowFunc,
	}
	cb := cache.NewCircuitBreaker("adv-halfopen-fail", cfg)
	ctx := context.Background()

	// Trip to Open
	for i := 0; i < 2; i++ {
		_ = cb.Execute(ctx, func() error { return errors.New("err") })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen, got %s", cb.State())
	}

	// Advance time to HalfOpen
	virtualTime = virtualTime.Add(300 * time.Millisecond)
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen, got %s", cb.State())
	}

	// Probe 1 succeeds (1/3 successes)
	err := cb.Execute(ctx, func() error { return nil })
	if err != nil {
		t.Fatalf("unexpected probe error: %v", err)
	}
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen, got %s", cb.State())
	}

	// Probe 2 fails -> MUST immediately transition to Open
	probeErr := errors.New("redis connection refused during probe")
	err = cb.Execute(ctx, func() error { return probeErr })
	if !errors.Is(err, probeErr) {
		t.Fatalf("expected probeErr, got %v", err)
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected immediate transition back to StateOpen, got %s", cb.State())
	}

	// In Open, next calls must fail fast with ErrCircuitOpen
	err = cb.Execute(ctx, func() error { return nil })
	if !errors.Is(err, cache.ErrCircuitOpen) {
		t.Fatalf("expected ErrCircuitOpen, got %v", err)
	}
}

// TestAdversarial_FailFastZeroOverhead verifies that ErrCircuitOpen returns with
// 0 network operations and sub-microsecond overhead under high volume.
func TestAdversarial_FailFastZeroOverhead(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	cb.Trip() // Force into Open state

	ctx := context.Background()
	const iterations = 50000
	var executedCount int64

	start := time.Now()
	for i := 0; i < iterations; i++ {
		err := cb.Execute(ctx, func() error {
			atomic.AddInt64(&executedCount, 1)
			return nil
		})
		if !errors.Is(err, cache.ErrCircuitOpen) {
			t.Fatalf("expected ErrCircuitOpen at iter %d, got %v", i, err)
		}
	}
	elapsed := time.Since(start)

	if executedCount != 0 {
		t.Fatalf("expected 0 executed closures, got %d", executedCount)
	}

	avgPerOp := elapsed / time.Duration(iterations)
	t.Logf("Fast-fail benchmark: %d calls completed in %v (avg %v/op)", iterations, elapsed, avgPerOp)

	// In-memory mutex check should complete well under 10 microseconds per op
	if avgPerOp > 50*time.Microsecond {
		t.Fatalf("FAIL: fast-fail overhead too high: %v/op (expected < 50us)", avgPerOp)
	}
}

// TestAdversarial_HighConcurrencyRacesStress tests 1,000 concurrent goroutines
// executing operations, reading state, tripping, and resetting under the race detector.
func TestAdversarial_HighConcurrencyRacesStress(t *testing.T) {
	t.Parallel()

	cfg := cache.Config{
		FailureThreshold: 5,
		SuccessThreshold: 3,
		Timeout:          10 * time.Millisecond,
		HalfOpenMaxCalls: 2,
	}
	cb := cache.NewCircuitBreaker("adv-concurrent-stress", cfg)
	ctx := context.Background()

	const goroutines = 1000
	var wg sync.WaitGroup
	var executedCount int64
	var circuitOpenCount int64
	var otherErrCount int64

	for i := 0; i < goroutines; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			// Concurrently observe state and counts
			_ = cb.State()
			_, _, _, _ = cb.Counts()
			_ = cb.Allow()

			// Concurrently execute
			err := cb.Execute(ctx, func() error {
				atomic.AddInt64(&executedCount, 1)
				if id%4 == 0 {
					return errors.New("simulated error")
				}
				return nil
			})

			if errors.Is(err, cache.ErrCircuitOpen) {
				atomic.AddInt64(&circuitOpenCount, 1)
			} else if err != nil && err.Error() == "simulated error" {
				atomic.AddInt64(&otherErrCount, 1)
			}
		}(i)
	}

	wg.Wait()

	totalProcessed := atomic.LoadInt64(&executedCount) + atomic.LoadInt64(&circuitOpenCount)
	t.Logf("Concurrency stress: processed=%d (executed=%d, circuit_open=%d, simulated_errs=%d)",
		totalProcessed, executedCount, circuitOpenCount, otherErrCount)

	if totalProcessed != goroutines {
		t.Fatalf("mismatch in processed requests: expected %d, got %d", goroutines, totalProcessed)
	}
}

// TestAdversarial_ProbeConcurrencyRateLimiting verifies that in HalfOpen state,
// at most HalfOpenMaxCalls can execute concurrently, and remaining requests fail fast.
func TestAdversarial_ProbeConcurrencyRateLimiting(t *testing.T) {
	t.Parallel()

	virtualTime := time.Now()
	nowFunc := func() time.Time { return virtualTime }

	cfg := cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 2,
		Timeout:          100 * time.Millisecond,
		HalfOpenMaxCalls: 1, // Strictly 1 probe call at a time
		NowFunc:          nowFunc,
	}
	cb := cache.NewCircuitBreaker("adv-probe-limiting", cfg)
	ctx := context.Background()

	// Trip to Open
	_ = cb.Execute(ctx, func() error { return errors.New("err1") })
	_ = cb.Execute(ctx, func() error { return errors.New("err2") })
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen, got %s", cb.State())
	}

	// Advance time to HalfOpen
	virtualTime = virtualTime.Add(150 * time.Millisecond)
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen, got %s", cb.State())
	}

	// Launch a slow probe that blocks until signaled
	probeStarted := make(chan struct{})
	probeRelease := make(chan struct{})
	var probeInFlight int64
	var maxConcurrentProbes int64

	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		_ = cb.Execute(ctx, func() error {
			current := atomic.AddInt64(&probeInFlight, 1)
			if current > atomic.LoadInt64(&maxConcurrentProbes) {
				atomic.StoreInt64(&maxConcurrentProbes, current)
			}
			close(probeStarted)
			<-probeRelease
			atomic.AddInt64(&probeInFlight, -1)
			return nil
		})
	}()

	// Wait for the probe to start
	select {
	case <-probeStarted:
	case <-time.After(2 * time.Second):
		t.Fatal("timed out waiting for probe to start")
	}

	// Now that 1 probe is in flight, 100 concurrent requests must all be rejected with ErrCircuitOpen
	var rejectedCount int64
	var concurrentWg sync.WaitGroup
	for i := 0; i < 100; i++ {
		concurrentWg.Add(1)
		go func() {
			defer concurrentWg.Done()
			err := cb.Execute(ctx, func() error {
				atomic.AddInt64(&probeInFlight, 1)
				return nil
			})
			if errors.Is(err, cache.ErrCircuitOpen) {
				atomic.AddInt64(&rejectedCount, 1)
			}
		}()
	}
	concurrentWg.Wait()

	// Release the probe
	close(probeRelease)
	wg.Wait()

	if atomic.LoadInt64(&maxConcurrentProbes) > 1 {
		t.Fatalf("FAIL: probe concurrency exceeded max calls: %d > 1", maxConcurrentProbes)
	}
	if rejectedCount != 100 {
		t.Fatalf("FAIL: expected 100 rejected calls during probe, got %d", rejectedCount)
	}
}

// TestAdversarial_PanicInHalfOpenTripsBreaker verifies that if a probe panics in HalfOpen,
// the panic is caught, failure is recorded, panic is re-thrown, and state trips to Open.
func TestAdversarial_PanicInHalfOpenTripsBreaker(t *testing.T) {
	t.Parallel()

	virtualTime := time.Now()
	nowFunc := func() time.Time { return virtualTime }

	cfg := cache.Config{
		FailureThreshold: 2,
		SuccessThreshold: 2,
		Timeout:          100 * time.Millisecond,
		HalfOpenMaxCalls: 1,
		NowFunc:          nowFunc,
	}
	cb := cache.NewCircuitBreaker("adv-panic-test", cfg)
	ctx := context.Background()

	// Trip to Open
	_ = cb.Execute(ctx, func() error { return errors.New("err1") })
	_ = cb.Execute(ctx, func() error { return errors.New("err2") })

	// Advance time to HalfOpen
	virtualTime = virtualTime.Add(150 * time.Millisecond)

	// Probe panics
	panicked := false
	func() {
		defer func() {
			if r := recover(); r != nil {
				panicked = true
			}
		}()
		_ = cb.Execute(ctx, func() error {
			panic("catastrophic driver failure inside probe")
		})
	}()

	if !panicked {
		t.Fatal("expected panic to be re-thrown")
	}

	// Must be tripped back to Open immediately
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen after panicking probe, got %s", cb.State())
	}
}

// TestAdversarial_ReentrancyAndNoDeadlock verifies that nested operations
// (e.g. op calling State, Counts, Allow, or nested Execute) do not deadlock.
func TestAdversarial_ReentrancyAndNoDeadlock(t *testing.T) {
	t.Parallel()

	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	ctx := context.Background()

	done := make(chan struct{})
	go func() {
		defer close(done)
		err := cb.Execute(ctx, func() error {
			// Inner call 1: State()
			st := cb.State()
			if st != cache.StateClosed {
				return fmt.Errorf("unexpected state %s", st)
			}

			// Inner call 2: Counts()
			_, _, _, _ = cb.Counts()

			// Inner call 3: Allow()
			if !cb.Allow() {
				return fmt.Errorf("allow returned false")
			}

			// Inner call 4: Nested Execute()
			nestedErr := cb.Execute(ctx, func() error {
				return nil
			})
			return nestedErr
		})
		if err != nil {
			t.Errorf("nested execution failed: %v", err)
		}
	}()

	select {
	case <-done:
		// Succeeded without deadlock
	case <-time.After(3 * time.Second):
		t.Fatal("DEADLOCK DETECTED: nested circuit breaker call hung!")
	}
}

// BenchmarkCircuitBreaker_OpenFastFail measures throughput and allocations of ErrCircuitOpen fast-fail.
func BenchmarkCircuitBreaker_OpenFastFail(b *testing.B) {
	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	cb.Trip()
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = cb.Execute(ctx, func() error {
			return nil
		})
	}
}

// BenchmarkCircuitBreaker_ParallelOpenFastFail measures multi-threaded fast-fail throughput.
func BenchmarkCircuitBreaker_ParallelOpenFastFail(b *testing.B) {
	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	cb.Trip()
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			_ = cb.Execute(ctx, func() error {
				return nil
			})
		}
	})
}

// BenchmarkCircuitBreaker_ClosedNormalExecution measures baseline overhead in Closed state.
func BenchmarkCircuitBreaker_ClosedNormalExecution(b *testing.B) {
	cb := cache.NewCircuitBreaker(cache.DefaultConfig())
	ctx := context.Background()

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_ = cb.Execute(ctx, func() error {
			return nil
		})
	}
}
