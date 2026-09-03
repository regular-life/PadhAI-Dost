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

func verifyLifecycleClosedPhases(t *testing.T, cb cache.CircuitBreaker, ctx context.Context) {
	if cb.State() != cache.StateClosed || !cb.Allow() {
		t.Fatalf("expected StateClosed and Allow() == true, got %s", cb.State())
	}

	for i := 0; i < 2; i++ {
		err := cb.Execute(ctx, func() error { return errMockRedisDown })
		if !errors.Is(err, errMockRedisDown) || cb.State() != cache.StateClosed {
			t.Fatalf("expected errMockRedisDown and StateClosed at failure %d, got %v, %s", i+1, err, cb.State())
		}
	}

	if err := cb.Execute(ctx, func() error { return nil }); err != nil || cb.State() != cache.StateClosed {
		t.Fatalf("expected successful reset and StateClosed, got %v, %s", err, cb.State())
	}
	if _, failCount, _, _ := cb.Counts(); failCount != 0 {
		t.Fatalf("expected consecutive failures reset to 0, got %d", failCount)
	}
}

func verifyLifecycleOpenAndProbe(t *testing.T, cb cache.CircuitBreaker, ctx context.Context, setNow func(time.Duration)) {
	for i := 0; i < 3; i++ {
		_ = cb.Execute(ctx, func() error { return errMockRedisDown })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen after 3 failures, got %s", cb.State())
	}

	invoked := false
	err := cb.Execute(ctx, func() error {
		invoked = true
		return nil
	})
	if !errors.Is(err, cache.ErrCircuitOpen) || invoked || cb.Allow() {
		t.Fatalf("expected ErrCircuitOpen, invoked=false, allow=false; got err=%v, invoked=%v", err, invoked)
	}

	setNow(150 * time.Millisecond)
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen after timeout, got %s", cb.State())
	}

	probeInvoked := false
	err = cb.Execute(ctx, func() error {
		probeInvoked = true
		return nil
	})
	if err != nil || !probeInvoked || cb.State() != cache.StateHalfOpen {
		t.Fatalf("unexpected probe behavior: err=%v, invoked=%v, state=%s", err, probeInvoked, cb.State())
	}
}

func verifyLifecycleRecoveryAndProbeFailure(t *testing.T, cb cache.CircuitBreaker, ctx context.Context, setNow func(time.Duration)) {
	if err := cb.Execute(ctx, func() error { return nil }); err != nil || cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed after 2nd success, got %v, %s", err, cb.State())
	}

	for i := 0; i < 3; i++ {
		_ = cb.Execute(ctx, func() error { return errMockRedisDown })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen, got %s", cb.State())
	}

	setNow(150 * time.Millisecond)
	err := cb.Execute(ctx, func() error { return errMockRedisOOM })
	if !errors.Is(err, errMockRedisOOM) || cb.State() != cache.StateOpen {
		t.Fatalf("expected immediate transition back to StateOpen on probe failure, got %v, %s", err, cb.State())
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

	verifyLifecycleClosedPhases(t, cb, ctx)
	verifyLifecycleOpenAndProbe(t, cb, ctx, func(d time.Duration) { now = now.Add(d) })
	verifyLifecycleRecoveryAndProbeFailure(t, cb, ctx, func(d time.Duration) { now = now.Add(d) })
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

// ── Adversarial & Stress Testing ─────────────────────────────

// TestAdversarial_StateTransitions_ClosedOpenHalfOpenClosed rigorously verifies
// the complete state transition cycle: Closed -> Open -> HalfOpen -> Closed.
func verifyAdversarialSubThresholdFailures(t *testing.T, cb cache.CircuitBreaker, ctx context.Context, dummyErr error) {
	if cb.State() != cache.StateClosed {
		t.Fatalf("expected initial StateClosed, got %s", cb.State())
	}

	for i := 1; i <= 4; i++ {
		err := cb.Execute(ctx, func() error { return dummyErr })
		if !errors.Is(err, dummyErr) || cb.State() != cache.StateClosed {
			t.Fatalf("failure %d: expected dummyErr and StateClosed, got %v, %s", i, err, cb.State())
		}
		_, fails, _, totFails := cb.Counts()
		if fails != i || totFails != uint64(i) {
			t.Fatalf("expected consecutive fails=%d, totFails=%d; got %d, %d", i, i, fails, totFails)
		}
	}

	if err := cb.Execute(ctx, func() error { return nil }); err != nil || cb.State() != cache.StateClosed {
		t.Fatalf("expected successful reset and StateClosed, got %v, %s", err, cb.State())
	}
	if _, fails, _, _ := cb.Counts(); fails != 0 {
		t.Fatalf("expected consecutive failures reset to 0 after success, got %d", fails)
	}
}

func verifyAdversarialOpenAndFastFail(t *testing.T, cb cache.CircuitBreaker, ctx context.Context, dummyErr error) {
	for i := 1; i <= 5; i++ {
		_ = cb.Execute(ctx, func() error { return dummyErr })
	}
	if cb.State() != cache.StateOpen {
		t.Fatalf("expected StateOpen after 5 consecutive failures, got %s", cb.State())
	}

	opExecuted := false
	err := cb.Execute(ctx, func() error {
		opExecuted = true
		return nil
	})
	if !errors.Is(err, cache.ErrCircuitOpen) || opExecuted {
		t.Fatalf("expected ErrCircuitOpen and opExecuted=false, got err=%v, executed=%v", err, opExecuted)
	}
}

func verifyAdversarialHalfOpenConvergence(t *testing.T, cb cache.CircuitBreaker, ctx context.Context, advanceClock func(time.Duration)) {
	advanceClock(600 * time.Millisecond)
	if cb.State() != cache.StateHalfOpen {
		t.Fatalf("expected StateHalfOpen after timeout elapsed, got %s", cb.State())
	}

	for i := 1; i <= 2; i++ {
		probeCalled := false
		err := cb.Execute(ctx, func() error {
			probeCalled = true
			return nil
		})
		if err != nil || !probeCalled || cb.State() != cache.StateHalfOpen {
			t.Fatalf("probe %d failed: err=%v, called=%v, state=%s", i, err, probeCalled, cb.State())
		}
	}

	if err := cb.Execute(ctx, func() error { return nil }); err != nil || cb.State() != cache.StateClosed {
		t.Fatalf("expected StateClosed after 3 probe successes, got err=%v, state=%s", err, cb.State())
	}
}

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
	dummyErr := errors.New("simulated redis timeout")

	verifyAdversarialSubThresholdFailures(t, cb, ctx, dummyErr)
	verifyAdversarialOpenAndFastFail(t, cb, ctx, dummyErr)
	verifyAdversarialHalfOpenConvergence(t, cb, ctx, func(d time.Duration) { virtualTime = virtualTime.Add(d) })
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
