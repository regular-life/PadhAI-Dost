package handlers

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/council"
)

var w3cTraceparentPattern = regexp.MustCompile(`^00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})$`)

// ─────────────────────────────────────────────────────────────────────────────
// Suite 1: Malformed & Adversarial Incoming traceparent Headers Matrix
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_MalformedTraceparent_ExtensiveMatrix(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	malformedCases := []struct {
		name      string
		headerVal string
	}{
		{"EmptyHeader", ""},
		{"WhitespaceOnly", "   "},
		{"TruncatedVersion", "0"},
		{"InvalidVersionNonHex", "zz-1234567890abcdef1234567890abcdef-abcdef1234567890-01"},
		{"InvalidVersionFF", "ff-1234567890abcdef1234567890abcdef-abcdef1234567890-01"},
		{"ThreeDigitVersion", "000-1234567890abcdef1234567890abcdef-abcdef1234567890-01"},
		{"TruncatedTraceID", "00-1234-abcdef1234567890-01"},
		{"TooLongTraceID", "00-1234567890abcdef1234567890abcdef00-abcdef1234567890-01"},
		{"NonHexTraceID", "00-1234567890abcdef1234567890abcdeg-abcdef1234567890-01"},
		{"AllZeroTraceID", "00-00000000000000000000000000000000-abcdef1234567890-01"},
		{"TruncatedParentSpanID", "00-1234567890abcdef1234567890abcdef-123-01"},
		{"TooLongParentSpanID", "00-1234567890abcdef1234567890abcdef-abcdef123456789012-01"},
		{"NonHexParentSpanID", "00-1234567890abcdef1234567890abcdef-abcdef123456789z-01"},
		{"AllZeroParentSpanID", "00-1234567890abcdef1234567890abcdef-0000000000000000-01"},
		{"MissingHyphens", "001234567890abcdef1234567890abcdefabcdef123456789001"},
		{"ExtraDashes", "00--1234567890abcdef1234567890abcdef--abcdef1234567890--01"},
		{"TrailingGarbage", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-01-extra-payload"},
		{"InvalidFlags", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-gg"},
		{"TruncatedFlags", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-0"},
		{"TooLongFlags", "00-1234567890abcdef1234567890abcdef-abcdef1234567890-001"},
		{"SQLInjectionPayload", "'; DROP TABLE traces; --"},
		{"UnicodeEmojiPayload", "00-💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡💡-🚀🚀🚀🚀🚀🚀🚀🚀-01"},
		{"HugeHeaderString", "00-" + strings.Repeat("a", 10000) + "-01"},
		{"ControlCharacters", "00-\x00\x01\x02\r\n\t-01"},
		{"RandomGarbageText", "not-a-traceparent-at-all"},
	}

	for _, tc := range malformedCases {
		tc := tc
		t.Run(tc.name+"_JSON", func(t *testing.T) {
			reqBody := fmt.Sprintf(`{"question": "Malformed test %s", "doc_id": "doc_malformed"}`, tc.name)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			if tc.headerVal != "" {
				req.Header.Set("traceparent", tc.headerVal)
			}

			w := httptest.NewRecorder()
			f.handlers.HandleQuery(w, req)

			// 1. Must never return 500 Internal Server Error
			if w.Code != http.StatusOK {
				t.Fatalf("case %s failed with unexpected HTTP status %d: %s", tc.name, w.Code, w.Body.String())
			}

			// 2. Response headers must contain valid traceparent and X-Trace-ID
			respTraceID := w.Header().Get("X-Trace-ID")
			if respTraceID == "" {
				t.Fatalf("case %s missing X-Trace-ID response header", tc.name)
			}
			if len(respTraceID) != 32 || respTraceID == "00000000000000000000000000000000" {
				t.Fatalf("case %s produced invalid X-Trace-ID %q", tc.name, respTraceID)
			}

			respTraceparent := w.Header().Get("traceparent")
			if respTraceparent == "" {
				t.Fatalf("case %s missing traceparent response header", tc.name)
			}
			match := w3cTraceparentPattern.FindStringSubmatch(respTraceparent)
			if len(match) != 4 {
				t.Fatalf("case %s traceparent response header %q invalid format", tc.name, respTraceparent)
			}
			if match[1] != respTraceID {
				t.Fatalf("case %s traceparent TraceID %s != X-Trace-ID %s", tc.name, match[1], respTraceID)
			}
		})

		t.Run(tc.name+"_SSE", func(t *testing.T) {
			reqBody := fmt.Sprintf(`{"question": "Malformed SSE test %s", "doc_id": "doc_malformed_sse"}`, tc.name)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")
			req.Header.Set("Accept", "text/event-stream")
			if tc.headerVal != "" {
				req.Header.Set("traceparent", tc.headerVal)
			}

			w := httptest.NewRecorder()
			f.handlers.HandleQuery(w, req)

			if w.Code != http.StatusOK {
				t.Fatalf("case %s SSE failed with status %d", tc.name, w.Code)
			}

			respContentType := w.Header().Get("Content-Type")
			if !strings.HasPrefix(respContentType, "text/event-stream") {
				t.Fatalf("case %s SSE expected Content-Type text/event-stream, got %q", tc.name, respContentType)
			}

			respTraceID := w.Header().Get("X-Trace-ID")
			if respTraceID == "" || len(respTraceID) != 32 || respTraceID == "00000000000000000000000000000000" {
				t.Fatalf("case %s SSE invalid X-Trace-ID %q", tc.name, respTraceID)
			}

			// Verify SSE frames parsed cleanly without error
			scanner := bufio.NewScanner(w.Body)
			var hasCandidateDraft, hasPeerReview, hasFinalAnswer bool
			for scanner.Scan() {
				line := scanner.Text()
				if strings.HasPrefix(line, "event: error") {
					t.Fatalf("case %s SSE emitted error frame on malformed traceparent: %s", tc.name, w.Body.String())
				}
				if strings.HasPrefix(line, "event: candidate_draft") {
					hasCandidateDraft = true
				}
				if strings.HasPrefix(line, "event: peer_review") {
					hasPeerReview = true
				}
				if strings.HasPrefix(line, "event: final_answer") {
					hasFinalAnswer = true
				}
			}

			if !hasCandidateDraft || !hasPeerReview || !hasFinalAnswer {
				t.Errorf("case %s SSE missing expected event frames: draft=%v, review=%v, final=%v",
					tc.name, hasCandidateDraft, hasPeerReview, hasFinalAnswer)
			}
		})
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 2: High Concurrency under -race with 64 Concurrent Requests
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_HighConcurrency_RaceDetector_64Goroutines(t *testing.T) {
	// Tests 64 concurrent requests generating spans simultaneously under Go race detector.
	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	numWorkers := 64
	var wg sync.WaitGroup
	wg.Add(numWorkers)

	startBarrier := make(chan struct{})

	type workerResult struct {
		index       int
		statusCode  int
		traceID     string
		traceparent string
		isSSE       bool
		err         error
	}

	results := make([]workerResult, numWorkers)

	for i := 0; i < numWorkers; i++ {
		go func(idx int) {
			defer wg.Done()

			// Decide configuration per worker:
			// 0-15: JSON with incoming traceparent
			// 16-31: JSON without traceparent
			// 32-47: SSE with incoming traceparent
			// 48-63: SSE with malformed traceparent
			isSSE := idx >= 32
			hasIncomingTrace := (idx < 16) || (idx >= 32 && idx < 48)
			hasMalformedTrace := idx >= 48

			reqBody := fmt.Sprintf(`{"question": "Concurrency query worker %d", "doc_id": "doc_conc_%d"}`, idx, idx%4)
			req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
			req.Header.Set("Content-Type", "application/json")

			if isSSE {
				req.Header.Set("Accept", "text/event-stream")
			}

			expectedTraceID := ""
			if hasIncomingTrace {
				expectedTraceID = fmt.Sprintf("a%031x", idx)
				incomingParentID := fmt.Sprintf("b%015x", idx)
				req.Header.Set("traceparent", fmt.Sprintf("00-%s-%s-01", expectedTraceID, incomingParentID))
			} else if hasMalformedTrace {
				req.Header.Set("traceparent", fmt.Sprintf("malformed-header-worker-%d-!", idx))
			}

			// Synchronize all goroutines to fire at the exact same instant
			<-startBarrier

			w := httptest.NewRecorder()
			f.handlers.HandleQuery(w, req)

			results[idx] = workerResult{
				index:       idx,
				statusCode:  w.Code,
				traceID:     w.Header().Get("X-Trace-ID"),
				traceparent: w.Header().Get("traceparent"),
				isSSE:       isSSE,
			}
		}(i)
	}

	// Release all 64 goroutines simultaneously
	close(startBarrier)
	wg.Wait()

	// Analyze results
	seenTraceIDs := make(map[string]int)

	for i, res := range results {
		if res.statusCode != http.StatusOK {
			t.Fatalf("worker %d returned HTTP %d", i, res.statusCode)
		}
		if res.traceID == "" || len(res.traceID) != 32 {
			t.Fatalf("worker %d produced empty or invalid TraceID %q", i, res.traceID)
		}
		if res.traceparent == "" || !w3cTraceparentPattern.MatchString(res.traceparent) {
			t.Fatalf("worker %d produced invalid traceparent header %q", i, res.traceparent)
		}

		// Verify that workers with explicit incoming traceparent received their exact TraceID
		if (i < 16) || (i >= 32 && i < 48) {
			expectedTraceID := fmt.Sprintf("a%031x", i)
			if res.traceID != expectedTraceID {
				t.Fatalf("worker %d expected incoming TraceID %s, but got %s", i, expectedTraceID, res.traceID)
			}
		}

		seenTraceIDs[res.traceID]++
	}

	// Verify that each worker without an incoming traceparent received a unique TraceID
	if len(seenTraceIDs) < numWorkers {
		t.Fatalf("detected duplicate trace IDs across concurrent requests: %d unique traces out of %d requests",
			len(seenTraceIDs), numWorkers)
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 3: SSE Streaming Deliberation Trace Context & Event Frame Verification
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_SSEStreaming_TraceContextIntegrity(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	incomingTraceID := "feedfacefeedfacefeedfacefeedface"
	incomingParentID := "c001cafe12345678"
	incomingHeader := "00-" + incomingTraceID + "-" + incomingParentID + "-01"

	reqBody := `{"question": "Verify SSE trace context integrity", "doc_id": "doc_sse_trace"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("traceparent", incomingHeader)

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK for SSE query, got %d", w.Code)
	}

	// 1. Response Headers Verification
	if ct := w.Header().Get("Content-Type"); !strings.Contains(ct, "text/event-stream") {
		t.Errorf("expected Content-Type text/event-stream, got %q", ct)
	}
	if w.Header().Get("X-Trace-ID") != incomingTraceID {
		t.Errorf("expected X-Trace-ID %s, got %s", incomingTraceID, w.Header().Get("X-Trace-ID"))
	}
	if !strings.Contains(w.Header().Get("traceparent"), incomingTraceID) {
		t.Errorf("expected response traceparent header %q to contain incoming TraceID %s",
			w.Header().Get("traceparent"), incomingTraceID)
	}

	// 2. SSE Frame Stream Parsing & Content Validation
	scanner := bufio.NewScanner(w.Body)
	var currentEvent string
	var candidateDrafts []council.CandidateDraftPayload
	var peerReviews []council.PeerReviewPayload
	var finalAnswer *QueryResponse
	var errorPayloads []map[string]interface{}

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			dataJSON := strings.TrimPrefix(line, "data: ")
			switch currentEvent {
			case "candidate_draft":
				var draft council.CandidateDraftPayload
				if err := json.Unmarshal([]byte(dataJSON), &draft); err != nil {
					t.Fatalf("failed to unmarshal candidate_draft data: %v", err)
				}
				candidateDrafts = append(candidateDrafts, draft)
			case "peer_review":
				var review council.PeerReviewPayload
				if err := json.Unmarshal([]byte(dataJSON), &review); err != nil {
					t.Fatalf("failed to unmarshal peer_review data: %v", err)
				}
				peerReviews = append(peerReviews, review)
			case "final_answer":
				var resp QueryResponse
				if err := json.Unmarshal([]byte(dataJSON), &resp); err != nil {
					t.Fatalf("failed to unmarshal final_answer data: %v", err)
				}
				finalAnswer = &resp
			case "error":
				var errMap map[string]interface{}
				_ = json.Unmarshal([]byte(dataJSON), &errMap)
				errorPayloads = append(errorPayloads, errMap)
			}
		}
	}

	if len(errorPayloads) > 0 {
		t.Fatalf("unexpected error frames encountered in SSE stream: %v", errorPayloads)
	}
	if len(candidateDrafts) < 2 {
		t.Fatalf("expected at least 2 candidate drafts streamed, got %d", len(candidateDrafts))
	}
	if len(peerReviews) < 2 {
		t.Fatalf("expected at least 2 peer reviews streamed, got %d", len(peerReviews))
	}
	if finalAnswer == nil {
		t.Fatal("expected final_answer event in SSE stream, got nil")
	}
	if finalAnswer.Answer == "" {
		t.Fatal("expected non-empty final answer text in final_answer payload")
	}

	// 3. OpenTelemetry Span Tree Verification
	spans := f.exporter.GetSpans()
	if len(spans) == 0 {
		t.Fatal("expected spans to be recorded for SSE deliberation, got 0")
	}

	tree := BuildSpanTree(t, spans)

	// Verify all spans share incoming TraceID
	for _, s := range spans {
		if s.SpanContext.TraceID().String() != incomingTraceID {
			t.Errorf("span %q TraceID %s does not match expected TraceID %s",
				s.Name, s.SpanContext.TraceID(), incomingTraceID)
		}
	}

	// Verify Root Attributes for SSE
	var foundSSEAttr, foundDocAttr bool
	for _, a := range tree.Root.Attributes {
		if a.Key == "query.sse" && a.Value.AsBool() == true {
			foundSSEAttr = true
		}
		if a.Key == "query.doc_id" && a.Value.AsString() == "doc_sse_trace" {
			foundDocAttr = true
		}
	}
	if !foundSSEAttr {
		t.Error("expected query.sse=true attribute on root span")
	}
	if !foundDocAttr {
		t.Error("expected query.doc_id=doc_sse_trace attribute on root span")
	}

	// Verify Council Stage Spans are properly linked to Root
	rootName := "HTTP POST /api/v1/query"
	tree.AssertParent(t, "council.candidate_fan_out", rootName)
	tree.AssertParent(t, "council.candidate_model", "council.candidate_fan_out")
	tree.AssertParent(t, "council.chairman_deliberation", rootName)
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 4: Client Premature Disconnection during SSE Deliberation
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_SSEStreaming_ClientPrematureDisconnect(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	ctx, cancel := context.WithCancel(context.Background())

	reqBody := `{"question": "Simulate client disconnect mid-deliberation", "doc_id": "doc_disconnect"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody)).WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "text/event-stream")

	// Trigger cancellation immediately to simulate client network drop
	cancel()

	w := httptest.NewRecorder()
	done := make(chan struct{})

	go func() {
		f.handlers.HandleQuery(w, req)
		close(done)
	}()

	select {
	case <-done:
		// Clean exit without hanging or panicking
	case <-time.After(2 * time.Second):
		t.Fatal("HandleQuery hung on client premature disconnect")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 5: Outgoing Python RAG Header Propagation Verification
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_OutgoingPythonRAG_TraceparentPropagation(t *testing.T) {
	t.Parallel()

	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	reqBody := `{"question": "Outgoing header verification", "doc_id": "doc_outgoing"}`
	req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
	req.Header.Set("Content-Type", "application/json")

	w := httptest.NewRecorder()
	f.handlers.HandleQuery(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200, got %d", w.Code)
	}

	f.callsMu.Lock()
	calls := f.recordedCalls
	f.callsMu.Unlock()

	var embedFound, retrieveFound bool
	for _, call := range calls {
		if strings.HasSuffix(call.Path, "/embed") {
			embedFound = true
			if call.Traceparent == "" || !w3cTraceparentPattern.MatchString(call.Traceparent) {
				t.Fatalf("outgoing /embed call missing valid W3C traceparent header, got %q", call.Traceparent)
			}
		}
		if strings.HasSuffix(call.Path, "/retrieve") {
			retrieveFound = true
			if call.Traceparent == "" || !w3cTraceparentPattern.MatchString(call.Traceparent) {
				t.Fatalf("outgoing /retrieve call missing valid W3C traceparent header, got %q", call.Traceparent)
			}
		}
	}

	if !embedFound {
		t.Error("expected outgoing HTTP call to /embed")
	}
	if !retrieveFound {
		t.Error("expected outgoing HTTP call to /retrieve")
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Suite 6: Fast Sub-Second Performance Guarantee
// ─────────────────────────────────────────────────────────────────────────────

func TestChallengerM4_BenchmarkLatencyOverhead(t *testing.T) {
	f := setupTelemetryFixture(t)
	defer f.ragServer.Close()

	iterations := 20
	start := time.Now()

	for i := 0; i < iterations; i++ {
		reqBody := fmt.Sprintf(`{"question": "Bench %d", "doc_id": "doc_perf"}`, i)
		req := httptest.NewRequest(http.MethodPost, "/api/v1/query", strings.NewReader(reqBody))
		req.Header.Set("Content-Type", "application/json")

		w := httptest.NewRecorder()
		f.handlers.HandleQuery(w, req)

		if w.Code != http.StatusOK {
			t.Fatalf("iteration %d failed with status %d", i, w.Code)
		}
	}

	elapsed := time.Since(start)
	avgPerOp := elapsed / time.Duration(iterations)
	t.Logf("Executed %d traced queries in %v (avg: %v/query)", iterations, elapsed, avgPerOp)

	if elapsed > 500*time.Millisecond {
		t.Fatalf("tracing overhead too high: %d iterations took %v (limit: 500ms)", iterations, elapsed)
	}
}
