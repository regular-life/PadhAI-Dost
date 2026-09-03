package council

import (
	"context"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

type testMockLLMClient struct {
	name             string
	delay            time.Duration
	fail             bool
	generateChatFunc func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error)
	generateFunc     func(ctx context.Context, prompt string) (*llm.Response, error)
}

func (m *testMockLLMClient) ModelName() string {
	if m.name != "" {
		return m.name
	}
	return "mock:test-model"
}

func (m *testMockLLMClient) Generate(ctx context.Context, prompt string) (*llm.Response, error) {
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if m.fail {
		return nil, context.DeadlineExceeded
	}
	if m.generateFunc != nil {
		return m.generateFunc(ctx, prompt)
	}
	return &llm.Response{
		Answer:     "RANKING: A, B, C\nREASONING: Model A provides the best answer",
		Model:      m.ModelName(),
		Confidence: 0.9,
	}, nil
}

func (m *testMockLLMClient) GenerateChat(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
	if m.delay > 0 {
		select {
		case <-time.After(m.delay):
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	if m.fail {
		return nil, context.DeadlineExceeded
	}
	if m.generateChatFunc != nil {
		return m.generateChatFunc(ctx, opts)
	}
	return &llm.Response{
		Answer:     "Mock candidate draft from " + m.ModelName(),
		Model:      m.ModelName(),
		Confidence: 0.9,
	}, nil
}

func TestOrchestratorQueryStream_Mock(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	orch := NewOrchestrator(nil, nil, 5*time.Second)
	eventChan := make(chan StreamEvent, 16)

	done := make(chan struct{})
	var events []StreamEvent

	go func() {
		for ev := range eventChan {
			events = append(events, ev)
		}
		close(done)
	}()

	res, err := orch.QueryStream(context.Background(), "What is CouncilAI?", nil, "", false, "council", nil, eventChan)
	<-done

	if err != nil {
		t.Fatalf("QueryStream failed: %v", err)
	}
	if res == nil {
		t.Fatalf("expected non-nil CouncilResult")
	}

	var draftCount, reviewCount int
	var foundFinal bool

	for _, ev := range events {
		switch ev.Type {
		case EventCandidateDraft:
			draftCount++
			payload, ok := ev.Data.(CandidateDraftPayload)
			if !ok {
				t.Errorf("expected CandidateDraftPayload, got %T", ev.Data)
			}
			if payload.Answer == "" || payload.Model == "" {
				t.Errorf("invalid draft payload: %+v", payload)
			}
		case EventPeerReview:
			reviewCount++
			payload, ok := ev.Data.(PeerReviewPayload)
			if !ok {
				t.Errorf("expected PeerReviewPayload, got %T", ev.Data)
			}
			if payload.Reviewer == "" || payload.Review == "" {
				t.Errorf("invalid review payload: %+v", payload)
			}
		case EventFinalAnswer:
			foundFinal = true
			result, ok := ev.Data.(*CouncilResult)
			if !ok {
				t.Errorf("expected *CouncilResult, got %T", ev.Data)
			}
			if result.FinalAnswer == "" {
				t.Errorf("empty final answer in event")
			}
		}
	}

	if draftCount != 3 {
		t.Errorf("expected 3 candidate_draft events, got %d", draftCount)
	}
	if reviewCount != 3 {
		t.Errorf("expected 3 peer_review events, got %d", reviewCount)
	}
	if !foundFinal {
		t.Errorf("final_answer event was not emitted")
	}
}

func TestOrchestratorQueryDirectStream_Mock(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	orch := NewOrchestrator(nil, nil, 5*time.Second)
	eventChan := make(chan StreamEvent, 16)

	done := make(chan struct{})
	var events []StreamEvent

	go func() {
		for ev := range eventChan {
			events = append(events, ev)
		}
		close(done)
	}()

	res, err := orch.QueryDirectStream(context.Background(), "Quick question", nil, nil, eventChan)
	<-done

	if err != nil {
		t.Fatalf("QueryDirectStream failed: %v", err)
	}
	if res == nil {
		t.Fatalf("expected non-nil CouncilResult")
	}

	var foundDraft, foundFinal bool
	for _, ev := range events {
		if ev.Type == EventCandidateDraft {
			foundDraft = true
		}
		if ev.Type == EventFinalAnswer {
			foundFinal = true
		}
	}

	if !foundDraft {
		t.Errorf("expected candidate_draft event in direct stream")
	}
	if !foundFinal {
		t.Errorf("expected final_answer event in direct stream")
	}
}

func TestOrchestratorDeliberateStream(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	orch := NewOrchestrator(nil, nil, 5*time.Second)
	eventChan := make(chan StreamEvent, 16)

	done := make(chan struct{})
	var events []StreamEvent

	go func() {
		for ev := range eventChan {
			events = append(events, ev)
		}
		close(done)
	}()

	res, err := orch.DeliberateStream(context.Background(), "Deliberate test", eventChan)
	<-done

	if err != nil {
		t.Fatalf("DeliberateStream failed: %v", err)
	}
	if res == nil {
		t.Fatalf("expected non-nil CouncilResult")
	}
	if len(events) == 0 {
		t.Errorf("expected streaming events from DeliberateStream")
	}
}

func TestOrchestratorQuery_Synchronous_BackwardCompatibility(t *testing.T) {
	os.Setenv("MOCK_LLM", "true")
	defer os.Unsetenv("MOCK_LLM")

	orch := NewOrchestrator(nil, nil, 5*time.Second)

	// Test Query
	res, err := orch.Query(context.Background(), "Sync query", nil, "", false, "council", nil)
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if res.FinalAnswer == "" {
		t.Errorf("expected non-empty FinalAnswer")
	}

	// Test QueryDirect
	resDirect, err := orch.QueryDirect(context.Background(), "Sync direct", nil, nil)
	if err != nil {
		t.Fatalf("QueryDirect failed: %v", err)
	}
	if resDirect.FinalAnswer == "" {
		t.Errorf("expected non-empty FinalAnswer for direct")
	}

	// Test Deliberate
	resDelib, err := orch.Deliberate(context.Background(), "Sync deliberate")
	if err != nil {
		t.Fatalf("Deliberate failed: %v", err)
	}
	if resDelib.FinalAnswer == "" {
		t.Errorf("expected non-empty FinalAnswer for deliberate")
	}
}

func TestOrchestratorQueryStream_RealClients(t *testing.T) {
	clientA := &testMockLLMClient{
		name: "mock:model-a",
		generateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Response from A", Model: "mock:model-a"}, nil
		},
		generateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: A, B\nREASONING: A is superior", Model: "mock:model-a"}, nil
		},
	}
	clientB := &testMockLLMClient{
		name: "mock:model-b",
		generateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Response from B", Model: "mock:model-b"}, nil
		},
		generateFunc: func(ctx context.Context, prompt string) (*llm.Response, error) {
			return &llm.Response{Answer: "RANKING: B, A\nREASONING: B is superior", Model: "mock:model-b"}, nil
		},
	}
	chairman := &testMockLLMClient{
		name: "mock:chairman",
		generateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{
				Answer: `{"answer":"Synthesized consensus answer","confidence":0.95,"source":"chairman:mock","reasoning":"Balanced synthesis"}`,
				Model:  "mock:chairman",
			}, nil
		},
	}

	orch := NewOrchestrator([]llm.LLMClient{clientA, clientB}, chairman, 5*time.Second)
	eventChan := make(chan StreamEvent, 16)

	done := make(chan struct{})
	var events []StreamEvent

	go func() {
		for ev := range eventChan {
			events = append(events, ev)
		}
		close(done)
	}()

	res, err := orch.QueryStream(context.Background(), "Explain LLM deliberation", nil, "", false, "council", nil, eventChan)
	<-done

	if err != nil {
		t.Fatalf("QueryStream failed: %v", err)
	}
	if res.FinalAnswer != "Synthesized consensus answer" {
		t.Errorf("expected 'Synthesized consensus answer', got %q", res.FinalAnswer)
	}

	drafts := 0
	reviews := 0
	finals := 0

	for _, ev := range events {
		switch ev.Type {
		case EventCandidateDraft:
			drafts++
		case EventPeerReview:
			reviews++
		case EventFinalAnswer:
			finals++
		}
	}

	if drafts != 2 {
		t.Errorf("expected 2 drafts, got %d", drafts)
	}
	if reviews != 2 {
		t.Errorf("expected 2 reviews, got %d", reviews)
	}
	if finals != 1 {
		t.Errorf("expected 1 final answer, got %d", finals)
	}
}

func TestOrchestratorQueryStream_SingleModelFallback(t *testing.T) {
	clientA := &testMockLLMClient{
		name: "mock:model-a",
		generateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Only valid response", Model: "mock:model-a"}, nil
		},
	}
	clientB := &testMockLLMClient{
		name: "mock:model-b",
		fail: true,
	}

	orch := NewOrchestrator([]llm.LLMClient{clientA, clientB}, nil, 5*time.Second)
	eventChan := make(chan StreamEvent, 16)

	done := make(chan struct{})
	var events []StreamEvent

	go func() {
		for ev := range eventChan {
			events = append(events, ev)
		}
		close(done)
	}()

	res, err := orch.QueryStream(context.Background(), "Single model test", nil, "", false, "council", nil, eventChan)
	<-done

	if err != nil {
		t.Fatalf("QueryStream failed: %v", err)
	}
	if res.FinalAnswer != "Only valid response" {
		t.Errorf("expected 'Only valid response', got %q", res.FinalAnswer)
	}
	if !strings.Contains(res.Source, "single-response") {
		t.Errorf("expected source to contain 'single-response', got %q", res.Source)
	}
}

func TestOrchestratorQueryStream_AllModelsFail(t *testing.T) {
	clientA := &testMockLLMClient{name: "mock:model-a", fail: true}
	clientB := &testMockLLMClient{name: "mock:model-b", fail: true}

	orch := NewOrchestrator([]llm.LLMClient{clientA, clientB}, nil, 5*time.Second)
	eventChan := make(chan StreamEvent, 16)

	done := make(chan struct{})
	var events []StreamEvent

	go func() {
		for ev := range eventChan {
			events = append(events, ev)
		}
		close(done)
	}()

	res, err := orch.QueryStream(context.Background(), "All fail test", nil, "", false, "council", nil, eventChan)
	<-done

	if err == nil {
		t.Fatalf("expected error when all models fail, got nil")
	}
	if res != nil {
		t.Fatalf("expected nil result on failure, got %+v", res)
	}

	var foundError bool
	for _, ev := range events {
		if ev.Type == EventError {
			foundError = true
		}
	}
	if !foundError {
		t.Errorf("expected EventError to be emitted when all models fail")
	}
}

func TestOrchestratorQueryStream_TTFT_AsynchronousEmission(t *testing.T) {
	clientFast := &testMockLLMClient{
		name:  "mock:fast",
		delay: 5 * time.Millisecond,
		generateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Fast response", Model: "mock:fast"}, nil
		},
	}
	clientSlow := &testMockLLMClient{
		name:  "mock:slow",
		delay: 50 * time.Millisecond,
		generateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: "Slow response", Model: "mock:slow"}, nil
		},
	}
	chairman := &testMockLLMClient{
		name: "mock:chairman",
		generateChatFunc: func(ctx context.Context, opts llm.GenerateOptions) (*llm.Response, error) {
			return &llm.Response{Answer: `{"answer":"Final answer"}`, Model: "mock:chairman"}, nil
		},
	}

	orch := NewOrchestrator([]llm.LLMClient{clientFast, clientSlow}, chairman, 5*time.Second)
	eventChan := make(chan StreamEvent, 16)

	start := time.Now()
	var firstDraftTime time.Duration
	var firstDraftModel string
	var mu sync.Mutex

	done := make(chan struct{})
	go func() {
		for ev := range eventChan {
			if ev.Type == EventCandidateDraft {
				mu.Lock()
				if firstDraftTime == 0 {
					firstDraftTime = time.Since(start)
					if payload, ok := ev.Data.(CandidateDraftPayload); ok {
						firstDraftModel = payload.Model
					}
				}
				mu.Unlock()
			}
		}
		close(done)
	}()

	_, err := orch.QueryStream(context.Background(), "TTFT test", nil, "", false, "council", nil, eventChan)
	<-done

	if err != nil {
		t.Fatalf("QueryStream failed: %v", err)
	}

	mu.Lock()
	defer mu.Unlock()

	if firstDraftTime > 100*time.Millisecond {
		t.Errorf("fast draft took %v, expected < 100ms", firstDraftTime)
	}
	if firstDraftModel != "mock:fast" {
		t.Errorf("expected first draft from mock:fast, got %q", firstDraftModel)
	}
}
