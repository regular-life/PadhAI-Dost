// Package council coordinates multi-agent consensus deliberation, peer review, and Chairman synthesis.
package council

import (
	"context"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
	"github.com/regular-life/CouncilAI/go-backend/internal/telemetry"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/trace"
)

// StreamEventType represents the classification of a streaming deliberation event.
type StreamEventType string

const (
	// EventCandidateDraft is emitted immediately when an individual council model completes its candidate answer in Stage 1.
	EventCandidateDraft StreamEventType = "candidate_draft"

	// EventPeerReview is emitted when an individual reviewer finishes evaluating candidates in Stage 2.
	EventPeerReview StreamEventType = "peer_review"

	// EventFinalAnswer is emitted when the Chairman completes final consensus synthesis in Stage 3 (or on early fallback).
	EventFinalAnswer StreamEventType = "final_answer"

	// EventError is emitted if a fatal error occurs during deliberation after stream initialization.
	EventError StreamEventType = "error"
)

// StreamEvent represents an asynchronous event frame emitted during deliberation.
type StreamEvent struct {
	Type StreamEventType `json:"type"`
	Data any             `json:"data"`
}

// CandidateDraftPayload contains the individual draft from a single council member in Stage 1.
type CandidateDraftPayload struct {
	Index     int    `json:"index"`
	Model     string `json:"model"`
	ModelName string `json:"model_name,omitempty"`
	Answer    string `json:"answer"`
	Content   string `json:"content,omitempty"`
	LatencyMs int64  `json:"latency_ms,omitempty"`
	Error     string `json:"error,omitempty"`
}

// PeerReviewPayload contains an individual reviewer's evaluation in Stage 2.
type PeerReviewPayload struct {
	Index     int            `json:"index"`
	Reviewer  string         `json:"reviewer"`
	Review    string         `json:"review"`
	Critique  string         `json:"critique,omitempty"`
	Scores    map[string]int `json:"scores,omitempty"`
	Ranking   []string       `json:"ranking,omitempty"`
	LatencyMs int64          `json:"latency_ms,omitempty"`
	Error     string         `json:"error,omitempty"`
}

// ErrorPayload contains error details emitted over the stream.
type ErrorPayload struct {
	Code    int    `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
	Error   string `json:"error,omitempty"`
}

// CouncilResult holds the full output from a council deliberation.
type CouncilResult struct {
	FinalAnswer      string            `json:"final_answer"`
	Answer           string            `json:"answer,omitempty"`
	Reasoning        string            `json:"reasoning,omitempty"`
	Confidence       float64           `json:"confidence"`
	Source           string            `json:"source"`
	Strategy         string            `json:"strategy,omitempty"`
	CandidateAnswers []CandidateAnswer `json:"candidate_answers,omitempty"`
	Candidates       []CandidateAnswer `json:"candidates,omitempty"`
	PeerReviews      []PeerReview      `json:"peer_reviews,omitempty"`
	PeerReviewed     bool              `json:"peer_reviewed"`
	Reflection       *ReflectionResult `json:"reflection,omitempty"`
	Latency          time.Duration     `json:"-"`
	LatencyStr       string            `json:"latency,omitempty"`
}

// CandidateAnswer is a single model's response to the query.
type CandidateAnswer struct {
	Answer  string  `json:"answer"`
	Content string  `json:"content,omitempty"`
	Model   string  `json:"model"`
	Score   float64 `json:"score,omitempty"`
	Error   string  `json:"error,omitempty"`
}

// PeerReview is a single model's evaluation of the other models' responses.
type PeerReview struct {
	Reviewer string `json:"reviewer"`
	Review   string `json:"review"`
	Error    string `json:"error,omitempty"`
}

// ConversationTurn represents a single turn in a conversation for multi-turn context.
type ConversationTurn struct {
	Role    string `json:"role"` // "user" | "assistant"
	Content string `json:"content"`
}

// Orchestrator coordinates the multi-model council deliberation.
type Orchestrator struct {
	clients        []llm.LLMClient
	chairmanClient llm.LLMClient
	stageTimeout   time.Duration
	tracer         telemetry.TracerProvider
}

// NewOrchestrator creates a council orchestrator with N council members and a chairman.
func NewOrchestrator(clients []llm.LLMClient, chairmanClient llm.LLMClient, stageTimeout time.Duration) *Orchestrator {
	if stageTimeout == 0 {
		stageTimeout = 30 * time.Second
	}
	return &Orchestrator{
		clients:        clients,
		chairmanClient: chairmanClient,
		stageTimeout:   stageTimeout,
	}
}

// SetTracer assigns a TracerProvider to the orchestrator.
func (o *Orchestrator) SetTracer(tp telemetry.TracerProvider) {
	o.tracer = tp
}

// emitEvent safely emits an event to the stream channel without blocking if context is done.
func emitEvent(ctx context.Context, events chan<- StreamEvent, event StreamEvent) bool {
	if events == nil {
		return false
	}
	select {
	case <-ctx.Done():
		return false
	case events <- event:
		return true
	}
}

// DeliberateStream is a convenience method for query-only streaming deliberation.
func (o *Orchestrator) DeliberateStream(ctx context.Context, query string, events chan<- StreamEvent) (*CouncilResult, error) {
	return o.QueryStream(ctx, query, nil, "", false, "council", nil, events)
}

// Deliberate is a convenience method for query-only synchronous deliberation.
func (o *Orchestrator) Deliberate(ctx context.Context, query string) (*CouncilResult, error) {
	return o.Query(ctx, query, nil, "", false, "council", nil)
}

// QueryDirect handles the "direct" strategy synchronously (backward compatibility).
func (o *Orchestrator) QueryDirect(ctx context.Context, question string, chunks []string, history []ConversationTurn) (*CouncilResult, error) {
	return o.QueryDirectStream(ctx, question, chunks, history, nil)
}

// QueryDirectStream handles the "direct" strategy with event streaming.
func (o *Orchestrator) QueryDirectStream(ctx context.Context, question string, chunks []string, history []ConversationTurn, events chan<- StreamEvent) (*CouncilResult, error) {
	if events != nil {
		defer close(events)
	}
	start := time.Now()

	if os.Getenv("MOCK_LLM") == "true" {
		time.Sleep(5 * time.Millisecond)
		cand := CandidateAnswer{
			Answer:  "MOCK RESPONSE (direct): " + question,
			Content: "MOCK RESPONSE (direct): " + question,
			Model:   "mock:direct",
		}
		emitEvent(ctx, events, StreamEvent{
			Type: EventCandidateDraft,
			Data: CandidateDraftPayload{
				Index:     0,
				Model:     cand.Model,
				ModelName: cand.Model,
				Answer:    cand.Answer,
				Content:   cand.Answer,
				LatencyMs: time.Since(start).Milliseconds(),
			},
		})

		time.Sleep(5 * time.Millisecond)
		result := &CouncilResult{
			FinalAnswer:      cand.Answer,
			Answer:           cand.Answer,
			Confidence:       0.99,
			Source:           "mock:direct",
			Strategy:         "direct",
			CandidateAnswers: []CandidateAnswer{cand},
			Candidates:       []CandidateAnswer{cand},
			Latency:          time.Since(start),
			LatencyStr:       time.Since(start).String(),
		}
		emitEvent(ctx, events, StreamEvent{
			Type: EventFinalAnswer,
			Data: result,
		})
		return result, nil
	}

	messages := buildMessages(question, chunks, history, "direct")

	directCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
	defer cancel()

	enableSearch := len(chunks) == 0 && o.chairmanClient != nil && strings.HasPrefix(o.chairmanClient.ModelName(), "gemini:")
	if o.chairmanClient == nil {
		errStr := "no chairman client configured for direct query"
		emitEvent(ctx, events, StreamEvent{
			Type: EventError,
			Data: ErrorPayload{Code: 500, Message: errStr, Error: errStr},
		})
		return nil, fmt.Errorf("%s", errStr)
	}

	resp, err := o.chairmanClient.GenerateChat(directCtx, llm.GenerateOptions{
		Messages:     messages,
		EnableSearch: enableSearch,
	})
	if err != nil {
		errStr := fmt.Sprintf("direct query failed: %v", err)
		emitEvent(ctx, events, StreamEvent{
			Type: EventError,
			Data: ErrorPayload{Code: 500, Message: errStr, Error: errStr},
		})
		return nil, fmt.Errorf("direct query failed: %w", err)
	}

	cand := CandidateAnswer{
		Answer:  resp.Answer,
		Content: resp.Answer,
		Model:   o.chairmanClient.ModelName(),
	}
	emitEvent(ctx, events, StreamEvent{
		Type: EventCandidateDraft,
		Data: CandidateDraftPayload{
			Index:     0,
			Model:     cand.Model,
			ModelName: cand.Model,
			Answer:    cand.Answer,
			Content:   cand.Answer,
			LatencyMs: time.Since(start).Milliseconds(),
		},
	})

	result := &CouncilResult{
		FinalAnswer:      resp.Answer,
		Answer:           resp.Answer,
		Confidence:       0.8,
		Source:           "direct:" + o.chairmanClient.ModelName(),
		Strategy:         "direct",
		CandidateAnswers: []CandidateAnswer{cand},
		Candidates:       []CandidateAnswer{cand},
		Latency:          time.Since(start),
		LatencyStr:       time.Since(start).String(),
	}

	emitEvent(ctx, events, StreamEvent{
		Type: EventFinalAnswer,
		Data: result,
	})

	return result, nil
}

// Query runs the full council deliberation pipeline synchronously (backward compatibility).
func (o *Orchestrator) Query(ctx context.Context, question string, chunks []string, customPrompt string, skipChairman bool, strategy string, history []ConversationTurn) (*CouncilResult, error) {
	return o.QueryStream(ctx, question, chunks, customPrompt, skipChairman, strategy, history, nil)
}

// QueryStream runs the full council deliberation pipeline with progressive event streaming.
func (o *Orchestrator) QueryStream(ctx context.Context, question string, chunks []string, customPrompt string, skipChairman bool, strategy string, history []ConversationTurn, events chan<- StreamEvent) (*CouncilResult, error) {
	if events != nil {
		defer close(events)
	}
	start := time.Now()

	if strategy == "" {
		strategy = "council"
	}

	if os.Getenv("MOCK_LLM") == "true" {
		if events != nil {
			mockModels := []string{"mock:claude-3.5-sonnet", "mock:gpt-4o", "mock:gemini-3-flash"}
			mockCandidates := make([]CandidateAnswer, len(mockModels))
			for i, m := range mockModels {
				time.Sleep(5 * time.Millisecond)
				cand := CandidateAnswer{
					Model:   m,
					Answer:  fmt.Sprintf("MOCK CANDIDATE %d (%s) for: %s", i+1, m, question),
					Content: fmt.Sprintf("MOCK CANDIDATE %d (%s) for: %s", i+1, m, question),
				}
				mockCandidates[i] = cand
				emitEvent(ctx, events, StreamEvent{
					Type: EventCandidateDraft,
					Data: CandidateDraftPayload{
						Index:     i,
						Model:     m,
						ModelName: m,
						Answer:    cand.Answer,
						Content:   cand.Answer,
						LatencyMs: int64((i + 1) * 5),
					},
				})
			}

			time.Sleep(5 * time.Millisecond)
			mockReviews := make([]PeerReview, len(mockModels))
			for i, m := range mockModels {
				rev := PeerReview{
					Reviewer: m,
					Review:   "RANKING: B, A, C\nREASONING: Response B is the most complete and accurate.",
				}
				mockReviews[i] = rev
				emitEvent(ctx, events, StreamEvent{
					Type: EventPeerReview,
					Data: PeerReviewPayload{
						Index:     i,
						Reviewer:  m,
						Review:    rev.Review,
						Critique:  "Response B is the most complete and accurate.",
						Ranking:   []string{"B", "A", "C"},
						Scores:    map[string]int{"A": 2, "B": 3, "C": 1},
						LatencyMs: int64((i + 1) * 5),
					},
				})
			}

			time.Sleep(5 * time.Millisecond)
			finalAnswerText := "MOCK RESPONSE: Answer generated locally to preserve API quotas. Original Question: " + question
			result := &CouncilResult{
				FinalAnswer:      finalAnswerText,
				Answer:           finalAnswerText,
				Confidence:       0.99,
				Source:           "mock:council",
				Strategy:         strategy,
				CandidateAnswers: mockCandidates,
				Candidates:       mockCandidates,
				PeerReviews:      mockReviews,
				PeerReviewed:     true,
				Latency:          time.Since(start),
				LatencyStr:       time.Since(start).String(),
			}
			emitEvent(ctx, events, StreamEvent{
				Type: EventFinalAnswer,
				Data: result,
			})
			return result, nil
		}

		time.Sleep(10 * time.Millisecond)
		finalAnswerText := "MOCK RESPONSE: Answer generated locally to preserve API quotas. Original Question: " + question
		return &CouncilResult{
			FinalAnswer:      finalAnswerText,
			Answer:           finalAnswerText,
			Confidence:       0.99,
			Source:           "mock:council",
			Strategy:         strategy,
			CandidateAnswers: []CandidateAnswer{{Answer: finalAnswerText, Content: finalAnswerText, Model: "mock:council"}},
			Candidates:       []CandidateAnswer{{Answer: finalAnswerText, Content: finalAnswerText, Model: "mock:council"}},
			Latency:          time.Since(start),
			LatencyStr:       time.Since(start).String(),
		}, nil
	}

	// Build prompt for council members
	prompt := customPrompt
	if prompt == "" {
		messages := buildMessages(question, chunks, history, "council")
		prompt = flattenMessages(messages)
	}

	// ── Stage 1: Fan-out streaming to all council members ───────────
	log.Printf("[Council] Collecting individual responses from %d models (streaming)", len(o.clients))
	var fanOutSpan trace.Span
	fanOutCtx := ctx
	if o.tracer != nil {
		fanOutCtx, fanOutSpan = o.tracer.StartSpan(ctx, "council.candidate_fan_out",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("council.stage", "candidate_fan_out"),
				attribute.Int("council.num_models", len(o.clients)),
				attribute.String("council.strategy", strategy),
			),
		)
	}
	candidates := o.fanOutStream(fanOutCtx, prompt, events)
	if fanOutSpan != nil {
		validCount := 0
		for _, c := range candidates {
			if c.Error == "" {
				validCount++
			}
		}
		fanOutSpan.SetAttributes(attribute.Int("council.valid_candidates", validCount))
		fanOutSpan.End()
	}

	var valid []CandidateAnswer
	for _, c := range candidates {
		if c.Error == "" {
			valid = append(valid, c)
		}
	}
	if len(valid) == 0 {
		errStr := "all council members failed to respond"
		emitEvent(ctx, events, StreamEvent{
			Type: EventError,
			Data: ErrorPayload{Code: 500, Message: errStr, Error: errStr},
		})
		return nil, fmt.Errorf("%s", errStr)
	}
	if len(valid) == 1 {
		log.Printf("[Council] Only 1 model responded, skipping peer review and chairman")
		result := &CouncilResult{
			FinalAnswer:      valid[0].Answer,
			Answer:           valid[0].Answer,
			Confidence:       0.5,
			Source:           valid[0].Model + " (single-response)",
			Strategy:         strategy,
			CandidateAnswers: candidates,
			Candidates:       candidates,
			Latency:          time.Since(start),
			LatencyStr:       time.Since(start).String(),
		}
		emitEvent(ctx, events, StreamEvent{
			Type: EventFinalAnswer,
			Data: result,
		})
		return result, nil
	}

	// ── Stage 2: Peer Review streaming ──────────────────────────────
	log.Printf("[Council] Peer review with %d valid candidates (streaming)", len(valid))
	reviews := o.peerReviewStream(ctx, question, valid, events)

	successfulReviews := 0
	for _, r := range reviews {
		if r.Error == "" {
			successfulReviews++
		}
	}
	if successfulReviews == 0 {
		log.Printf("[Council] Peer review failed entirely, falling back to best candidate")
		best := pickBestCandidate(valid, nil)
		result := &CouncilResult{
			FinalAnswer:      best.Answer,
			Answer:           best.Answer,
			Confidence:       0.6,
			Source:           best.Model + " (peer-review-failed-fallback)",
			Strategy:         strategy,
			CandidateAnswers: candidates,
			Candidates:       candidates,
			PeerReviews:      reviews,
			PeerReviewed:     false,
			Latency:          time.Since(start),
			LatencyStr:       time.Since(start).String(),
		}
		emitEvent(ctx, events, StreamEvent{
			Type: EventFinalAnswer,
			Data: result,
		})
		return result, nil
	}

	if skipChairman {
		log.Printf("[Council] Skipping chairman (skipChairman=true), using best peer-reviewed candidate")
		best := pickBestCandidate(valid, reviews)
		metrics.CouncilResponseTime.Observe(time.Since(start).Seconds())
		result := &CouncilResult{
			FinalAnswer:      best.Answer,
			Answer:           best.Answer,
			Confidence:       0.75,
			Source:           best.Model + " (peer-reviewed, no chairman)",
			Strategy:         strategy,
			CandidateAnswers: candidates,
			Candidates:       candidates,
			PeerReviews:      reviews,
			PeerReviewed:     true,
			Latency:          time.Since(start),
			LatencyStr:       time.Since(start).String(),
		}
		emitEvent(ctx, events, StreamEvent{
			Type: EventFinalAnswer,
			Data: result,
		})
		return result, nil
	}

	// ── Stage 3: Chairman Synthesis ─────────────────────────────────
	log.Printf("[Council] Chairman synthesis")
	var chairmanSpan trace.Span
	chairmanCtx := ctx
	if o.tracer != nil {
		chairmanCtx, chairmanSpan = o.tracer.StartSpan(ctx, "council.chairman_deliberation",
			trace.WithSpanKind(trace.SpanKindInternal),
			trace.WithAttributes(
				attribute.String("council.stage", "chairman"),
				attribute.String("llm.model", o.chairmanClient.ModelName()),
			),
		)
	}
	chairmanResult, err := o.chairmanSynthesize(chairmanCtx, question, chunks, valid, reviews)
	if chairmanSpan != nil {
		if err != nil {
			chairmanSpan.RecordError(err)
			chairmanSpan.SetStatus(codes.Error, err.Error())
		} else {
			chairmanSpan.SetStatus(codes.Ok, "")
		}
		chairmanSpan.End()
	}
	if err != nil {
		log.Printf("[Council] Chairman synthesis failed: %v, falling back to best candidate", err)
		best := pickBestCandidate(valid, reviews)
		result := &CouncilResult{
			FinalAnswer:      best.Answer,
			Answer:           best.Answer,
			Confidence:       0.65,
			Source:           best.Model + " (chairman-failed-fallback)",
			Strategy:         strategy,
			CandidateAnswers: candidates,
			Candidates:       candidates,
			PeerReviews:      reviews,
			PeerReviewed:     true,
			Latency:          time.Since(start),
			LatencyStr:       time.Since(start).String(),
		}
		emitEvent(ctx, events, StreamEvent{
			Type: EventFinalAnswer,
			Data: result,
		})
		return result, nil
	}

	result := &CouncilResult{
		FinalAnswer:      chairmanResult.Answer,
		Answer:           chairmanResult.Answer,
		Reasoning:        chairmanResult.Reasoning,
		Confidence:       chairmanResult.Confidence,
		Source:           chairmanResult.Source,
		Strategy:         strategy,
		CandidateAnswers: candidates,
		Candidates:       candidates,
		PeerReviews:      reviews,
		PeerReviewed:     len(reviews) > 0,
		Latency:          time.Since(start),
		LatencyStr:       time.Since(start).String(),
	}

	// ── Stage 4 (optional): Reflection — only for council_deep ──────
	if strategy == "council_deep" {
		log.Printf("[Council] Running reflection loop (council_deep strategy)")
		reflection, err := o.Reflect(ctx, question, chunks, chairmanResult.Answer)
		if err != nil {
			log.Printf("[Council] Reflection failed: %v, keeping original answer", err)
		} else {
			result.Reflection = reflection
			log.Printf("[Council] Reflection: quality=%s faithful=%v confidence=%.2f",
				reflection.Quality, reflection.Faithful, reflection.Confidence)

			if reflection.Quality == "needs_revision" {
				log.Printf("[Council] Answer needs revision, running one more chairman pass")
				revised, err := o.reviseAnswer(ctx, question, chunks, chairmanResult.Answer, reflection)
				if err != nil {
					log.Printf("[Council] Revision failed: %v, keeping original answer", err)
				} else {
					result.FinalAnswer = revised.Answer
					result.Answer = revised.Answer
					result.Reasoning = revised.Reasoning
					result.Confidence = revised.Confidence
					result.Source = revised.Source
				}
			} else {
				if reflection.Confidence > 0 {
					result.Confidence = reflection.Confidence
				}
			}
		}
	}

	metrics.CouncilResponseTime.Observe(time.Since(start).Seconds())
	result.Latency = time.Since(start)
	result.LatencyStr = time.Since(start).String()

	emitEvent(ctx, events, StreamEvent{
		Type: EventFinalAnswer,
		Data: result,
	})

	return result, nil
}

// fanOutStream runs Stage 1 in parallel and streams CandidateDraft events progressively.
func (o *Orchestrator) fanOutStream(ctx context.Context, prompt string, events chan<- StreamEvent) []CandidateAnswer {
	var wg sync.WaitGroup
	results := make([]CandidateAnswer, len(o.clients))

	for i, client := range o.clients {
		wg.Add(1)
		go func(idx int, c llm.LLMClient) {
			defer wg.Done()
			modelStart := time.Now()

			var modelSpan trace.Span
			modelCtx := ctx
			if o.tracer != nil {
				modelCtx, modelSpan = o.tracer.StartSpan(ctx, "council.candidate_model",
					trace.WithSpanKind(trace.SpanKindClient),
					trace.WithAttributes(
						attribute.String("llm.model", c.ModelName()),
						attribute.Int("model.index", idx),
					),
				)
				defer modelSpan.End()
			}

			subCtx, cancel := context.WithTimeout(modelCtx, o.stageTimeout)
			defer cancel()

			enableSearch := strings.HasPrefix(c.ModelName(), "gemini:") && !strings.Contains(prompt, "Document Excerpts:")
			resp, err := c.GenerateChat(subCtx, llm.GenerateOptions{
				Messages:     []llm.Message{{Role: "user", Content: prompt}},
				EnableSearch: enableSearch,
			})
			latency := time.Since(modelStart)

			if err != nil {
				if modelSpan != nil {
					modelSpan.RecordError(err)
					modelSpan.SetStatus(codes.Error, err.Error())
				}
				log.Printf("[Council] Model %s failed: %v", c.ModelName(), err)
				metrics.LLMFailureCount.Inc()
				results[idx] = CandidateAnswer{Model: c.ModelName(), Error: err.Error()}
				emitEvent(ctx, events, StreamEvent{
					Type: EventCandidateDraft,
					Data: CandidateDraftPayload{
						Index:     idx,
						Model:     c.ModelName(),
						ModelName: c.ModelName(),
						LatencyMs: latency.Milliseconds(),
						Error:     err.Error(),
					},
				})
				return
			}

			if modelSpan != nil {
				modelSpan.SetAttributes(
					attribute.Int64("latency_ms", latency.Milliseconds()),
					attribute.Int("response.length", len(resp.Answer)),
					attribute.Int("prompt_tokens", resp.Usage.PromptTokens),
					attribute.Int("completion_tokens", resp.Usage.CompletionTokens),
				)
				modelSpan.SetStatus(codes.Ok, "")
			}

			cand := CandidateAnswer{
				Answer:  resp.Answer,
				Content: resp.Answer,
				Model:   resp.Model,
			}
			results[idx] = cand

			// Emit candidate draft immediately to unlock < 1.5s TTFT
			emitEvent(ctx, events, StreamEvent{
				Type: EventCandidateDraft,
				Data: CandidateDraftPayload{
					Index:     idx,
					Model:     cand.Model,
					ModelName: cand.Model,
					Answer:    cand.Answer,
					Content:   cand.Answer,
					LatencyMs: latency.Milliseconds(),
				},
			})
		}(i, client)
	}

	wg.Wait()
	return results
}

// peerReviewStream runs Stage 2 in parallel and streams PeerReview events progressively.
func (o *Orchestrator) peerReviewStream(ctx context.Context, question string, candidates []CandidateAnswer, events chan<- StreamEvent) []PeerReview {
	var wg sync.WaitGroup
	reviews := make([]PeerReview, len(o.clients))

	var anonymized []string
	for i, c := range candidates {
		anonymized = append(anonymized, fmt.Sprintf("=== Response %c ===\n%s", 'A'+i, c.Answer))
	}
	answersBlock := strings.Join(anonymized, "\n\n")

	for i, client := range o.clients {
		wg.Add(1)
		go func(idx int, c llm.LLMClient) {
			defer wg.Done()
			reviewStart := time.Now()

			prompt := fmt.Sprintf(`You are reviewing multiple AI-generated answers to this question:

Question: %s

Here are the anonymized responses:

%s

Your task:
1. Evaluate each response for accuracy, completeness, and clarity
2. Rank them from best to worst
3. Explain briefly why you ranked them that way

Format your response as:
RANKING: [best to worst, e.g., "B, A, C"]
REASONING: [1-2 sentence explanation of your ranking]`, question, answersBlock)

			subCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
			defer cancel()

			resp, err := c.Generate(subCtx, prompt)
			latency := time.Since(reviewStart)

			if err != nil {
				log.Printf("[Council] Reviewer %s failed: %v", c.ModelName(), err)
				reviews[idx] = PeerReview{Reviewer: c.ModelName(), Error: err.Error()}
				emitEvent(ctx, events, StreamEvent{
					Type: EventPeerReview,
					Data: PeerReviewPayload{
						Index:     idx,
						Reviewer:  c.ModelName(),
						LatencyMs: latency.Milliseconds(),
						Error:     err.Error(),
					},
				})
				return
			}

			reviews[idx] = PeerReview{Reviewer: c.ModelName(), Review: resp.Answer}

			ranking, scores := parseReviewRanking(resp.Answer, len(candidates))

			emitEvent(ctx, events, StreamEvent{
				Type: EventPeerReview,
				Data: PeerReviewPayload{
					Index:     idx,
					Reviewer:  c.ModelName(),
					Review:    resp.Answer,
					Critique:  extractReasoning(resp.Answer),
					Ranking:   ranking,
					Scores:    scores,
					LatencyMs: latency.Milliseconds(),
				},
			})
		}(i, client)
	}

	wg.Wait()
	return reviews
}

// ── Candidate selection ─────────────────────────────────────────────

// rankingPattern matches "RANKING: B, A, C" or "RANKING: [B, A, C]"
var rankingPattern = regexp.MustCompile(`(?i)RANKING:\s*\[?\s*([A-Z](?:\s*,\s*[A-Z])*)\s*\]?`)

// extractReasoning extracts the REASONING section from a peer review string.
func extractReasoning(review string) string {
	lines := strings.Split(review, "\n")
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(trimmed), "REASONING:") {
			return strings.TrimSpace(trimmed[len("REASONING:"):])
		}
	}
	return review
}

// parseReviewRanking extracts rankings and scores from a review text.
func parseReviewRanking(review string, numCandidates int) ([]string, map[string]int) {
	matches := rankingPattern.FindStringSubmatch(review)
	if len(matches) < 2 {
		return nil, nil
	}
	rankStr := strings.ReplaceAll(matches[1], " ", "")
	letters := strings.Split(rankStr, ",")
	scores := make(map[string]int)
	var ranking []string
	for rank, letter := range letters {
		if len(letter) != 1 {
			continue
		}
		upper := strings.ToUpper(letter)
		ranking = append(ranking, upper)
		idx := int(upper[0] - 'A')
		if idx >= 0 && idx < numCandidates {
			scores[upper] = numCandidates - rank
		}
	}
	return ranking, scores
}

// pickBestCandidate selects the best candidate based on peer review rankings.
func pickBestCandidate(candidates []CandidateAnswer, reviews []PeerReview) CandidateAnswer {
	if len(candidates) == 0 {
		return CandidateAnswer{}
	}

	if len(reviews) == 0 {
		return pickLongest(candidates)
	}

	// Score each candidate based on peer review rankings
	scores := make(map[int]int) // candidate index → aggregate score
	parsedAny := false

	for _, review := range reviews {
		if review.Error != "" {
			continue
		}

		matches := rankingPattern.FindStringSubmatch(review.Review)
		if len(matches) < 2 {
			continue
		}

		rankStr := strings.ReplaceAll(matches[1], " ", "")
		letters := strings.Split(rankStr, ",")

		n := len(candidates)
		for rank, letter := range letters {
			if len(letter) != 1 {
				continue
			}
			idx := int(strings.ToUpper(letter)[0] - 'A')
			if idx >= 0 && idx < n {
				// Points: 1st place gets N points, 2nd gets N-1, etc.
				scores[idx] += n - rank
				parsedAny = true
			}
		}
	}

	if !parsedAny {
		log.Printf("[Council] Could not parse any peer review rankings, falling back to longest")
		return pickLongest(candidates)
	}

	// Find candidate with highest score
	bestIdx := 0
	bestScore := -1
	for idx, score := range scores {
		if score > bestScore {
			bestScore = score
			bestIdx = idx
		}
	}

	log.Printf("[Council] Peer review scores: %v → selected %c (%s)", scores, 'A'+bestIdx, candidates[bestIdx].Model)
	return candidates[bestIdx]
}

// pickLongest selects the candidate with the longest answer (fallback).
func pickLongest(candidates []CandidateAnswer) CandidateAnswer {
	best := candidates[0]
	for _, c := range candidates[1:] {
		if len(c.Answer) > len(best.Answer) {
			best = c
		}
	}
	return best
}

// ── Prompt building ─────────────────────────────────────────────────

// buildMessages constructs the LLM message list based on context.
func buildMessages(question string, chunks []string, history []ConversationTurn, strategy string) []llm.Message {
	var messages []llm.Message

	if len(chunks) > 0 {
		messages = append(messages, llm.Message{
			Role: "system",
			Content: `You are a knowledgeable AI assistant. Answer the question based on the provided document excerpts. 
If the answer is not in the excerpts, say so clearly. Be concise but thorough. Cite relevant parts of the source material when possible.`,
		})
	} else {
		messages = append(messages, llm.Message{
			Role: "system",
			Content: `You are a knowledgeable AI assistant. Answer the question accurately and thoroughly using your knowledge. 
Be honest when uncertain. Provide reasoning and cite your sources of knowledge when relevant.`,
		})
	}

	for _, turn := range history {
		messages = append(messages, llm.Message{
			Role:    turn.Role,
			Content: turn.Content,
		})
	}

	userContent := ""
	if len(chunks) > 0 {
		contextText := strings.Join(chunks, "\n\n---\n\n")
		userContent = fmt.Sprintf("Document Excerpts:\n%s\n\nQuestion: %s", contextText, question)
	} else {
		userContent = question
	}

	messages = append(messages, llm.Message{
		Role:    "user",
		Content: userContent,
	})

	return messages
}

// flattenMessages converts a message list into a single prompt string.
func flattenMessages(messages []llm.Message) string {
	var parts []string
	for _, msg := range messages {
		switch msg.Role {
		case "system":
			parts = append(parts, "Instructions:\n"+msg.Content)
		case "user":
			parts = append(parts, msg.Content)
		case "assistant":
			parts = append(parts, "Previous response:\n"+msg.Content)
		}
	}
	return strings.Join(parts, "\n\n")
}
