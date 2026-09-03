// Package agent provides intent classification, query routing, and document ingestion planning agents.
package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

// QueryPlan describes the execution strategy chosen by the router agent.
type QueryPlan struct {
	Strategy  string `json:"strategy"`  // "direct" | "council" | "council_deep"
	Reasoning string `json:"reasoning"` // Why this strategy was chosen
	NeedsDoc  bool   `json:"needs_doc"` // Does this query need document context?
}

// Router is a lightweight agent that classifies incoming queries and
// decides the execution strategy before the council runs.
type Router struct {
	client llm.LLMClient
}

// NewRouter creates a router agent backed by the given LLM client
// (typically a fast, cheap model like Gemini Flash).
func NewRouter(client llm.LLMClient) *Router {
	return &Router{client: client}
}

const routerSystemPrompt = `You are a query classifier for a multi-LLM council system.
Given a user question, and optionally a summary of an attached document, choose the best strategy:

- "direct": Simple factual question, greeting, meta-question, or something a single model handles well. One model is enough.
- "council": Needs multiple perspectives, nuanced reasoning, or a thorough answer. Use the full multi-model council.
- "council_deep": Complex, ambiguous, high-stakes, or requires careful deliberation with quality verification.

Also decide if the query genuinely needs document context (needs_doc).
If no document summary is provided, set needs_doc to false regardless.
If a document summary IS provided, evaluate if the user's question is related to the topics in the summary. If the question is completely unrelated (e.g., general knowledge, greetings), set needs_doc to false.

Respond ONLY with a raw JSON object and NO OTHER TEXT whatsoever. Do not include markdown code blocks, do not say 'Here is the JSON'. Just output the raw JSON string: {"strategy":"...", "reasoning":"...", "needs_doc": true/false}`

// Plan analyzes a question and returns an execution plan.
func (r *Router) Plan(ctx context.Context, question string, docSummary string) (*QueryPlan, error) {
	hasDocument := docSummary != ""

	var userMsg string
	if hasDocument {
		userMsg = fmt.Sprintf("Question: %s\nAttached Document Summary:\n%s", question, docSummary)
	} else {
		userMsg = fmt.Sprintf("Question: %s\nNo document attached.", question)
	}

	resp, err := r.client.GenerateChat(ctx, llm.GenerateOptions{
		Messages: []llm.Message{
			{Role: "system", Content: routerSystemPrompt},
			{Role: "user", Content: userMsg},
		},
		Temperature:  llm.Float64Ptr(0.1), // Low temperature for deterministic classification
		ResponseJSON: true,
		MaxTokens:    200,
	})
	if err != nil {
		log.Printf("[Router] Classification failed, defaulting to council: %v", err)
		return defaultPlan(hasDocument), nil
	}

	plan := &QueryPlan{}
	jsonStr := extractRouterJSON(resp.Answer)
	if err := json.Unmarshal([]byte(jsonStr), plan); err != nil {
		log.Printf("[Router] JSON parse failed, defaulting to council: %v (raw: %s)", err, resp.Answer)
		return defaultPlan(hasDocument), nil
	}

	// Validate strategy
	switch plan.Strategy {
	case "direct", "council", "council_deep":
		// valid
	default:
		log.Printf("[Router] Unknown strategy %q, defaulting to council", plan.Strategy)
		plan.Strategy = "council"
	}

	// If no document attached, can't need doc
	if !hasDocument {
		plan.NeedsDoc = false
	}

	log.Printf("[Router] Strategy=%s NeedsDoc=%v Reason=%s", plan.Strategy, plan.NeedsDoc, plan.Reasoning)
	return plan, nil
}

// defaultPlan returns a safe default when the router fails.
func defaultPlan(hasDocument bool) *QueryPlan {
	return &QueryPlan{
		Strategy:  "council",
		Reasoning: "Router unavailable, using default council strategy",
		NeedsDoc:  hasDocument,
	}
}

// extractRouterJSON pulls a JSON object from the router response,
// handling markdown fences and extra text.
func extractRouterJSON(s string) string {
	s = strings.TrimSpace(s)

	// Strip markdown code fences
	if strings.HasPrefix(s, "```json") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	} else if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	}

	// Find JSON object
	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		s = s[start : end+1]
	}
	return s
}
