package council

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"strings"

	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
)

// ChairmanResult holds the synthesized output from the chairman model.
type ChairmanResult struct {
	Answer     string  `json:"answer"`
	Reasoning  string  `json:"reasoning"`
	Source     string  `json:"source"`
	Confidence float64 `json:"confidence"`
}

// ReflectionResult holds the quality assessment from the reflection loop.
type ReflectionResult struct {
	Quality    string  `json:"quality"`    // "good" | "needs_revision"
	Issues     string  `json:"issues"`     // What's wrong (empty if good)
	Suggestion string  `json:"suggestion"` // How to fix (empty if good)
	Faithful   bool    `json:"faithful"`   // Is the answer grounded in sources?
	Confidence float64 `json:"confidence"` // Revised confidence score
}

const chairmanSystemPrompt = `You are the Chairman of an AI council. Multiple AI models have independently answered a question, and their peers have reviewed and ranked the answers. Your job is to synthesize the best possible final answer.

Your task:
1. Consider all responses and the peer review feedback
2. Identify the strongest and weakest points across all answers
3. Synthesize a single, authoritative final answer
4. Rate your confidence in the final answer (0.0 to 1.0)

Respond ONLY in JSON format:
{
  "answer": "Your synthesized final answer",
  "reasoning": "How you synthesized the answer and which responses contributed most",
  "source": "chairman-synthesis",
  "confidence": 0.85
}`

const reflectionSystemPrompt = `You are a quality assurance agent reviewing an AI-generated answer. Your job is to critically evaluate the answer for accuracy, completeness, and faithfulness to source material.

Evaluate the answer on these criteria:
1. Faithfulness — Is every claim grounded in the provided sources? (If no sources provided, is the answer factually reasonable?)
2. Relevance — Does the answer actually address the question asked?
3. Completeness — Are there important aspects the answer missed?
4. Consistency — Are there internal contradictions?

Respond ONLY in JSON format:
{
  "quality": "good" or "needs_revision",
  "issues": "Description of problems found (empty string if none)",
  "suggestion": "How to improve (empty string if good)",
  "faithful": true/false,
  "confidence": 0.0-1.0
}`

// chairmanSynthesize calls the chairman model to produce a synthesized answer.
func (o *Orchestrator) chairmanSynthesize(
	ctx context.Context,
	question string,
	chunks []string,
	candidates []CandidateAnswer,
	reviews []PeerReview,
) (*ChairmanResult, error) {
	var responsesBlock []string
	for i, c := range candidates {
		responsesBlock = append(responsesBlock, fmt.Sprintf("=== Response %c ===\n%s", 'A'+i, c.Answer))
	}

	var reviewsBlock []string
	for i, r := range reviews {
		if r.Error == "" {
			reviewsBlock = append(reviewsBlock, fmt.Sprintf("--- Reviewer %d ---\n%s", i+1, r.Review))
		}
	}

	reviewsText := "No peer reviews available."
	if len(reviewsBlock) > 0 {
		reviewsText = strings.Join(reviewsBlock, "\n\n")
	}

	// Build context section — only include document excerpts if we have them
	contextSection := ""
	if len(chunks) > 0 {
		contextSection = fmt.Sprintf("\nDocument Excerpts:\n%s\n", strings.Join(chunks, "\n\n---\n\n"))
	}

	userMsg := fmt.Sprintf(`Question: %s
%s
Individual Responses:
%s

Peer Reviews:
%s`,
		question,
		contextSection,
		strings.Join(responsesBlock, "\n\n"),
		reviewsText,
	)

	if ctx == nil {
		ctx = context.Background()
	}
	callCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
	defer cancel()

	resp, err := o.chairmanClient.GenerateChat(callCtx, llm.GenerateOptions{
		Messages: []llm.Message{
			{Role: "system", Content: chairmanSystemPrompt},
			{Role: "user", Content: userMsg},
		},
		ResponseJSON: true,
	})
	if err != nil {
		return nil, fmt.Errorf("chairman generation failed: %w", err)
	}

	result := &ChairmanResult{}
	jsonStr := extractJSON(resp.Answer)
	if err := json.Unmarshal([]byte(jsonStr), result); err != nil {
		log.Printf("[Council] Chairman JSON parse failed, using raw answer: %v", err)
		result = &ChairmanResult{
			Answer:     resp.Answer,
			Reasoning:  "Chairman response was not valid JSON",
			Source:     "chairman:" + o.chairmanClient.ModelName(),
			Confidence: 0.7,
		}
	}
	if result.Source == "" {
		result.Source = "chairman:" + o.chairmanClient.ModelName()
	}

	return result, nil
}

// Reflect evaluates the quality of a council answer. Only called for
// "council_deep" strategy. Returns a quality assessment that may trigger
// a revision pass.
func (o *Orchestrator) Reflect(ctx context.Context, question string, chunks []string, answer string) (*ReflectionResult, error) {
	contextSection := ""
	if len(chunks) > 0 {
		contextSection = fmt.Sprintf("\nSource Material:\n%s\n", strings.Join(chunks, "\n\n---\n\n"))
	}

	userMsg := fmt.Sprintf(`Question: %s
%s
Answer to evaluate:
%s`, question, contextSection, answer)

	reflectCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
	defer cancel()

	resp, err := o.chairmanClient.GenerateChat(reflectCtx, llm.GenerateOptions{
		Messages: []llm.Message{
			{Role: "system", Content: reflectionSystemPrompt},
			{Role: "user", Content: userMsg},
		},
		Temperature:  llm.Float64Ptr(0.2),
		ResponseJSON: true,
		MaxTokens:    500,
	})
	if err != nil {
		return nil, fmt.Errorf("reflection failed: %w", err)
	}

	result := &ReflectionResult{}
	jsonStr := extractJSON(resp.Answer)
	if err := json.Unmarshal([]byte(jsonStr), result); err != nil {
		log.Printf("[Council] Reflection JSON parse failed: %v", err)
		// Default to passing — don't block on reflection failure
		return &ReflectionResult{
			Quality:    "good",
			Faithful:   true,
			Confidence: 0.7,
		}, nil
	}

	return result, nil
}

// reviseAnswer asks the chairman to improve an answer based on reflection feedback.
func (o *Orchestrator) reviseAnswer(ctx context.Context, question string, chunks []string, originalAnswer string, reflection *ReflectionResult) (*ChairmanResult, error) {
	contextSection := ""
	if len(chunks) > 0 {
		contextSection = fmt.Sprintf("\nDocument Excerpts:\n%s\n", strings.Join(chunks, "\n\n---\n\n"))
	}

	userMsg := fmt.Sprintf(`Question: %s
%s
Your previous answer:
%s

Quality review feedback:
- Issues: %s
- Suggestion: %s

Please revise your answer to address the issues identified. Maintain accuracy and faithfulness to sources.`,
		question, contextSection, originalAnswer, reflection.Issues, reflection.Suggestion)

	reviseCtx, cancel := context.WithTimeout(ctx, o.stageTimeout)
	defer cancel()

	resp, err := o.chairmanClient.GenerateChat(reviseCtx, llm.GenerateOptions{
		Messages: []llm.Message{
			{Role: "system", Content: chairmanSystemPrompt},
			{Role: "user", Content: userMsg},
		},
		ResponseJSON: true,
	})
	if err != nil {
		return nil, fmt.Errorf("revision failed: %w", err)
	}

	result := &ChairmanResult{}
	jsonStr := extractJSON(resp.Answer)
	if err := json.Unmarshal([]byte(jsonStr), result); err != nil {
		result = &ChairmanResult{
			Answer:     resp.Answer,
			Reasoning:  "Revised answer (JSON parse failed)",
			Source:     "chairman-revised:" + o.chairmanClient.ModelName(),
			Confidence: 0.75,
		}
	}
	if result.Source == "" {
		result.Source = "chairman-revised:" + o.chairmanClient.ModelName()
	}

	return result, nil
}

// extractJSON pulls a JSON object out of LLM output that may include markdown
// code fences, control characters, and invalid backslash escapes.
func extractJSON(s string) string {
	s = strings.TrimSpace(s)

	if strings.HasPrefix(s, "```json") {
		s = strings.TrimPrefix(s, "```json")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	} else if strings.HasPrefix(s, "```") {
		s = strings.TrimPrefix(s, "```")
		s = strings.TrimSuffix(s, "```")
		s = strings.TrimSpace(s)
	}

	start := strings.Index(s, "{")
	end := strings.LastIndex(s, "}")
	if start >= 0 && end > start {
		s = s[start : end+1]
	}

	// Strip control characters
	var cleaned strings.Builder
	cleaned.Grow(len(s))
	for _, r := range s {
		if r >= 0x20 || r == '\t' || r == '\n' || r == '\r' {
			cleaned.WriteRune(r)
		}
	}
	s = cleaned.String()

	s = sanitizeJSONBackslashes(s)
	return s
}

// sanitizeJSONBackslashes double-escapes backslashes that aren't valid JSON
// escape sequences (handles LaTeX like \Lambda, \theta, etc).
func sanitizeJSONBackslashes(s string) string {
	var result strings.Builder
	result.Grow(len(s))
	for i := 0; i < len(s); i++ {
		if s[i] == '\\' && i+1 < len(s) {
			next := s[i+1]
			if next == '"' || next == '\\' || next == '/' ||
				next == 'b' || next == 'f' || next == 'n' ||
				next == 'r' || next == 't' || next == 'u' {
				result.WriteByte(s[i])
				result.WriteByte(next)
				i++
			} else {
				result.WriteString("\\\\")
			}
		} else {
			result.WriteByte(s[i])
		}
	}
	return result.String()
}
