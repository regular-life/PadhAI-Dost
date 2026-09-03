// Package llm defines client interfaces and provider adapters for Gemini, OpenRouter, NVIDIA NIM, and vLLM.
package llm

import "context"

// Message represents a single message in a conversation.
type Message struct {
	Role    string `json:"role"` // "system", "user", "assistant"
	Content string `json:"content"`
}

// GenerateOptions provides full control over LLM generation.
type GenerateOptions struct {
	Messages     []Message // Multi-message conversation (system + user + assistant turns)
	Temperature  *float64  // nil = provider default
	MaxTokens    int       // 0 = provider default
	ResponseJSON bool      // Request JSON output mode if supported by provider
	EnableSearch bool      // Enable live web search grounding if supported by provider
}

// Response from an LLM generation call.
type Response struct {
	Answer     string  `json:"answer"`
	Model      string  `json:"model"`
	Confidence float64 `json:"confidence"`
	Usage      Usage   `json:"usage"`
}

// Usage tracks token consumption for a single LLM call.
type Usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// LLMClient is the interface that all LLM providers must implement.
type LLMClient interface {
	// Generate sends a single prompt and returns a response.
	// Kept for backward compatibility — internally wraps GenerateChat.
	Generate(ctx context.Context, prompt string) (*Response, error)

	// GenerateChat sends a multi-message conversation with full control
	// over system prompts, temperature, and output format.
	GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error)

	// ModelName returns the provider-prefixed model identifier.
	ModelName() string
}

// Float64Ptr is a helper to create a *float64 for GenerateOptions.Temperature.
func Float64Ptr(v float64) *float64 {
	return &v
}
