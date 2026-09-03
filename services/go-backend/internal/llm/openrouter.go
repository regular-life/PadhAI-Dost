package llm

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"time"
)

// OpenRouterClient communicates with the OpenRouter API (OpenAI-compatible chat completion endpoints).
type OpenRouterClient struct {
	apiKey string
	apiURL string
	model  string
	client *http.Client
}

// NewOpenRouterClient constructs a new OpenRouterClient with the specified credentials, base URL, and model.
func NewOpenRouterClient(apiKey, apiURL, model string, timeout time.Duration) *OpenRouterClient {
	return &OpenRouterClient{
		apiKey: apiKey,
		apiURL: apiURL,
		model:  model,
		client: &http.Client{Timeout: timeout},
	}
}

// ── OpenAI-compatible request/response types ────────────────────────

type openAIRequest struct {
	Model          string            `json:"model"`
	Messages       []openAIMessage   `json:"messages"`
	Temperature    *float64          `json:"temperature,omitempty"`
	MaxTokens      int               `json:"max_tokens,omitempty"`
	ResponseFormat *openAIRespFormat `json:"response_format,omitempty"`
}

type openAIMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openAIRespFormat struct {
	Type string `json:"type"` // "json_object" or "text"
}

type openAIResponse struct {
	ID      string `json:"id"`
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
}

// ── Interface implementation ────────────────────────────────────────

// Generate sends a single prompt — backward-compatible wrapper around GenerateChat.
func (c *OpenRouterClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{
		Messages: []Message{{Role: "user", Content: prompt}},
	})
}

// GenerateChat sends a full multi-turn conversation with system prompt support.
func (c *OpenRouterClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	reqBody := openAIRequest{
		Model: c.model,
	}

	// Map messages
	for _, msg := range opts.Messages {
		reqBody.Messages = append(reqBody.Messages, openAIMessage{
			Role:    msg.Role, // system, user, assistant all map directly
			Content: msg.Content,
		})
	}

	if opts.Temperature != nil {
		reqBody.Temperature = opts.Temperature
	}
	if opts.MaxTokens > 0 {
		reqBody.MaxTokens = opts.MaxTokens
	}
	if opts.ResponseJSON {
		reqBody.ResponseFormat = &openAIRespFormat{Type: "json_object"}
	}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, c.apiURL, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)
	req.Header.Set("HTTP-Referer", "https://github.com/regular-life/CouncilAI")

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("OpenRouter request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("OpenRouter API error (status %d): %s", resp.StatusCode, string(body))
	}

	var orResp openAIResponse
	if err := json.Unmarshal(body, &orResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(orResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in OpenRouter response")
	}

	return &Response{
		Answer: orResp.Choices[0].Message.Content,
		Model:  c.model,
		Usage: Usage{
			PromptTokens:     orResp.Usage.PromptTokens,
			CompletionTokens: orResp.Usage.CompletionTokens,
			TotalTokens:      orResp.Usage.TotalTokens,
		},
	}, nil
}

// ModelName returns the configured OpenRouter model identifier.
func (c *OpenRouterClient) ModelName() string {
	return c.model
}
