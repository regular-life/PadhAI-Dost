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

// NIMClient communicates with NVIDIA NIM microservices.
type NIMClient struct {
	apiKey string
	apiURL string
	model  string
	client *http.Client
}

// NewNIMClient constructs a new NIMClient with the specified credentials, base URL, and model identifier.
func NewNIMClient(apiKey, apiURL, model string, timeout time.Duration) *NIMClient {
	return &NIMClient{
		apiKey: apiKey,
		apiURL: apiURL,
		model:  model,
		client: &http.Client{Timeout: timeout},
	}
}

// NIM uses the OpenAI-compatible format — reuses openAIRequest/openAIResponse
// types defined in openrouter.go.

// Generate sends a single prompt — backward-compatible wrapper around GenerateChat.
func (c *NIMClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{
		Messages: []Message{{Role: "user", Content: prompt}},
	})
}

// GenerateChat sends a full multi-turn conversation with system prompt support.
func (c *NIMClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	reqBody := openAIRequest{
		Model: c.model,
	}

	for _, msg := range opts.Messages {
		reqBody.Messages = append(reqBody.Messages, openAIMessage{
			Role:    msg.Role,
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

	resp, err := c.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("NIM request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("NIM API error (status %d): %s", resp.StatusCode, string(body))
	}

	var nimResp openAIResponse
	if err := json.Unmarshal(body, &nimResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(nimResp.Choices) == 0 {
		return nil, fmt.Errorf("no choices in NIM response")
	}

	return &Response{
		Answer: nimResp.Choices[0].Message.Content,
		Model:  c.model,
		Usage: Usage{
			PromptTokens:     nimResp.Usage.PromptTokens,
			CompletionTokens: nimResp.Usage.CompletionTokens,
			TotalTokens:      nimResp.Usage.TotalTokens,
		},
	}, nil
}

// ModelName returns the qualified model identifier prefixed with the nim provider name.
func (c *NIMClient) ModelName() string {
	return "nim:" + c.model
}
