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

// GeminiClient communicates with the Google Gemini REST API.
type GeminiClient struct {
	apiKey string
	model  string
	client *http.Client
}

// NewGeminiClient constructs a new GeminiClient with the specified API key, model name, and HTTP timeout.
func NewGeminiClient(apiKey, model string, timeout time.Duration) *GeminiClient {
	return &GeminiClient{
		apiKey: apiKey,
		model:  model,
		client: &http.Client{Timeout: timeout},
	}
}

// ── Gemini API types ────────────────────────────────────────────────

type geminiRequest struct {
	Contents          []geminiContent         `json:"contents"`
	SystemInstruction *geminiContent          `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenerationConfig `json:"generationConfig,omitempty"`
	Tools             []geminiTool            `json:"tools,omitempty"`
}

type geminiTool struct {
	GoogleSearch *struct{} `json:"googleSearch,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenerationConfig struct {
	Temperature      *float64 `json:"temperature,omitempty"`
	MaxOutputTokens  int      `json:"maxOutputTokens,omitempty"`
	ResponseMimeType string   `json:"responseMimeType,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
		FinishReason string `json:"finishReason"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
}

// ── Interface implementation ────────────────────────────────────────

// Generate sends a single prompt — backward-compatible wrapper around GenerateChat.
func (c *GeminiClient) Generate(ctx context.Context, prompt string) (*Response, error) {
	return c.GenerateChat(ctx, GenerateOptions{
		Messages: []Message{{Role: "user", Content: prompt}},
	})
}

// GenerateChat sends a full multi-turn conversation with system prompt support.
func (c *GeminiClient) GenerateChat(ctx context.Context, opts GenerateOptions) (*Response, error) {
	req := geminiRequest{}

	// Build generation config
	genCfg := &geminiGenerationConfig{}
	hasGenCfg := false
	if opts.Temperature != nil {
		genCfg.Temperature = opts.Temperature
		hasGenCfg = true
	}
	if opts.MaxTokens > 0 {
		genCfg.MaxOutputTokens = opts.MaxTokens
		hasGenCfg = true
	}
	if opts.ResponseJSON {
		genCfg.ResponseMimeType = "application/json"
		hasGenCfg = true
	}
	if hasGenCfg {
		req.GenerationConfig = genCfg
	}

	if opts.EnableSearch {
		req.Tools = []geminiTool{{GoogleSearch: &struct{}{}}}
	}

	// Map messages to Gemini format
	for _, msg := range opts.Messages {
		switch msg.Role {
		case "system":
			// Gemini uses a dedicated systemInstruction field
			req.SystemInstruction = &geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
			}
		case "user":
			req.Contents = append(req.Contents, geminiContent{
				Role:  "user",
				Parts: []geminiPart{{Text: msg.Content}},
			})
		case "assistant":
			// Gemini calls this "model"
			req.Contents = append(req.Contents, geminiContent{
				Role:  "model",
				Parts: []geminiPart{{Text: msg.Content}},
			})
		}
	}

	// Must have at least one content entry
	if len(req.Contents) == 0 {
		return nil, fmt.Errorf("at least one user or assistant message is required")
	}

	jsonBody, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal request: %w", err)
	}

	url := fmt.Sprintf(
		"https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s",
		c.model, c.apiKey,
	)

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(jsonBody))
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")

	resp, err := c.client.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("Gemini request failed: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("failed to read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("Gemini API error (status %d): %s", resp.StatusCode, string(body))
	}

	var gResp geminiResponse
	if err := json.Unmarshal(body, &gResp); err != nil {
		return nil, fmt.Errorf("failed to parse response: %w", err)
	}

	if len(gResp.Candidates) == 0 || len(gResp.Candidates[0].Content.Parts) == 0 {
		return nil, fmt.Errorf("no content in Gemini response")
	}

	return &Response{
		Answer: gResp.Candidates[0].Content.Parts[0].Text,
		Model:  c.model,
		Usage: Usage{
			PromptTokens:     gResp.UsageMetadata.PromptTokenCount,
			CompletionTokens: gResp.UsageMetadata.CandidatesTokenCount,
			TotalTokens:      gResp.UsageMetadata.TotalTokenCount,
		},
	}, nil
}

// ModelName returns the qualified model identifier prefixed with the provider name.
func (c *GeminiClient) ModelName() string {
	return "gemini:" + c.model
}
