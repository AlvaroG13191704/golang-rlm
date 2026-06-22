package client

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"sync"
	"time"

	"rlm-golang/internal/prompt"
	"rlm-golang/internal/types"
)

const defaultOllamaBaseURL = "http://localhost:11434/api"

// OllamaMetrics holds performance metrics derived from a single Ollama
// response. Durations are converted from nanosecond integers to time.Duration.
type OllamaMetrics struct {
	TotalDuration      time.Duration
	PromptEvalDuration time.Duration
	EvalDuration       time.Duration
	TTFT               time.Duration
	PromptEvalRate     float64 // input tokens per second
	EvalRate           float64 // output tokens per second
}

// OllamaClient implements BaseLM using the Ollama HTTP API.
type OllamaClient struct {
	model      string
	baseURL    string
	httpClient *http.Client

	mu      sync.Mutex
	summary types.ModelUsageSummary
	last    types.ModelUsageSummary
	metrics OllamaMetrics
}

// NewOllamaClient creates an Ollama client for the given model. If baseURL is
// empty it falls back to the OLLAMA_HOST environment variable, then to the
// default localhost URL. A nil httpClient uses [http.DefaultClient].
func NewOllamaClient(model, baseURL string, httpClient *http.Client) (*OllamaClient, error) {
	if strings.TrimSpace(model) == "" {
		return nil, fmt.Errorf("ollama model is required")
	}

	if httpClient == nil {
		httpClient = http.DefaultClient
	}

	resolved := baseURL
	if resolved == "" {
		resolved = os.Getenv("OLLAMA_HOST")
		if resolved == "" {
			resolved = defaultOllamaBaseURL
		}
	}
	resolved = strings.TrimRight(resolved, "/")

	u, err := url.Parse(resolved)
	if err != nil || (u.Scheme != "http" && u.Scheme != "https") || u.Host == "" {
		return nil, fmt.Errorf("invalid ollama base URL %q: %w", resolved, err)
	}

	return &OllamaClient{
		model:      model,
		baseURL:    resolved,
		httpClient: httpClient,
	}, nil
}

// Completion sends a prompt to Ollama. String prompts use /api/generate;
// message-list prompts use /api/chat. All requests set stream:false so usage
// fields are deterministic.
func (c *OllamaClient) Completion(ctx context.Context, promptArg any) (string, error) {
	var (
		endpoint string
		body     any
		content  string
		usage    ollamaUsage
	)

	switch p := promptArg.(type) {
	case string:
		endpoint = c.baseURL + "/generate"
		body = ollamaGenerateRequest{Model: c.model, Prompt: p, Stream: false}
	default:
		msgs, err := toOllamaMessages(promptArg)
		if err != nil {
			return "", fmt.Errorf("ollama: %w", err)
		}
		endpoint = c.baseURL + "/chat"
		body = ollamaChatRequest{Model: c.model, Messages: msgs, Stream: false}
	}

	payload, err := json.Marshal(body)
	if err != nil {
		return "", fmt.Errorf("ollama: encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(payload))
	if err != nil {
		return "", fmt.Errorf("ollama: build request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("ollama: do request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("ollama: read response: %w", err)
	}

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		preview := string(respBody)
		if len(preview) > 200 {
			preview = preview[:200]
		}
		return "", fmt.Errorf("ollama request failed with status %d: %s", resp.StatusCode, preview)
	}

	if _, ok := body.(ollamaGenerateRequest); ok {
		var gen ollamaGenerateResponse
		if err := json.Unmarshal(respBody, &gen); err != nil {
			return "", fmt.Errorf("ollama: decode generate response: %w", err)
		}
		content = gen.Response
		usage = gen.usage()
	} else {
		var chat ollamaChatResponse
		if err := json.Unmarshal(respBody, &chat); err != nil {
			return "", fmt.Errorf("ollama: decode chat response: %w", err)
		}
		content = chat.Message.Content
		usage = chat.usage()
	}

	c.recordUsage(usage)
	return content, nil
}

// GetUsageSummary returns the aggregate usage across all Completion calls.
func (c *OllamaClient) GetUsageSummary() types.ModelUsageSummary {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.summary
}

// GetLastUsage returns the usage from the most recent Completion call.
func (c *OllamaClient) GetLastUsage() types.ModelUsageSummary {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.last
}

// Metrics returns the derived performance metrics for the most recent
// Completion call.
func (c *OllamaClient) Metrics() OllamaMetrics {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.metrics
}

func (c *OllamaClient) recordUsage(u ollamaUsage) {
	last := types.ModelUsageSummary{
		TotalCalls:        1,
		TotalInputTokens:  u.promptEvalCount,
		TotalOutputTokens: u.evalCount,
	}

	totalDuration := time.Duration(u.totalDuration)
	promptEvalDuration := time.Duration(u.promptEvalDuration)
	evalDuration := time.Duration(u.evalDuration)

	metrics := OllamaMetrics{
		TotalDuration:      totalDuration,
		PromptEvalDuration: promptEvalDuration,
		EvalDuration:       evalDuration,
		TTFT:               promptEvalDuration,
	}
	if promptEvalDuration > 0 {
		metrics.PromptEvalRate = float64(u.promptEvalCount) / promptEvalDuration.Seconds()
	}
	if evalDuration > 0 {
		metrics.EvalRate = float64(u.evalCount) / evalDuration.Seconds()
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	c.last = last
	c.summary = AddModelUsage(c.summary, last)
	c.metrics = metrics
}

type ollamaUsage struct {
	promptEvalCount    int
	evalCount          int
	totalDuration      int64
	promptEvalDuration int64
	evalDuration       int64
}

type ollamaGenerateRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
	Stream bool   `json:"stream"`
}

type ollamaChatRequest struct {
	Model    string          `json:"model"`
	Messages []ollamaMessage `json:"messages"`
	Stream   bool            `json:"stream"`
}

type ollamaMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type ollamaGenerateResponse struct {
	Response           string `json:"response"`
	PromptEvalCount    int    `json:"prompt_eval_count"`
	EvalCount          int    `json:"eval_count"`
	TotalDuration      int64  `json:"total_duration"`
	PromptEvalDuration int64  `json:"prompt_eval_duration"`
	EvalDuration       int64  `json:"eval_duration"`
}

func (r ollamaGenerateResponse) usage() ollamaUsage {
	return ollamaUsage{
		promptEvalCount:    r.PromptEvalCount,
		evalCount:          r.EvalCount,
		totalDuration:      r.TotalDuration,
		promptEvalDuration: r.PromptEvalDuration,
		evalDuration:       r.EvalDuration,
	}
}

type ollamaChatResponse struct {
	Message            ollamaMessage `json:"message"`
	PromptEvalCount    int           `json:"prompt_eval_count"`
	EvalCount          int           `json:"eval_count"`
	TotalDuration      int64         `json:"total_duration"`
	PromptEvalDuration int64         `json:"prompt_eval_duration"`
	EvalDuration       int64         `json:"eval_duration"`
}

func (r ollamaChatResponse) usage() ollamaUsage {
	return ollamaUsage{
		promptEvalCount:    r.PromptEvalCount,
		evalCount:          r.EvalCount,
		totalDuration:      r.TotalDuration,
		promptEvalDuration: r.PromptEvalDuration,
		evalDuration:       r.EvalDuration,
	}
}

func toOllamaMessages(value any) ([]ollamaMessage, error) {
	switch p := value.(type) {
	case []prompt.Message:
		msgs := make([]ollamaMessage, len(p))
		for i, m := range p {
			msgs[i] = ollamaMessage{Role: m.Role, Content: m.Content}
		}
		return msgs, nil
	case []map[string]string:
		msgs := make([]ollamaMessage, len(p))
		for i, m := range p {
			msgs[i] = ollamaMessage{Role: m["role"], Content: m["content"]}
		}
		return msgs, nil
	case []map[string]any:
		msgs := make([]ollamaMessage, len(p))
		for i, m := range p {
			msgs[i] = ollamaMessage{Role: toString(m["role"]), Content: toString(m["content"])}
		}
		return msgs, nil
	case []any:
		msgs := make([]ollamaMessage, 0, len(p))
		for _, item := range p {
			switch v := item.(type) {
			case prompt.Message:
				msgs = append(msgs, ollamaMessage{Role: v.Role, Content: v.Content})
			case map[string]string:
				msgs = append(msgs, ollamaMessage{Role: v["role"], Content: v["content"]})
			case map[string]any:
				msgs = append(msgs, ollamaMessage{Role: toString(v["role"]), Content: toString(v["content"])})
			default:
				return nil, fmt.Errorf("unsupported message element type %T", item)
			}
		}
		return msgs, nil
	default:
		return nil, fmt.Errorf("unsupported prompt type %T", value)
	}
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if v == nil {
		return ""
	}
	return fmt.Sprint(v)
}
