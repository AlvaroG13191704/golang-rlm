// Package rlm is the public API for the Recursive Language Model runtime.
// It wraps the internal orchestrator and exposes a minimal surface for
// library and CLI consumers.
package rlm

import (
	"context"
	"os"
	"time"

	"rlm-golang/internal/client"
	"rlm-golang/internal/environment"
	internal "rlm-golang/internal/rlm"
	"rlm-golang/internal/types"
)

// ModelUsageSummary aggregates usage counters for a single model.
type ModelUsageSummary struct {
	TotalCalls        int
	TotalInputTokens  int
	TotalOutputTokens int
}

// UsageSummary aggregates per-model usage across the whole session.
type UsageSummary struct {
	ModelUsageSummaries map[string]ModelUsageSummary
}

// CompletionResult captures the result of a single RLM completion.
type CompletionResult struct {
	RootModel     string
	Prompt        any
	Response      string
	UsageSummary  UsageSummary
	ExecutionTime time.Duration
	Metadata      map[string]any
	Error         string
}

// RLM is the public wrapper around the internal recursive language model
// orchestrator.
type RLM struct {
	internal *internal.RLM
}

// Option configures an RLM.
type Option func(*config)

type config struct {
	backend               string
	model                 string
	baseURL               string
	maxDepth              int
	maxIterations         int
	maxBudget             *float64
	maxTimeout            *float64
	maxTokens             *int
	maxErrors             *int
	maxConcurrentSubcalls int
	useDockerREPL         bool
}

// New returns a usable RLM with safe defaults.
//
// Defaults:
//   - backend: ollama
//   - max-depth: 2
//   - max-iterations: 30
//   - max-concurrent-subcalls: 4
//   - max-errors: 3
func New(opts ...Option) (*RLM, error) {
	maxErrorsDefault := 3
	cfg := &config{
		backend:               "ollama",
		maxDepth:              2,
		maxIterations:         30,
		maxConcurrentSubcalls: 4,
		maxErrors:             &maxErrorsDefault,
	}
	for _, opt := range opts {
		opt(cfg)
	}

	backendKwargs := map[string]any{}
	if cfg.model != "" {
		backendKwargs["model_name"] = cfg.model
	}
	if cfg.baseURL != "" {
		backendKwargs["base_url"] = cfg.baseURL
	}

	internalRLM := &internal.RLM{
		Backend:               cfg.backend,
		BackendKwargs:         backendKwargs,
		MaxDepth:              cfg.maxDepth,
		MaxIterations:         cfg.maxIterations,
		MaxBudget:             cfg.maxBudget,
		MaxTimeout:            cfg.maxTimeout,
		MaxTokens:             cfg.maxTokens,
		MaxErrors:             cfg.maxErrors,
		MaxConcurrentSubcalls: cfg.maxConcurrentSubcalls,
	}

	if cfg.useDockerREPL {
		internalRLM.ClientFactory = func(model string) (client.BaseLM, error) {
			baseURL := cfg.baseURL
			if baseURL == "" {
				baseURL = os.Getenv("OLLAMA_HOST")
			}
			if baseURL == "" {
				baseURL = "http://localhost:11434/api"
			}
			return client.NewOllamaClient(model, baseURL, nil)
		}
		internalRLM.EnvFactory = func(host string, port int, depth int, workspace string) (internal.Environment, error) {
			return environment.NewDockerREPL(environment.DockerREPLConfig{
				Image:         "rlm-sandbox",
				LMHandlerHost: host,
				LMHandlerPort: port,
				Depth:         depth,
				MaxDepth:      cfg.maxDepth,
				Workspace:     workspace,
			})
		}
	}

	return &RLM{internal: internalRLM}, nil
}

// Completion runs the iterative RLM loop and returns the model's final answer.
// The prompt is used as both the REPL context and the root question.
func (r *RLM) Completion(ctx context.Context, prompt string) (*CompletionResult, error) {
	return r.CompletionWithContext(ctx, prompt, prompt)
}

// CompletionWithContext runs the iterative RLM loop using the provided context
// as the REPL payload and the prompt as the root question.
func (r *RLM) CompletionWithContext(ctx context.Context, prompt string, context any) (*CompletionResult, error) {
	res, err := r.internal.Completion(ctx, context, prompt)
	if err != nil {
		return nil, err
	}
	return fromInternalCompletion(res), nil
}

func fromInternalCompletion(res *types.RLMChatCompletion) *CompletionResult {
	if res == nil {
		return nil
	}
	return &CompletionResult{
		RootModel:     res.RootModel,
		Prompt:        res.Prompt,
		Response:      res.Response,
		UsageSummary:  fromInternalUsage(res.UsageSummary),
		ExecutionTime: res.ExecutionTime,
		Metadata:      res.Metadata,
		Error:         res.Error,
	}
}

func fromInternalUsage(u types.UsageSummary) UsageSummary {
	models := make(map[string]ModelUsageSummary, len(u.ModelUsageSummaries))
	for name, summary := range u.ModelUsageSummaries {
		models[name] = ModelUsageSummary{
			TotalCalls:        summary.TotalCalls,
			TotalInputTokens:  summary.TotalInputTokens,
			TotalOutputTokens: summary.TotalOutputTokens,
		}
	}
	return UsageSummary{ModelUsageSummaries: models}
}

// WithModel sets the model name passed to the backend.
func WithModel(model string) Option {
	return func(cfg *config) {
		cfg.model = model
	}
}

// WithBackend sets the LM backend. The default is "ollama".
func WithBackend(backend string) Option {
	return func(cfg *config) {
		cfg.backend = backend
	}
}

// WithDockerREPL enables the Docker-based Python REPL environment. This is
// required for the public API to execute model-generated code.
func WithDockerREPL() Option {
	return func(cfg *config) {
		cfg.useDockerREPL = true
	}
}

// WithOllamaHost sets the Ollama base URL.
func WithOllamaHost(host string) Option {
	return func(cfg *config) {
		cfg.baseURL = host
	}
}

// WithMaxDepth sets the maximum recursion depth.
func WithMaxDepth(n int) Option {
	return func(cfg *config) {
		cfg.maxDepth = n
	}
}

// WithMaxIterations sets the maximum number of REPL turns.
func WithMaxIterations(n int) Option {
	return func(cfg *config) {
		cfg.maxIterations = n
	}
}

// WithMaxBudget sets the maximum cost budget in dollars.
func WithMaxBudget(budget float64) Option {
	return func(cfg *config) {
		cfg.maxBudget = &budget
	}
}

// WithMaxTimeout sets the maximum completion timeout in seconds.
func WithMaxTimeout(seconds float64) Option {
	return func(cfg *config) {
		cfg.maxTimeout = &seconds
	}
}

// WithMaxTokens sets the maximum total token budget.
func WithMaxTokens(n int) Option {
	return func(cfg *config) {
		cfg.maxTokens = &n
	}
}

// WithMaxErrors sets the maximum consecutive errors before aborting.
// Pass a value <= 0 to disable error-based aborting.
func WithMaxErrors(n int) Option {
	return func(cfg *config) {
		if n <= 0 {
			cfg.maxErrors = nil
		} else {
			cfg.maxErrors = &n
		}
	}
}
