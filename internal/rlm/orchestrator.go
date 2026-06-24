package rlm

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"net"
	"os"
	"strconv"
	"strings"
	"sync"
	"time"

	"rlm-golang/internal/client"
	"rlm-golang/internal/prompt"
	"rlm-golang/internal/server"
	"rlm-golang/internal/types"
)

const defaultMaxDepth = 2
const defaultMaxIterations = 30
const defaultMaxConcurrentSubcalls = 4

// RLM runs the recursive prompt → code → execute loop and spawns child RLMs for
// recursive sub-calls.
type RLM struct {
	Backend               string
	BackendKwargs         map[string]any
	Depth                 int
	MaxDepth              int
	MaxIterations         int
	MaxBudget             *float64
	MaxTimeout            *float64
	MaxTokens             *int
	MaxErrors             *int
	MaxConcurrentSubcalls int

	// ClientFactory creates BaseLM instances. In production this builds the
	// configured backend client; tests override it with mocks.
	ClientFactory func(model string) (client.BaseLM, error)
	// EnvFactory creates the execution environment. In production this builds a
	// DockerREPL; tests override it with mocks.
	EnvFactory func(host string, port int, depth int, workspace string) (Environment, error)

	mu                  sync.Mutex
	cumulativeCost      float64
	consecutiveErrors   int
	bestPartialAnswer   string
	completionStartTime time.Time
}

// NewRLM returns an RLM with safe defaults.
func NewRLM() *RLM {
	maxErrorsDefault := 3
	return &RLM{
		MaxDepth:              defaultMaxDepth,
		MaxIterations:         defaultMaxIterations,
		MaxConcurrentSubcalls: defaultMaxConcurrentSubcalls,
		MaxErrors:             &maxErrorsDefault,
	}
}

// Completion runs the iterative RLM loop and returns the model's final answer.
func (r *RLM) Completion(ctx context.Context, promptPayload any, rootPrompt string) (*types.RLMChatCompletion, error) {
	if r.MaxDepth <= 0 {
		r.MaxDepth = defaultMaxDepth
	}
	if r.MaxIterations <= 0 {
		r.MaxIterations = defaultMaxIterations
	}
	if r.MaxConcurrentSubcalls <= 0 {
		r.MaxConcurrentSubcalls = defaultMaxConcurrentSubcalls
	}

	slog.Info("RLM Completion started", "depth", r.Depth, "max_depth", r.MaxDepth, "max_iterations", r.MaxIterations, "model", r.rootModel())
	slog.Debug("RLM Completion prompt", "root_prompt", truncateString(rootPrompt, 200), "payload", truncateString(fmt.Sprintf("%v", promptPayload), 200))

	if r.Depth >= r.MaxDepth {
		slog.Info("RLM Completion falling back to plain LM at max depth", "depth", r.Depth)
		return r.fallbackAnswer(ctx, promptPayload)
	}

	r.resetTracking()

	defaultClient, err := r.newClient(r.rootModel())
	if err != nil {
		return nil, fmt.Errorf("create default client: %w", err)
	}

	lmHandler := NewLMHandler(defaultClient, WithDefaultModel(r.rootModel()))

	host, port, stopServer, err := r.startServices(lmHandler)
	if err != nil {
		return nil, fmt.Errorf("start services: %w", err)
	}
	defer stopServer()

	workspace, err := os.MkdirTemp("", "rlm-workspace-")
	if err != nil {
		return nil, fmt.Errorf("create workspace: %w", err)
	}

	env, err := r.newEnvironment(host, port, r.Depth+1, workspace)
	if err != nil {
		_ = os.RemoveAll(workspace)
		return nil, fmt.Errorf("create environment: %w", err)
	}
	defer func() {
		_ = env.Cleanup(ctx)
	}()

	if err := env.LoadContext(ctx, promptPayload); err != nil {
		return nil, fmt.Errorf("load context: %w", err)
	}

	meta, err := prompt.NewQueryMetadata(promptPayload)
	if err != nil {
		return nil, fmt.Errorf("query metadata: %w", err)
	}

	messageHistory, err := r.buildInitialMessages(meta, rootPrompt)
	if err != nil {
		return nil, fmt.Errorf("build initial messages: %w", err)
	}

	start := time.Now()
	r.setStartTime(start)

	contextCount := 1
	if meta.ContextType == "list" || meta.ContextType == "dict" {
		contextCount = len(meta.ContextLengths)
	}

	for i := 0; i < r.MaxIterations; i++ {
		slog.Info("RLM iteration", "iteration", i+1, "max_iterations", r.MaxIterations, "depth", r.Depth)

		if err := r.checkTimeout(i); err != nil {
			slog.Error("RLM timeout exceeded", "iteration", i+1, "error", err)
			return nil, err
		}

		messageHistory = append(messageHistory, prompt.BuildUserPrompt(rootPrompt, i, contextCount, 0, r.MaxIterations))

		iter, err := r.completionTurn(ctx, messageHistory, defaultClient, env)
		if err != nil {
			slog.Error("RLM completion turn failed", "iteration", i+1, "error", err)
			return nil, err
		}

		if err := r.checkIterationLimits(iter, i, lmHandler); err != nil {
			slog.Error("RLM iteration limit exceeded", "iteration", i+1, "error", err)
			return nil, err
		}

		if iter.FinalAnswer != "" {
			slog.Info("RLM final answer produced", "iteration", i+1, "depth", r.Depth, "answer_len", len(iter.FinalAnswer))
			return r.buildCompletion(iter.FinalAnswer, promptPayload, lmHandler, start), nil
		}

		slog.Debug("RLM iteration produced no final answer", "iteration", i+1, "response_len", len(iter.Response), "code_blocks", len(iter.CodeBlocks))
		r.updateBestPartial(iter.Response)
		messageHistory = append(messageHistory, FormatIteration(iter, defaultMaxIterationChars)...)
	}

	slog.Info("RLM max iterations reached, generating default answer", "max_iterations", r.MaxIterations, "depth", r.Depth)
	finalAnswer, err := r.defaultAnswer(ctx, messageHistory, defaultClient)
	if err != nil {
		slog.Error("RLM default answer failed", "error", err)
		return nil, err
	}
	completion := r.buildCompletion(finalAnswer, promptPayload, lmHandler, start)
	slog.Info("RLM Completion finished", "depth", r.Depth, "answer_len", len(completion.Response), "execution_time", completion.ExecutionTime)
	return completion, nil
}

// Subcall implements server.SubcallHandler. It is invoked when code running
// inside a container calls rlm_query over gRPC. The host spawns a child RLM in
// a sibling container sharing the workspace, unless recursion depth has been
// exhausted.
func (r *RLM) Subcall(ctx context.Context, subcallPrompt any, model string, depth int32) (types.RLMChatCompletion, error) {
	childDepth := int(depth)
	resolvedModel := r.resolveModel(model)

	slog.Info("RLM Subcall received", "child_depth", childDepth, "max_depth", r.MaxDepth, "model", resolvedModel)
	slog.Debug("RLM Subcall prompt", "prompt", truncateString(subcallPrompt.(string), 200))

	// At or past the recursion cap, fall back to a plain LM completion.
	if childDepth >= r.MaxDepth {
		slog.Info("RLM Subcall falling back to LM at max depth", "child_depth", childDepth)
		return r.fallbackLMCall(ctx, subcallPrompt, resolvedModel)
	}

	r.mu.Lock()
	remainingBudget := r.maxBudgetRemainingLocked()
	remainingTimeout := r.maxTimeoutRemainingLocked()
	r.mu.Unlock()

	if remainingBudget != nil && *remainingBudget <= 0 {
		slog.Warn("RLM Subcall budget exhausted", "spent", r.cumulativeCost, "budget", *r.MaxBudget)
		return types.RLMChatCompletion{
			RootModel: resolvedModel,
			Prompt:    subcallPrompt,
			Response:  fmt.Sprintf("Error: Budget exhausted (spent $%.6f of $%.6f)", r.cumulativeCost, *r.MaxBudget),
		}, nil
	}
	if remainingTimeout != nil && *remainingTimeout <= 0 {
		slog.Warn("RLM Subcall timeout exhausted", "elapsed", time.Since(r.completionStartTime).Seconds(), "limit", *r.MaxTimeout)
		return types.RLMChatCompletion{
			RootModel: resolvedModel,
			Prompt:    subcallPrompt,
			Response:  fmt.Sprintf("Error: Timeout exhausted (%.1fs of %.1fs)", time.Since(r.completionStartTime).Seconds(), *r.MaxTimeout),
		}, nil
	}

	child := &RLM{
		Backend:               r.Backend,
		BackendKwargs:         r.resolveBackendKwargs(resolvedModel),
		Depth:                 childDepth,
		MaxDepth:              r.MaxDepth,
		MaxIterations:         r.MaxIterations,
		MaxBudget:             remainingBudget,
		MaxTimeout:            remainingTimeout,
		MaxTokens:             r.MaxTokens,
		MaxErrors:             r.MaxErrors,
		MaxConcurrentSubcalls: r.MaxConcurrentSubcalls,
		ClientFactory:         r.ClientFactory,
		EnvFactory:            r.EnvFactory,
	}

	subcallStart := time.Now()
	slog.Debug("RLM Subcall spawning child", "child_depth", childDepth, "model", resolvedModel)
	result, err := child.Completion(ctx, subcallPrompt, "")
	if result != nil && result.UsageSummary.TotalCost() != nil {
		r.mu.Lock()
		r.cumulativeCost += *result.UsageSummary.TotalCost()
		r.mu.Unlock()
	}

	if err != nil {
		execTime := time.Since(subcallStart)
		var limitErr *types.LimitError
		if errors.As(err, &limitErr) {
			slog.Warn("RLM Subcall child hit limit", "child_depth", childDepth, "error", err, "execution_time", execTime)
			return types.RLMChatCompletion{
				RootModel:     resolvedModel,
				Prompt:        subcallPrompt,
				Response:      fmt.Sprintf("Error: Child RLM Budget exceeded - %v", err),
				ExecutionTime: execTime,
			}, nil
		}
		slog.Error("RLM Subcall child failed", "child_depth", childDepth, "error", err, "execution_time", execTime)
		return types.RLMChatCompletion{
			RootModel:     resolvedModel,
			Prompt:        subcallPrompt,
			Response:      fmt.Sprintf("Error: Child RLM completion failed - %v", err),
			ExecutionTime: execTime,
		}, nil
	}

	slog.Info("RLM Subcall child completed", "child_depth", childDepth, "answer_len", len(result.Response), "execution_time", time.Since(subcallStart))
	return *result, nil
}

func (r *RLM) rootModel() string {
	if r.BackendKwargs != nil {
		if m, ok := r.BackendKwargs["model_name"].(string); ok && m != "" {
			return m
		}
	}
	return "unknown"
}

func (r *RLM) resolveModel(model string) string {
	if model != "" {
		return model
	}
	return r.rootModel()
}

func (r *RLM) resolveBackendKwargs(model string) map[string]any {
	kwargs := make(map[string]any, len(r.BackendKwargs))
	for k, v := range r.BackendKwargs {
		kwargs[k] = v
	}
	if model != "" {
		kwargs["model_name"] = model
	}
	return kwargs
}

func (r *RLM) newClient(model string) (client.BaseLM, error) {
	if r.ClientFactory != nil {
		return r.ClientFactory(model)
	}
	if r.Backend == "ollama" || r.Backend == "" {
		baseURL := ""
		if r.BackendKwargs != nil {
			if u, ok := r.BackendKwargs["base_url"].(string); ok {
				baseURL = u
			}
		}
		return client.NewOllamaClient(model, baseURL, nil)
	}
	return nil, fmt.Errorf("unsupported backend %q and no ClientFactory configured", r.Backend)
}

func (r *RLM) newEnvironment(host string, port int, depth int, workspace string) (Environment, error) {
	if r.EnvFactory != nil {
		return r.EnvFactory(host, port, depth, workspace)
	}
	return nil, fmt.Errorf("no EnvFactory configured")
}

func (r *RLM) startServices(lmHandler *LMHandler) (string, int, func(), error) {
	rlmHandler := server.NewRLMHandler(r, server.WithSubcallConcurrency(r.MaxConcurrentSubcalls))
	srv := server.NewServer(lmHandler, rlmHandler)

	lis, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		return "", 0, nil, fmt.Errorf("listen: %w", err)
	}

	if err := srv.RegisterAndStart(lis); err != nil {
		_ = lis.Close()
		return "", 0, nil, fmt.Errorf("register and start: %w", err)
	}

	host, portStr, err := net.SplitHostPort(lis.Addr().String())
	if err != nil {
		_ = srv.Stop()
		_ = lis.Close()
		return "", 0, nil, fmt.Errorf("split host port: %w", err)
	}
	port, err := strconv.Atoi(portStr)
	if err != nil {
		_ = srv.Stop()
		_ = lis.Close()
		return "", 0, nil, fmt.Errorf("parse port: %w", err)
	}

	return host, port, func() {
		_ = srv.Stop()
		_ = lis.Close()
	}, nil
}

func (r *RLM) buildInitialMessages(meta prompt.QueryMetadata, rootPrompt string) ([]prompt.Message, error) {
	return prompt.BuildSystemPrompt(prompt.RLM_SYSTEM_PROMPT, meta, nil, rootPrompt, true)
}

func (r *RLM) completionTurn(ctx context.Context, messageHistory []prompt.Message, lm client.BaseLM, env Environment) (RLMIteration, error) {
	iterStart := time.Now()
	response, err := lm.Completion(ctx, messageHistory)
	if err != nil {
		return RLMIteration{}, fmt.Errorf("model completion: %w", err)
	}

	codeBlockStrs := FindCodeBlocks(response)
	codeBlocks := make([]CodeBlock, 0, len(codeBlockStrs))
	for _, code := range codeBlockStrs {
		result, err := env.ExecuteCode(ctx, code)
		if err != nil {
			slog.Warn("DockerREPL execution failed", "error", err, "stderr", result.Stderr)
			result.Stderr = result.Stderr + "\n" + err.Error()
		} else if strings.TrimSpace(result.Stderr) != "" {
			if strings.Contains(result.Stderr, "Traceback") || strings.Contains(result.Stderr, "Error") || strings.Contains(result.Stderr, "Exception") {
				slog.Warn("DockerREPL execution finished with error", "stderr", result.Stderr)
			}
		}
		codeBlocks = append(codeBlocks, CodeBlock{Code: code, Result: result})
	}

	iter := RLMIteration{
		Prompt:        messageHistory,
		Response:      response,
		CodeBlocks:    codeBlocks,
		IterationTime: time.Since(iterStart),
	}
	for _, block := range codeBlocks {
		if block.Result.FinalAnswer != "" {
			iter.FinalAnswer = block.Result.FinalAnswer
			break
		}
	}
	return iter, nil
}

func (r *RLM) checkTimeout(iteration int) error {
	if r.MaxTimeout == nil {
		return nil
	}
	r.mu.Lock()
	elapsed := time.Since(r.completionStartTime)
	limit := *r.MaxTimeout
	r.mu.Unlock()

	if elapsed.Seconds() > limit {
		return types.NewLimitError(types.ErrTimeoutExceeded, r.bestPartialAnswer)
	}
	return nil
}

func (r *RLM) checkIterationLimits(iter RLMIteration, iterationNum int, lmHandler *LMHandler) error {
	iterationHadError := false
	for _, block := range iter.CodeBlocks {
		if strings.TrimSpace(block.Result.Stderr) != "" {
			iterationHadError = true
			break
		}
	}

	r.mu.Lock()
	if iterationHadError {
		r.consecutiveErrors++
	} else {
		r.consecutiveErrors = 0
	}
	consecutive := r.consecutiveErrors
	r.mu.Unlock()

	if r.MaxErrors != nil && consecutive >= *r.MaxErrors {
		return types.NewLimitError(types.ErrErrorThresholdExceeded, r.bestPartialAnswer)
	}

	if r.MaxBudget != nil {
		cost := lmHandler.GetUsageSummary().TotalCost()
		if cost != nil {
			r.mu.Lock()
			r.cumulativeCost = *cost
			r.mu.Unlock()
			if *cost > *r.MaxBudget {
				return types.NewLimitError(types.ErrBudgetExceeded, r.bestPartialAnswer)
			}
		}
	}

	if r.MaxTokens != nil {
		usage := lmHandler.GetUsageSummary()
		total := usage.TotalInputTokens() + usage.TotalOutputTokens()
		if total > *r.MaxTokens {
			return types.NewLimitError(types.ErrTokenLimitExceeded, r.bestPartialAnswer)
		}
	}

	return nil
}

func (r *RLM) updateBestPartial(response string) {
	if strings.TrimSpace(response) != "" {
		r.mu.Lock()
		r.bestPartialAnswer = response
		r.mu.Unlock()
	}
}

func (r *RLM) defaultAnswer(ctx context.Context, messageHistory []prompt.Message, lm client.BaseLM) (string, error) {
	finalPrompt := append(messageHistory, prompt.Message{
		Role:    "assistant",
		Content: "Please provide a final answer to the user's question based on the information provided.",
	})
	response, err := lm.Completion(ctx, finalPrompt)
	if err != nil {
		return "", fmt.Errorf("default answer: %w", err)
	}
	return response, nil
}

func (r *RLM) fallbackAnswer(ctx context.Context, prompt any) (*types.RLMChatCompletion, error) {
	start := time.Now()
	lm, err := r.newClient(r.rootModel())
	if err != nil {
		return nil, fmt.Errorf("fallback client: %w", err)
	}
	response, err := lm.Completion(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("fallback completion: %w", err)
	}
	return &types.RLMChatCompletion{
		RootModel:     r.rootModel(),
		Prompt:        prompt,
		Response:      response,
		UsageSummary:  types.UsageSummary{ModelUsageSummaries: map[string]types.ModelUsageSummary{r.rootModel(): lm.GetLastUsage()}},
		ExecutionTime: time.Since(start),
	}, nil
}

func (r *RLM) fallbackLMCall(ctx context.Context, prompt any, model string) (types.RLMChatCompletion, error) {
	start := time.Now()
	lm, err := r.newClient(model)
	if err != nil {
		return types.RLMChatCompletion{}, err
	}
	response, err := lm.Completion(ctx, prompt)
	if err != nil {
		return types.RLMChatCompletion{
			RootModel:     model,
			Prompt:        prompt,
			Response:      fmt.Sprintf("Error: LM query failed at max depth - %v", err),
			ExecutionTime: time.Since(start),
		}, nil
	}
	return types.RLMChatCompletion{
		RootModel:     model,
		Prompt:        prompt,
		Response:      response,
		UsageSummary:  types.UsageSummary{ModelUsageSummaries: map[string]types.ModelUsageSummary{model: lm.GetLastUsage()}},
		ExecutionTime: time.Since(start),
	}, nil
}

func (r *RLM) buildCompletion(answer string, prompt any, lmHandler *LMHandler, start time.Time) *types.RLMChatCompletion {
	return &types.RLMChatCompletion{
		RootModel:     r.rootModel(),
		Prompt:        prompt,
		Response:      answer,
		UsageSummary:  lmHandler.GetUsageSummary(),
		ExecutionTime: time.Since(start),
	}
}

func (r *RLM) resetTracking() {
	r.mu.Lock()
	r.cumulativeCost = 0
	r.consecutiveErrors = 0
	r.bestPartialAnswer = ""
	r.completionStartTime = time.Time{}
	r.mu.Unlock()
}

func (r *RLM) setStartTime(t time.Time) {
	r.mu.Lock()
	r.completionStartTime = t
	r.mu.Unlock()
}

func (r *RLM) maxBudgetRemainingLocked() *float64 {
	if r.MaxBudget == nil {
		return nil
	}
	remaining := *r.MaxBudget - r.cumulativeCost
	return &remaining
}

func (r *RLM) maxTimeoutRemainingLocked() *float64 {
	if r.MaxTimeout == nil || r.completionStartTime.IsZero() {
		return nil
	}
	remaining := *r.MaxTimeout - time.Since(r.completionStartTime).Seconds()
	return &remaining
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
