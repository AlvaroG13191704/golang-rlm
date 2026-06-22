package rlm_test

import (
	"context"
	"errors"
	"strings"
	"sync"
	"testing"
	"time"

	"rlm-golang/internal/client"
	"rlm-golang/internal/rlm"
	"rlm-golang/internal/types"
)

// mockLM is a test double for client.BaseLM.
type mockLM struct {
	model     string
	mu        sync.Mutex
	calls     []any
	responses []string
	index     int
	usageFn   func() types.ModelUsageSummary
	fn        func(ctx context.Context, prompt any) (string, error)
}

func newFakeLM(model string, responses ...string) *mockLM {
	return &mockLM{model: model, responses: responses}
}

func (f *mockLM) Completion(ctx context.Context, prompt any) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, prompt)
	if f.fn != nil {
		return f.fn(ctx, prompt)
	}
	i := f.index
	if i < len(f.responses)-1 {
		f.index++
	} else if len(f.responses) > 0 {
		f.index = len(f.responses) - 1
	}
	if i < len(f.responses) {
		return f.responses[i], nil
	}
	return "ok", nil
}

func (f *mockLM) GetUsageSummary() types.ModelUsageSummary { return f.currentUsage() }
func (f *mockLM) GetLastUsage() types.ModelUsageSummary    { return f.currentUsage() }

func (f *mockLM) currentUsage() types.ModelUsageSummary {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.usageFn != nil {
		return f.usageFn()
	}
	return types.ModelUsageSummary{TotalCalls: len(f.calls)}
}

// mockEnv is a test double for rlm.Environment.
type mockEnv struct {
	mu        sync.Mutex
	calls     []string
	results   []rlm.REPLResult
	index     int
	workspace string
}

func (f *mockEnv) ExecuteCode(ctx context.Context, code string) (rlm.REPLResult, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, code)
	i := f.index
	if i < len(f.results)-1 {
		f.index++
	} else if len(f.results) > 0 {
		f.index = len(f.results) - 1
	}
	if i < len(f.results) {
		return f.results[i], nil
	}
	return rlm.REPLResult{}, nil
}

func (f *mockEnv) LoadContext(ctx context.Context, payload any) error { return nil }
func (f *mockEnv) Cleanup(ctx context.Context) error                  { return nil }

func newTestRLM(t *testing.T, lm *mockLM, env *mockEnv, opts ...func(*rlm.RLM)) *rlm.RLM {
	t.Helper()
	r := &rlm.RLM{
		Backend:               "test",
		BackendKwargs:         map[string]any{"model_name": lm.model},
		MaxDepth:              2,
		MaxIterations:         5,
		MaxConcurrentSubcalls: 4,
		ClientFactory: func(model string) (client.BaseLM, error) {
			return lm, nil
		},
		EnvFactory: func(host string, port int, depth int, workspace string) (rlm.Environment, error) {
			if env != nil {
				env.workspace = workspace
			}
			return env, nil
		},
	}
	for _, opt := range opts {
		opt(r)
	}
	return r
}

func TestRLMCompletionSingleTurn(t *testing.T) {
	lm := newFakeLM("test-model",
		"```repl\nanswer['content']='done'\nanswer['ready']=True\n```")
	env := &mockEnv{
		results: []rlm.REPLResult{{
			FinalAnswer: "done",
		}},
	}
	r := newTestRLM(t, lm, env)

	completion, err := r.Completion(context.Background(), "context payload", "answer this")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if completion.Response != "done" {
		t.Errorf("response = %q, want %q", completion.Response, "done")
	}
	if len(env.calls) != 1 {
		t.Errorf("env calls = %d, want 1", len(env.calls))
	}
	if len(lm.calls) != 1 {
		t.Errorf("lm calls = %d, want 1", len(lm.calls))
	}
}

func TestRLMCompletionIterationBudgetExhausted(t *testing.T) {
	lm := newFakeLM("test-model",
		"```repl\nprint('still thinking')\n```",
		"```repl\nprint('still thinking')\n```",
		"default answer")
	env := &mockEnv{
		results: []rlm.REPLResult{
			{Stdout: "still thinking\n"},
			{Stdout: "still thinking\n"},
		},
	}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.MaxIterations = 2
	})

	completion, err := r.Completion(context.Background(), "ctx", "question")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if completion.Response != "default answer" {
		t.Errorf("response = %q, want %q", completion.Response, "default answer")
	}
	if len(env.calls) != 2 {
		t.Errorf("env calls = %d, want 2", len(env.calls))
	}
}

func TestRLMCompletionTimeout(t *testing.T) {
	maxTimeout := 0.05
	lm := newFakeLM("test-model", "```repl\nprint('step 1')\n```")
	lm.fn = func(ctx context.Context, prompt any) (string, error) {
		time.Sleep(80 * time.Millisecond)
		return lm.responses[0], nil
	}
	env := &mockEnv{
		results: []rlm.REPLResult{{Stdout: "step 1\n"}},
	}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.MaxTimeout = &maxTimeout
	})

	_, err := r.Completion(context.Background(), "ctx", "question")
	if err == nil {
		t.Fatalf("expected timeout error")
	}
	if !errors.Is(err, types.ErrTimeoutExceeded) {
		t.Errorf("err = %v, want timeout exceeded", err)
	}
	var limitErr *types.LimitError
	if !errors.As(err, &limitErr) {
		t.Fatalf("expected *LimitError, got %T", err)
	}
	if !strings.Contains(limitErr.PartialAnswer(), "step 1") {
		t.Errorf("partial answer = %q, want step 1", limitErr.PartialAnswer())
	}
}

func TestRLMCompletionBudgetExceeded(t *testing.T) {
	maxBudget := 0.5
	cost := 1.0
	lm := newFakeLM("test-model", "```repl\nprint('costly')\n```")
	lm.usageFn = func() types.ModelUsageSummary {
		return types.ModelUsageSummary{TotalCalls: 1, TotalCost: &cost}
	}
	env := &mockEnv{results: []rlm.REPLResult{{Stdout: "costly\n"}}}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.MaxBudget = &maxBudget
	})

	_, err := r.Completion(context.Background(), "ctx", "q")
	if err == nil {
		t.Fatalf("expected budget error")
	}
	if !errors.Is(err, types.ErrBudgetExceeded) {
		t.Errorf("err = %v, want budget exceeded", err)
	}
}

func TestRLMCompletionTokenLimitExceeded(t *testing.T) {
	maxTokens := 5
	lm := newFakeLM("test-model", "```repl\nprint('tok')\n```")
	lm.usageFn = func() types.ModelUsageSummary {
		return types.ModelUsageSummary{TotalCalls: 1, TotalInputTokens: 3, TotalOutputTokens: 3}
	}
	env := &mockEnv{results: []rlm.REPLResult{{Stdout: "tok\n"}}}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.MaxTokens = &maxTokens
	})

	_, err := r.Completion(context.Background(), "ctx", "q")
	if err == nil {
		t.Fatalf("expected token error")
	}
	if !errors.Is(err, types.ErrTokenLimitExceeded) {
		t.Errorf("err = %v, want token limit exceeded", err)
	}
}

func TestRLMCompletionErrorThresholdExceeded(t *testing.T) {
	maxErrors := 2
	lm := newFakeLM("test-model",
		"```repl\nprint('bad')\n```",
		"```repl\nprint('bad')\n```")
	env := &mockEnv{
		results: []rlm.REPLResult{
			{Stderr: "error one"},
			{Stderr: "error two"},
		},
	}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.MaxErrors = &maxErrors
	})

	_, err := r.Completion(context.Background(), "ctx", "q")
	if err == nil {
		t.Fatalf("expected error threshold")
	}
	if !errors.Is(err, types.ErrErrorThresholdExceeded) {
		t.Errorf("err = %v, want error threshold exceeded", err)
	}
}

func TestRLMCompletionErrorsResetOnSuccess(t *testing.T) {
	maxErrors := 2
	lm := newFakeLM("test-model",
		"```repl\nprint('bad')\n```",
		"```repl\nprint('ok')\n```",
		"```repl\nprint('bad again')\n```",
		"```repl\nanswer['content']='done'\nanswer['ready']=True\n```")
	env := &mockEnv{
		results: []rlm.REPLResult{
			{Stderr: "error"},
			{Stdout: "ok\n"},
			{Stderr: "error"},
			{FinalAnswer: "done"},
		},
	}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.MaxErrors = &maxErrors
	})

	completion, err := r.Completion(context.Background(), "ctx", "q")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if completion.Response != "done" {
		t.Errorf("response = %q, want done", completion.Response)
	}
}

func TestRLMCompletionFallbackAtMaxDepth(t *testing.T) {
	lm := newFakeLM("test-model", "plain answer")
	env := &mockEnv{}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.Depth = 1
		r.MaxDepth = 1
	})

	completion, err := r.Completion(context.Background(), "ctx", "q")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if completion.Response != "plain answer" {
		t.Errorf("response = %q, want plain answer", completion.Response)
	}
	if len(env.calls) != 0 {
		t.Errorf("env calls = %d, want 0", len(env.calls))
	}
}

func TestRLMCompletionCleansUpEnvironment(t *testing.T) {
	lm := newFakeLM("test-model", "```repl\nanswer['content']='done'\nanswer['ready']=True\n```")
	env := &mockEnv{results: []rlm.REPLResult{{FinalAnswer: "done"}}}
	r := newTestRLM(t, lm, env)

	_, err := r.Completion(context.Background(), "ctx", "q")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if env.workspace == "" {
		t.Errorf("env factory was not called with a workspace")
	}
}
