package rlm

import (
	"context"
	"sync"
	"testing"

	"rlm-golang/internal/client"
	internal "rlm-golang/internal/rlm"
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
}

func newFakeLM(model string, responses ...string) *mockLM {
	return &mockLM{model: model, responses: responses}
}

func (f *mockLM) Completion(ctx context.Context, prompt any) (string, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, prompt)
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

// mockEnv is a test double for internal.Environment.
type mockEnv struct {
	mu        sync.Mutex
	calls     []string
	results   []internal.REPLResult
	index     int
	workspace string
}

func (f *mockEnv) ExecuteCode(ctx context.Context, code string) (internal.REPLResult, error) {
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
	return internal.REPLResult{}, nil
}

func (f *mockEnv) LoadContext(ctx context.Context, payload any) error { return nil }
func (f *mockEnv) Cleanup(ctx context.Context) error                  { return nil }

func newTestRLM(t *testing.T, lm *mockLM, env *mockEnv) *RLM {
	t.Helper()
	r, err := New(WithModel(lm.model))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	r.internal.Backend = "test"
	r.internal.BackendKwargs = map[string]any{"model_name": lm.model}
	r.internal.MaxDepth = 2
	r.internal.MaxIterations = 5
	r.internal.MaxConcurrentSubcalls = 4
	r.internal.ClientFactory = func(model string) (client.BaseLM, error) {
		return lm, nil
	}
	r.internal.EnvFactory = func(host string, port int, depth int, workspace string) (internal.Environment, error) {
		if env != nil {
			env.workspace = workspace
		}
		return env, nil
	}
	return r
}

func TestIntegrationSingleTurnCompletion(t *testing.T) {
	lm := newFakeLM("test-model",
		"```repl\nanswer['content']='done'\nanswer['ready']=True\n```")
	env := &mockEnv{
		results: []internal.REPLResult{{FinalAnswer: "done"}},
	}
	r := newTestRLM(t, lm, env)

	result, err := r.Completion(context.Background(), "answer this")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}
	if result.RootModel != "test-model" {
		t.Errorf("RootModel = %q, want %q", result.RootModel, "test-model")
	}
	if len(env.calls) != 1 {
		t.Errorf("env calls = %d, want 1", len(env.calls))
	}
}

func TestIntegrationIterationBudgetExhausted(t *testing.T) {
	lm := newFakeLM("test-model",
		"```repl\nprint('still thinking')\n```",
		"```repl\nprint('still thinking')\n```",
		"default answer")
	env := &mockEnv{
		results: []internal.REPLResult{
			{Stdout: "still thinking\n"},
			{Stdout: "still thinking\n"},
		},
	}
	r := newTestRLM(t, lm, env)
	r.internal.MaxIterations = 2

	result, err := r.Completion(context.Background(), "question")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if result.Response != "default answer" {
		t.Errorf("Response = %q, want %q", result.Response, "default answer")
	}
	if len(env.calls) != 2 {
		t.Errorf("env calls = %d, want 2", len(env.calls))
	}
}

func TestIntegrationResultShape(t *testing.T) {
	lm := newFakeLM("shape-model",
		"```repl\nanswer['content']='shape answer'\nanswer['ready']=True\n```")
	env := &mockEnv{
		results: []internal.REPLResult{{FinalAnswer: "shape answer"}},
	}
	r := newTestRLM(t, lm, env)

	result, err := r.Completion(context.Background(), "question")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if result.UsageSummary.ModelUsageSummaries == nil {
		t.Error("expected non-nil ModelUsageSummaries")
	}
	if result.ExecutionTime < 0 {
		t.Errorf("ExecutionTime = %v, want >= 0", result.ExecutionTime)
	}
}
