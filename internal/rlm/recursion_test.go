package rlm_test

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"rlm-golang/internal/client"
	"rlm-golang/internal/rlm"
	"rlm-golang/internal/types"
)

func TestRLMSubcallSpawnsChildRLM(t *testing.T) {
	rootLM := newFakeLM("root-model", "root response")
	childLM := newFakeLM("child-model",
		"```repl\nanswer['content']='child answer'\nanswer['ready']=True\n```")

	childEnv := &mockEnv{
		results: []rlm.REPLResult{{FinalAnswer: "child answer"}},
	}

	var childWorkspace string
	r := &rlm.RLM{
		Backend:               "test",
		BackendKwargs:         map[string]any{"model_name": "root-model"},
		Depth:                 0,
		MaxDepth:              2,
		MaxIterations:         5,
		MaxConcurrentSubcalls: 4,
		ClientFactory: func(model string) (client.BaseLM, error) {
			if model == "child-model" {
				return childLM, nil
			}
			return rootLM, nil
		},
		EnvFactory: func(host string, port int, depth int, workspace string) (rlm.Environment, error) {
			childWorkspace = workspace
			childEnv.workspace = workspace
			return childEnv, nil
		},
	}

	completion, err := r.Subcall(context.Background(), "child prompt", "child-model", 1)
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if completion.Response != "child answer" {
		t.Errorf("response = %q, want %q", completion.Response, "child answer")
	}
	if childWorkspace == "" {
		t.Errorf("child workspace was empty")
	}
	if len(childEnv.calls) != 1 {
		t.Errorf("child env calls = %d, want 1", len(childEnv.calls))
	}
}

func TestRLMSubcallFallbackAtMaxDepth(t *testing.T) {
	rootLM := newFakeLM("root-model", "plain lm answer")
	r := &rlm.RLM{
		Backend:               "test",
		BackendKwargs:         map[string]any{"model_name": "root-model"},
		Depth:                 0,
		MaxDepth:              1,
		MaxIterations:         5,
		MaxConcurrentSubcalls: 4,
		ClientFactory: func(model string) (client.BaseLM, error) {
			return rootLM, nil
		},
		EnvFactory: func(host string, port int, depth int, workspace string) (rlm.Environment, error) {
			t.Fatal("should not spawn environment at max depth")
			return nil, nil
		},
	}

	completion, err := r.Subcall(context.Background(), "prompt", "", 1)
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if completion.Response != "plain lm answer" {
		t.Errorf("response = %q, want %q", completion.Response, "plain lm answer")
	}
	if len(rootLM.calls) != 1 {
		t.Errorf("root LM calls = %d, want 1", len(rootLM.calls))
	}
}

func TestRLMSubcallExhaustedBudget(t *testing.T) {
	maxBudget := 0.5
	cost := 1.0
	lm := newFakeLM("root-model", "```repl\nprint('costly')\n```")
	lm.usageFn = func() types.ModelUsageSummary {
		return types.ModelUsageSummary{TotalCalls: 1, TotalCost: &cost}
	}
	env := &mockEnv{results: []rlm.REPLResult{{Stdout: "costly\n"}}}
	r := newTestRLM(t, lm, env, func(r *rlm.RLM) {
		r.MaxBudget = &maxBudget
	})

	// Run the parent completion once; it should fail with budget exceeded
	// and leave cumulative cost greater than the budget.
	_, err := r.Completion(context.Background(), "ctx", "q")
	if err == nil {
		t.Fatalf("expected budget error from parent completion")
	}

	// A subsequent subcall should immediately report budget exhaustion.
	completion, err := r.Subcall(context.Background(), "subtask", "", 1)
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if !strings.Contains(completion.Response, "Budget") {
		t.Errorf("response = %q, want budget exhaustion message", completion.Response)
	}
}

func TestRLMSubcallInheritsRemainingBudget(t *testing.T) {
	maxBudget := 1.0
	rootCost := 0.3
	childCost := 0.9

	rootLM := newFakeLM("root-model", "```repl\nprint('step')\n```")
	rootLM.usageFn = func() types.ModelUsageSummary {
		return types.ModelUsageSummary{TotalCalls: 1, TotalCost: &rootCost}
	}

	childLM := newFakeLM("child-model", "```repl\nprint('child step')\n```")
	childLM.usageFn = func() types.ModelUsageSummary {
		return types.ModelUsageSummary{TotalCalls: 1, TotalCost: &childCost}
	}
	childEnv := &mockEnv{results: []rlm.REPLResult{{Stdout: "child step\n"}}}

	r := &rlm.RLM{
		Backend:               "test",
		BackendKwargs:         map[string]any{"model_name": "root-model"},
		Depth:                 0,
		MaxDepth:              2,
		MaxIterations:         5,
		MaxBudget:             &maxBudget,
		MaxConcurrentSubcalls: 4,
		ClientFactory: func(model string) (client.BaseLM, error) {
			if model == "child-model" {
				return childLM, nil
			}
			return rootLM, nil
		},
		EnvFactory: func(host string, port int, depth int, workspace string) (rlm.Environment, error) {
			// Capture the child RLM's remaining budget by executing one iteration
			// and checking whether it stops on the budget limit.
			childEnv.workspace = workspace
			return childEnv, nil
		},
	}

	// Spend part of the parent's budget so the child inherits the remainder.
	_, err := r.Completion(context.Background(), "ctx", "q")
	if err != nil {
		t.Fatalf("parent Completion: %v", err)
	}

	// The child is created with remaining budget = maxBudget - rootCost = 0.7.
	// Its single iteration costs 0.9, so it should exceed its own budget.
	completion, err := r.Subcall(context.Background(), "subtask", "child-model", 1)
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if !strings.Contains(completion.Response, "Budget") {
		t.Errorf("response = %q, want budget exceeded in child", completion.Response)
	}
}

func TestRLMSubcallInheritsRemainingTimeout(t *testing.T) {
	maxTimeout := 0.1
	rootLM := newFakeLM("root-model", "root response")
	rootLM.fn = func(ctx context.Context, prompt any) (string, error) {
		time.Sleep(200 * time.Millisecond)
		return rootLM.responses[0], nil
	}

	r := newTestRLM(t, rootLM, &mockEnv{}, func(r *rlm.RLM) {
		r.MaxTimeout = &maxTimeout
	})

	_, err := r.Completion(context.Background(), "ctx", "q")
	if err == nil {
		t.Fatalf("expected timeout error")
	}
	if !errors.Is(err, types.ErrTimeoutExceeded) {
		t.Errorf("err = %v, want timeout exceeded", err)
	}

	// After the parent timed out, a subcall should also report timeout exhaustion.
	completion, err := r.Subcall(context.Background(), "subtask", "", 1)
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if !strings.Contains(completion.Response, "Timeout") {
		t.Errorf("response = %q, want timeout exhaustion message", completion.Response)
	}
}
