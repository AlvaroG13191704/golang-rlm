package rlm

import (
	"testing"
	"time"

	"rlm-golang/internal/types"
)

func TestREPLResultConstruction(t *testing.T) {
	result := REPLResult{
		Stdout:        "42\n",
		Stderr:        "",
		Locals:        map[string]string{"x": "42"},
		FinalAnswer:   "42",
		ExecutionTime: 50 * time.Millisecond,
		LLMCalls: []types.RLMChatCompletion{
			{RootModel: "llama3.1", Response: "ok"},
		},
	}

	if got := result.Stdout; got != "42\n" {
		t.Errorf("Stdout = %q, want %q", got, "42\n")
	}
	if got := result.FinalAnswer; got != "42" {
		t.Errorf("FinalAnswer = %q, want %q", got, "42")
	}
	if got := len(result.LLMCalls); got != 1 {
		t.Errorf("len(LLMCalls) = %d, want 1", got)
	}
	if got := result.Locals["x"]; got != "42" {
		t.Errorf("Locals[x] = %q, want %q", got, "42")
	}
}
