package rlm

import (
	"context"
	"time"

	"rlm-golang/internal/types"
)

// REPLResult captures the outcome of executing code inside the REPL.
type REPLResult struct {
	Stdout        string
	Stderr        string
	Locals        map[string]string
	FinalAnswer   string
	ExecutionTime time.Duration
	LLMCalls      []types.RLMChatCompletion
}

// Environment is the runtime the orchestrator uses to execute model code and
// manage context.
type Environment interface {
	ExecuteCode(ctx context.Context, code string) (REPLResult, error)
	LoadContext(ctx context.Context, payload any) error
	Cleanup(ctx context.Context) error
}
