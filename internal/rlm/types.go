package rlm

import (
	"errors"
	"fmt"
	"time"
)

// ModelUsageSummary aggregates usage counters for a single model.
type ModelUsageSummary struct {
	TotalCalls        int
	TotalInputTokens  int
	TotalOutputTokens int
	TotalCost         *float64
}

// UsageSummary aggregates per-model usage across the whole session.
type UsageSummary struct {
	ModelUsageSummaries map[string]ModelUsageSummary
}

// TotalCost returns the aggregate cost across all models, or nil if no model
// reports cost.
func (u UsageSummary) TotalCost() *float64 {
	var total float64
	hasCost := false
	for _, summary := range u.ModelUsageSummaries {
		if summary.TotalCost != nil {
			total += *summary.TotalCost
			hasCost = true
		}
	}
	if !hasCost {
		return nil
	}
	return &total
}

// TotalInputTokens returns the aggregate input tokens across all models.
func (u UsageSummary) TotalInputTokens() int {
	total := 0
	for _, summary := range u.ModelUsageSummaries {
		total += summary.TotalInputTokens
	}
	return total
}

// TotalOutputTokens returns the aggregate output tokens across all models.
func (u UsageSummary) TotalOutputTokens() int {
	total := 0
	for _, summary := range u.ModelUsageSummaries {
		total += summary.TotalOutputTokens
	}
	return total
}

// RLMChatCompletion records a single LLM call made from within the REPL.
type RLMChatCompletion struct {
	RootModel     string
	Prompt        any
	Response      string
	UsageSummary  UsageSummary
	ExecutionTime time.Duration
	Metadata      map[string]any
	Error         string
}

// REPLResult captures the outcome of executing code inside the REPL.
type REPLResult struct {
	Stdout        string
	Stderr        string
	Locals        map[string]string
	FinalAnswer   string
	ExecutionTime time.Duration
	LLMCalls      []RLMChatCompletion
}

// Sentinel errors for limit exceeded conditions.
var (
	ErrTimeoutExceeded        = errors.New("timeout exceeded")
	ErrBudgetExceeded         = errors.New("budget exceeded")
	ErrTokenLimitExceeded     = errors.New("token limit exceeded")
	ErrErrorThresholdExceeded = errors.New("error threshold exceeded")
)

// LimitError wraps a sentinel limit error and carries the best partial answer
// available when the limit was reached.
type LimitError struct {
	Err           error
	partialAnswer string
}

// Error implements the error interface.
func (e *LimitError) Error() string {
	return e.Err.Error()
}

// Unwrap returns the underlying sentinel error for errors.Is matching.
func (e *LimitError) Unwrap() error {
	return e.Err
}

// PartialAnswer returns the best partial answer captured when the limit was
// exceeded.
func (e *LimitError) PartialAnswer() string {
	return e.partialAnswer
}

// NewLimitError constructs a typed limit error for the given sentinel and
// partial answer.
func NewLimitError(sentinel error, partialAnswer string) *LimitError {
	return &LimitError{Err: sentinel, partialAnswer: partialAnswer}
}

// LimitErrorf constructs a typed limit error with a formatted partial answer.
func LimitErrorf(sentinel error, format string, args ...any) *LimitError {
	return NewLimitError(sentinel, fmt.Sprintf(format, args...))
}
