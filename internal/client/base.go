// Package client defines the BaseLM abstraction for language-model backends and
// ships a concrete Ollama HTTP implementation.
package client

import (
	"context"

	"rlm-golang/internal/types"
)

// BaseLM is the common interface for all LM backends used by the RLM handler.
type BaseLM interface {
	Completion(ctx context.Context, prompt any) (string, error)
	GetUsageSummary() types.ModelUsageSummary
	GetLastUsage() types.ModelUsageSummary
}

// AddModelUsage returns a new ModelUsageSummary with the counters of a and b
// combined. Costs are summed only when both inputs report a cost; otherwise the
// result has a nil cost.
func AddModelUsage(a, b types.ModelUsageSummary) types.ModelUsageSummary {
	sum := types.ModelUsageSummary{
		TotalCalls:        a.TotalCalls + b.TotalCalls,
		TotalInputTokens:  a.TotalInputTokens + b.TotalInputTokens,
		TotalOutputTokens: a.TotalOutputTokens + b.TotalOutputTokens,
	}
	if a.TotalCost != nil && b.TotalCost != nil {
		total := *a.TotalCost + *b.TotalCost
		sum.TotalCost = &total
	}
	return sum
}

// TotalTokens returns the sum of input and output tokens in a summary.
func TotalTokens(summary types.ModelUsageSummary) int {
	return summary.TotalInputTokens + summary.TotalOutputTokens
}
