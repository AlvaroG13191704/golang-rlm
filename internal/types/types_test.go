package types

import (
	"errors"
	"testing"
	"time"
)

func TestModelUsageSummaryConstruction(t *testing.T) {
	summary := ModelUsageSummary{
		TotalCalls:        3,
		TotalInputTokens:  100,
		TotalOutputTokens: 50,
		TotalCost:         nil,
	}

	if got := summary.TotalCalls; got != 3 {
		t.Errorf("TotalCalls = %d, want 3", got)
	}
	if got := summary.TotalInputTokens; got != 100 {
		t.Errorf("TotalInputTokens = %d, want 100", got)
	}
	if got := summary.TotalOutputTokens; got != 50 {
		t.Errorf("TotalOutputTokens = %d, want 50", got)
	}
	if summary.TotalCost != nil {
		t.Errorf("TotalCost = %v, want nil", *summary.TotalCost)
	}
}

func TestUsageSummaryAggregatesModels(t *testing.T) {
	usage := UsageSummary{
		ModelUsageSummaries: map[string]ModelUsageSummary{
			"llama3.1": {
				TotalCalls:        2,
				TotalInputTokens:  20,
				TotalOutputTokens: 10,
			},
			"qwen2.5": {
				TotalCalls:        1,
				TotalInputTokens:  5,
				TotalOutputTokens: 5,
			},
		},
	}

	if got := len(usage.ModelUsageSummaries); got != 2 {
		t.Fatalf("len(ModelUsageSummaries) = %d, want 2", got)
	}

	llama := usage.ModelUsageSummaries["llama3.1"]
	if got := llama.TotalCalls; got != 2 {
		t.Errorf("llama3.1 TotalCalls = %d, want 2", got)
	}
}

func TestRLMChatCompletionConstruction(t *testing.T) {
	completion := RLMChatCompletion{
		RootModel:     "llama3.1",
		Prompt:        "hello",
		Response:      "hi",
		UsageSummary:  UsageSummary{ModelUsageSummaries: map[string]ModelUsageSummary{}},
		ExecutionTime: 100 * time.Millisecond,
		Metadata:      map[string]any{"turn": 1},
		Error:         "",
	}

	if got := completion.RootModel; got != "llama3.1" {
		t.Errorf("RootModel = %q, want %q", got, "llama3.1")
	}
	if got := completion.Response; got != "hi" {
		t.Errorf("Response = %q, want %q", got, "hi")
	}
	if got := completion.ExecutionTime; got != 100*time.Millisecond {
		t.Errorf("ExecutionTime = %v, want %v", got, 100*time.Millisecond)
	}
}

func TestUsageSummaryAggregation(t *testing.T) {
	cost1 := 0.5
	cost2 := 1.5
	usage := UsageSummary{
		ModelUsageSummaries: map[string]ModelUsageSummary{
			"llama3.1": {TotalCalls: 2, TotalInputTokens: 20, TotalOutputTokens: 10, TotalCost: &cost1},
			"qwen2.5":  {TotalCalls: 1, TotalInputTokens: 5, TotalOutputTokens: 5, TotalCost: &cost2},
		},
	}

	if got := usage.TotalInputTokens(); got != 25 {
		t.Errorf("TotalInputTokens() = %d, want 25", got)
	}
	if got := usage.TotalOutputTokens(); got != 15 {
		t.Errorf("TotalOutputTokens() = %d, want 15", got)
	}

	got := usage.TotalCost()
	if got == nil || *got != 2.0 {
		t.Errorf("TotalCost() = %v, want 2.0", got)
	}
}

func TestUsageSummaryTotalCostNilWhenNoCosts(t *testing.T) {
	usage := UsageSummary{
		ModelUsageSummaries: map[string]ModelUsageSummary{
			"llama3.1": {TotalCalls: 1, TotalInputTokens: 10, TotalOutputTokens: 5},
		},
	}

	if got := usage.TotalCost(); got != nil {
		t.Errorf("TotalCost() = %v, want nil", got)
	}
}

func TestLimitErrorExposesPartialAnswer(t *testing.T) {
	tests := []struct {
		name   string
		err    error
		want   string
		answer string
	}{
		{
			name:   "timeout exceeded",
			err:    NewLimitError(ErrTimeoutExceeded, "partial"),
			want:   "timeout exceeded",
			answer: "partial",
		},
		{
			name:   "budget exceeded",
			err:    NewLimitError(ErrBudgetExceeded, ""),
			want:   "budget exceeded",
			answer: "",
		},
		{
			name:   "token limit exceeded",
			err:    NewLimitError(ErrTokenLimitExceeded, "too many tokens"),
			want:   "token limit exceeded",
			answer: "too many tokens",
		},
		{
			name:   "error threshold exceeded",
			err:    NewLimitError(ErrErrorThresholdExceeded, "best effort"),
			want:   "error threshold exceeded",
			answer: "best effort",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.Error(); got != tt.want {
				t.Errorf("Error() = %q, want %q", got, tt.want)
			}

			var le *LimitError
			if !errors.As(tt.err, &le) {
				t.Fatalf("error is not a *LimitError")
			}
			if got := le.PartialAnswer(); got != tt.answer {
				t.Errorf("PartialAnswer() = %q, want %q", got, tt.answer)
			}
		})
	}
}

func TestLimitErrorf(t *testing.T) {
	err := LimitErrorf(ErrTimeoutExceeded, "answer after %d turns", 3)
	if !errors.Is(err, ErrTimeoutExceeded) {
		t.Errorf("LimitErrorf did not wrap ErrTimeoutExceeded")
	}
	if got := err.PartialAnswer(); got != "answer after 3 turns" {
		t.Errorf("PartialAnswer() = %q, want %q", got, "answer after 3 turns")
	}
}

func TestLimitErrorMatchesSentinel(t *testing.T) {
	if !errors.Is(&LimitError{Err: ErrTimeoutExceeded}, ErrTimeoutExceeded) {
		t.Errorf("timeout limit error did not match sentinel")
	}
	if !errors.Is(&LimitError{Err: ErrBudgetExceeded}, ErrBudgetExceeded) {
		t.Errorf("budget limit error did not match sentinel")
	}
	if !errors.Is(&LimitError{Err: ErrTokenLimitExceeded}, ErrTokenLimitExceeded) {
		t.Errorf("token limit error did not match sentinel")
	}
	if !errors.Is(&LimitError{Err: ErrErrorThresholdExceeded}, ErrErrorThresholdExceeded) {
		t.Errorf("error threshold limit error did not match sentinel")
	}
}
