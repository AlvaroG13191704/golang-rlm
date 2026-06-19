package rlmv1

import (
	"testing"
	"time"

	"google.golang.org/protobuf/proto"
	"google.golang.org/protobuf/types/known/durationpb"
)

func TestCompleteRequestConstruction(t *testing.T) {
	tests := []struct {
		name       string
		prompt     string
		promptType PromptType
		model      string
		depth      int32
	}{
		{
			name:       "raw prompt with model and depth",
			prompt:     "Summarize",
			promptType: PromptType_PROMPT_RAW,
			model:      "llama3.1",
			depth:      1,
		},
		{
			name:       "JSON prompt with defaults",
			prompt:     `{"query":"hello"}`,
			promptType: PromptType_PROMPT_JSON,
			model:      "",
			depth:      0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			req := &CompleteRequest{
				Prompt:     tt.prompt,
				PromptType: tt.promptType,
				Model:      tt.model,
				Depth:      tt.depth,
			}

			if got := req.GetPrompt(); got != tt.prompt {
				t.Errorf("GetPrompt() = %q, want %q", got, tt.prompt)
			}
			if got := req.GetPromptType(); got != tt.promptType {
				t.Errorf("GetPromptType() = %v, want %v", got, tt.promptType)
			}
			if got := req.GetModel(); got != tt.model {
				t.Errorf("GetModel() = %q, want %q", got, tt.model)
			}
			if got := req.GetDepth(); got != tt.depth {
				t.Errorf("GetDepth() = %d, want %d", got, tt.depth)
			}
		})
	}
}

func TestCompleteResponseConstruction(t *testing.T) {
	tests := []struct {
		name  string
		build func() *CompleteResponse
		want  *CompleteResponse
	}{
		{
			name: "success response",
			build: func() *CompleteResponse {
				return &CompleteResponse{
					Content:       "Answer A",
					RootModel:     "llama3.1",
					Usage:         &ModelUsageSummary{TotalCalls: 1, TotalInputTokens: 10, TotalOutputTokens: 5},
					ExecutionTime: durationpb.New(100 * time.Millisecond),
				}
			},
			want: &CompleteResponse{
				Content:       "Answer A",
				RootModel:     "llama3.1",
				Usage:         &ModelUsageSummary{TotalCalls: 1, TotalInputTokens: 10, TotalOutputTokens: 5},
				ExecutionTime: durationpb.New(100 * time.Millisecond),
			},
		},
		{
			name: "error response",
			build: func() *CompleteResponse {
				return &CompleteResponse{Error: "model unavailable"}
			},
			want: &CompleteResponse{Error: "model unavailable"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := tt.build()
			if !proto.Equal(got, tt.want) {
				t.Errorf("CompleteResponse mismatch:\n got: %v\nwant: %v", got, tt.want)
			}
		})
	}
}

func TestBatchedRequestOrderPreserved(t *testing.T) {
	req := &BatchedRequest{
		Items: []*CompleteRequest{
			{Prompt: "A", Model: "m1"},
			{Prompt: "B", Model: "m1"},
			{Prompt: "C", Model: "m1"},
		},
	}

	if got := len(req.GetItems()); got != 3 {
		t.Fatalf("len(Items) = %d, want 3", got)
	}

	for i, want := range []string{"A", "B", "C"} {
		if got := req.GetItems()[i].GetPrompt(); got != want {
			t.Errorf("Items[%d].Prompt = %q, want %q", i, got, want)
		}
	}
}

func TestBatchedResponseOrderPreserved(t *testing.T) {
	resp := &BatchedResponse{
		Responses: []*CompleteResponse{
			{Content: "Answer A", RootModel: "m1"},
			{Content: "Answer B", RootModel: "m1"},
			{Content: "Answer C", RootModel: "m1"},
		},
	}

	if got := len(resp.GetResponses()); got != 3 {
		t.Fatalf("len(Responses) = %d, want 3", got)
	}

	for i, want := range []string{"Answer A", "Answer B", "Answer C"} {
		if got := resp.GetResponses()[i].GetContent(); got != want {
			t.Errorf("Responses[%d].Content = %q, want %q", i, got, want)
		}
	}
}

func TestSubcallRequestConstruction(t *testing.T) {
	req := &SubcallRequest{
		Prompt:     "child task",
		PromptType: PromptType_PROMPT_RAW,
		Model:      "qwen2.5",
		Depth:      2,
	}

	if got := req.GetPrompt(); got != "child task" {
		t.Errorf("GetPrompt() = %q, want %q", got, "child task")
	}
	if got := req.GetDepth(); got != 2 {
		t.Errorf("GetDepth() = %d, want 2", got)
	}
}

func TestGetUsageResponseConstruction(t *testing.T) {
	resp := &GetUsageResponse{
		Usage: &UsageSummary{
			ModelUsageSummaries: map[string]*ModelUsageSummary{
				"llama3.1": {TotalCalls: 2, TotalInputTokens: 20, TotalOutputTokens: 10},
			},
		},
	}

	got := resp.GetUsage().GetModelUsageSummaries()["llama3.1"]
	want := &ModelUsageSummary{TotalCalls: 2, TotalInputTokens: 20, TotalOutputTokens: 10}
	if !proto.Equal(got, want) {
		t.Errorf("model usage mismatch:\n got: %v\nwant: %v", got, want)
	}
}

func TestBatchedSubcallOrderPreserved(t *testing.T) {
	req := &BatchedSubcallRequest{
		Items: []*SubcallRequest{
			{Prompt: "x", Depth: 2},
			{Prompt: "y", Depth: 2},
		},
	}

	if got := len(req.GetItems()); got != 2 {
		t.Fatalf("len(Items) = %d, want 2", got)
	}

	for i, want := range []string{"x", "y"} {
		if got := req.GetItems()[i].GetPrompt(); got != want {
			t.Errorf("Items[%d].Prompt = %q, want %q", i, got, want)
		}
	}

	resp := &BatchedSubcallResponse{
		Responses: []*SubcallResponse{
			{Content: "answer x", RootModel: "m1"},
			{Content: "answer y", RootModel: "m1"},
		},
	}

	if got := len(resp.GetResponses()); got != 2 {
		t.Fatalf("len(Responses) = %d, want 2", got)
	}

	for i, want := range []string{"answer x", "answer y"} {
		if got := resp.GetResponses()[i].GetContent(); got != want {
			t.Errorf("Responses[%d].Content = %q, want %q", i, got, want)
		}
	}
}

func TestLMServiceRegistered(t *testing.T) {
	// Verifies the gRPC service descriptors were generated.
	if LMService_ServiceDesc.ServiceName != "rlm.v1.LMService" {
		t.Errorf("LMService service name = %q, want %q", LMService_ServiceDesc.ServiceName, "rlm.v1.LMService")
	}
	if RLMService_ServiceDesc.ServiceName != "rlm.v1.RLMService" {
		t.Errorf("RLMService service name = %q, want %q", RLMService_ServiceDesc.ServiceName, "rlm.v1.RLMService")
	}
}

func TestRLMChatCompletionConstruction(t *testing.T) {
	completion := &RLMChatCompletion{
		RootModel:     "llama3.1",
		Prompt:        `{"turn":1}`,
		PromptType:    PromptType_PROMPT_JSON,
		Response:      "ok",
		UsageSummary:  &UsageSummary{},
		ExecutionTime: durationpb.New(10 * time.Millisecond),
	}

	if got := completion.GetRootModel(); got != "llama3.1" {
		t.Errorf("GetRootModel() = %q, want %q", got, "llama3.1")
	}
	if got := completion.GetResponse(); got != "ok" {
		t.Errorf("GetResponse() = %q, want %q", got, "ok")
	}
}
