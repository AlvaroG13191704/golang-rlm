package server_test

import (
	"context"
	"errors"
	"net"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	rlmv1 "rlm-golang/gen/rlm/v1"
	"rlm-golang/internal/server"
	"rlm-golang/internal/types"
)

const bufSize = 1024 * 1024

type fakeSubcallHandler struct {
	mu    sync.Mutex
	calls []subcallCall
	fn    func(prompt any, model string, depth int32) (types.RLMChatCompletion, error)
}

type subcallCall struct {
	prompt any
	model  string
	depth  int32
}

func (f *fakeSubcallHandler) Subcall(ctx context.Context, prompt any, model string, depth int32) (types.RLMChatCompletion, error) {
	f.mu.Lock()
	f.calls = append(f.calls, subcallCall{prompt: prompt, model: model, depth: depth})
	f.mu.Unlock()
	if f.fn != nil {
		return f.fn(prompt, model, depth)
	}
	return types.RLMChatCompletion{Response: "subcall ok", RootModel: model}, nil
}

func dialBufconn(lis *bufconn.Listener) (*grpc.ClientConn, error) {
	return grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return lis.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
}

func TestRLMServiceSubcall(t *testing.T) {
	handler := &fakeSubcallHandler{
		fn: func(prompt any, model string, depth int32) (types.RLMChatCompletion, error) {
			return types.RLMChatCompletion{
				Response:  "child answer",
				RootModel: "qwen2.5",
				UsageSummary: types.UsageSummary{
					ModelUsageSummaries: map[string]types.ModelUsageSummary{
						"qwen2.5": {TotalCalls: 1, TotalInputTokens: 3, TotalOutputTokens: 2},
					},
				},
				ExecutionTime: 50 * time.Millisecond,
			}, nil
		},
	}

	rlmHandler := server.NewRLMHandler(handler, server.WithSubcallConcurrency(2))
	srv := server.NewServer(nil, rlmHandler)

	lis := bufconn.Listen(bufSize)
	if err := srv.RegisterAndStart(lis); err != nil {
		t.Fatalf("RegisterAndStart: %v", err)
	}
	defer srv.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewRLMServiceClient(conn)
	resp, err := cli.Subcall(context.Background(), &rlmv1.SubcallRequest{
		Prompt: "child task",
		Model:  "qwen2.5",
		Depth:  2,
	})
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if resp.GetContent() != "child answer" {
		t.Errorf("content = %q, want %q", resp.GetContent(), "child answer")
	}
	if resp.GetRootModel() != "qwen2.5" {
		t.Errorf("root_model = %q, want %q", resp.GetRootModel(), "qwen2.5")
	}
	if got := resp.GetUsage().GetTotalCalls(); got != 1 {
		t.Errorf("usage calls = %d, want 1", got)
	}
}

func TestRLMServiceSubcallBatchedPreservesOrder(t *testing.T) {
	handler := &fakeSubcallHandler{
		fn: func(prompt any, model string, depth int32) (types.RLMChatCompletion, error) {
			return types.RLMChatCompletion{Response: "answer " + prompt.(string), RootModel: model}, nil
		},
	}

	rlmHandler := server.NewRLMHandler(handler, server.WithSubcallConcurrency(2))
	srv := server.NewServer(nil, rlmHandler)

	lis := bufconn.Listen(bufSize)
	if err := srv.RegisterAndStart(lis); err != nil {
		t.Fatalf("RegisterAndStart: %v", err)
	}
	defer srv.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewRLMServiceClient(conn)
	resp, err := cli.SubcallBatched(context.Background(), &rlmv1.BatchedSubcallRequest{
		Items: []*rlmv1.SubcallRequest{
			{Prompt: "A", Model: "m1"},
			{Prompt: "B", Model: "m1"},
			{Prompt: "C", Model: "m1"},
		},
	})
	if err != nil {
		t.Fatalf("SubcallBatched: %v", err)
	}

	if got := len(resp.GetResponses()); got != 3 {
		t.Fatalf("len(responses) = %d, want 3", got)
	}
	for i, want := range []string{"answer A", "answer B", "answer C"} {
		if got := resp.GetResponses()[i].GetContent(); got != want {
			t.Errorf("Responses[%d].Content = %q, want %q", i, got, want)
		}
	}
}

func TestRLMServiceSubcallBatchedPartialFailure(t *testing.T) {
	handler := &fakeSubcallHandler{
		fn: func(prompt any, model string, depth int32) (types.RLMChatCompletion, error) {
			if prompt.(string) == "B" {
				return types.RLMChatCompletion{}, errors.New("subcall failed")
			}
			return types.RLMChatCompletion{Response: "answer " + prompt.(string), RootModel: model}, nil
		},
	}

	rlmHandler := server.NewRLMHandler(handler)
	srv := server.NewServer(nil, rlmHandler)

	lis := bufconn.Listen(bufSize)
	if err := srv.RegisterAndStart(lis); err != nil {
		t.Fatalf("RegisterAndStart: %v", err)
	}
	defer srv.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewRLMServiceClient(conn)
	resp, err := cli.SubcallBatched(context.Background(), &rlmv1.BatchedSubcallRequest{
		Items: []*rlmv1.SubcallRequest{
			{Prompt: "A"},
			{Prompt: "B"},
			{Prompt: "C"},
		},
	})
	if err != nil {
		t.Fatalf("SubcallBatched: %v", err)
	}

	if got := resp.GetResponses()[0].GetContent(); got != "answer A" {
		t.Errorf("Responses[0].Content = %q, want %q", got, "answer A")
	}
	if got := resp.GetResponses()[1].GetError(); got == "" {
		t.Errorf("Responses[1].Error = %q, want non-empty", got)
	}
	if got := resp.GetResponses()[2].GetContent(); got != "answer C" {
		t.Errorf("Responses[2].Content = %q, want %q", got, "answer C")
	}
}

func TestRLMServiceSubcallJSONPrompt(t *testing.T) {
	handler := &fakeSubcallHandler{
		fn: func(prompt any, model string, depth int32) (types.RLMChatCompletion, error) {
			_, ok := prompt.([]any)
			if !ok {
				return types.RLMChatCompletion{}, errors.New("expected JSON array prompt")
			}
			return types.RLMChatCompletion{Response: "json ok", RootModel: model}, nil
		},
	}

	rlmHandler := server.NewRLMHandler(handler)
	srv := server.NewServer(nil, rlmHandler)

	lis := bufconn.Listen(bufSize)
	if err := srv.RegisterAndStart(lis); err != nil {
		t.Fatalf("RegisterAndStart: %v", err)
	}
	defer srv.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewRLMServiceClient(conn)
	resp, err := cli.Subcall(context.Background(), &rlmv1.SubcallRequest{
		Prompt:     `[{"role":"user","content":"hi"}]`,
		PromptType: rlmv1.PromptType_PROMPT_JSON,
	})
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if resp.GetContent() != "json ok" {
		t.Errorf("content = %q, want %q", resp.GetContent(), "json ok")
	}
}

func TestServerWiresBothLMAndRLMServices(t *testing.T) {
	lmHandler := &fakeLMService{}
	rlmHandler := server.NewRLMHandler(&fakeSubcallHandler{
		fn: func(prompt any, model string, depth int32) (types.RLMChatCompletion, error) {
			return types.RLMChatCompletion{Response: "rlm", RootModel: model}, nil
		},
	})

	srv := server.NewServer(lmHandler, rlmHandler)

	lis := bufconn.Listen(bufSize)
	if err := srv.RegisterAndStart(lis); err != nil {
		t.Fatalf("RegisterAndStart: %v", err)
	}
	defer srv.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	lmCli := rlmv1.NewLMServiceClient(conn)
	lmResp, err := lmCli.Complete(context.Background(), &rlmv1.CompleteRequest{Prompt: "hi"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if lmResp.GetContent() != "lm" {
		t.Errorf("lm content = %q, want %q", lmResp.GetContent(), "lm")
	}

	rlmCli := rlmv1.NewRLMServiceClient(conn)
	rlmResp, err := rlmCli.Subcall(context.Background(), &rlmv1.SubcallRequest{Prompt: "go"})
	if err != nil {
		t.Fatalf("Subcall: %v", err)
	}
	if rlmResp.GetContent() != "rlm" {
		t.Errorf("rlm content = %q, want %q", rlmResp.GetContent(), "rlm")
	}
}

type fakeLMService struct {
	rlmv1.UnimplementedLMServiceServer
}

func (f *fakeLMService) Complete(ctx context.Context, req *rlmv1.CompleteRequest) (*rlmv1.CompleteResponse, error) {
	return &rlmv1.CompleteResponse{Content: "lm"}, nil
}
