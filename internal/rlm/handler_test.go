package rlm_test

import (
	"context"
	"errors"
	"fmt"
	"net"
	"sync"
	"testing"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/test/bufconn"

	rlmv1 "rlm-golang/gen/rlm/v1"
	"rlm-golang/internal/rlm"
	"rlm-golang/internal/types"
)

const bufSize = 1024 * 1024

type fakeLM struct {
	model string
	fn    func(prompt any) (string, error)
	usage types.ModelUsageSummary
	mu    sync.Mutex
	calls []fakeCall
}

type fakeCall struct {
	prompt any
}

func (f *fakeLM) Completion(ctx context.Context, prompt any) (string, error) {
	f.mu.Lock()
	f.calls = append(f.calls, fakeCall{prompt: prompt})
	f.mu.Unlock()
	if f.fn != nil {
		return f.fn(prompt)
	}
	return "ok", nil
}

func (f *fakeLM) GetUsageSummary() types.ModelUsageSummary { return f.usage }
func (f *fakeLM) GetLastUsage() types.ModelUsageSummary    { return f.usage }

func dialBufconn(lis *bufconn.Listener) (*grpc.ClientConn, error) {
	return grpc.NewClient(
		"passthrough:///bufnet",
		grpc.WithContextDialer(func(context.Context, string) (net.Conn, error) { return lis.Dial() }),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
}

func TestLMServiceCompleteRoutesByModelName(t *testing.T) {
	defaultLM := &fakeLM{model: "llama3.1", fn: func(prompt any) (string, error) { return "default", nil }}
	otherLM := &fakeLM{model: "qwen2.5", fn: func(prompt any) (string, error) { return "other", nil }}

	handler := rlm.NewLMHandler(
		defaultLM,
		rlm.WithDefaultModel("llama3.1"),
		rlm.WithOtherBackend(otherLM, "qwen2.5"),
	)
	handler.RegisterClient("custom", &fakeLM{model: "custom", fn: func(prompt any) (string, error) { return "custom-answer", nil }})

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	resp, err := cli.Complete(context.Background(), &rlmv1.CompleteRequest{
		Prompt: "hi",
		Model:  "custom",
		Depth:  0,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.GetContent() != "custom-answer" {
		t.Errorf("content = %q, want %q", resp.GetContent(), "custom-answer")
	}
	if resp.GetRootModel() != "custom" {
		t.Errorf("root_model = %q, want %q", resp.GetRootModel(), "custom")
	}
}

func TestLMServiceCompleteRoutesByDepth(t *testing.T) {
	defaultLM := &fakeLM{model: "llama3.1", fn: func(prompt any) (string, error) { return "default", nil }}
	otherLM := &fakeLM{model: "qwen2.5", fn: func(prompt any) (string, error) { return "other", nil }}

	handler := rlm.NewLMHandler(
		defaultLM,
		rlm.WithDefaultModel("llama3.1"),
		rlm.WithOtherBackend(otherLM, "qwen2.5"),
	)

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	resp, err := cli.Complete(context.Background(), &rlmv1.CompleteRequest{
		Prompt: "hi",
		Depth:  1,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.GetContent() != "other" {
		t.Errorf("content = %q, want %q", resp.GetContent(), "other")
	}
	if resp.GetRootModel() != "qwen2.5" {
		t.Errorf("root_model = %q, want %q", resp.GetRootModel(), "qwen2.5")
	}
}

func TestLMServiceCompleteFallsBackToDefault(t *testing.T) {
	defaultLM := &fakeLM{model: "llama3.1", fn: func(prompt any) (string, error) { return "default", nil }}

	handler := rlm.NewLMHandler(defaultLM, rlm.WithDefaultModel("llama3.1"))

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	resp, err := cli.Complete(context.Background(), &rlmv1.CompleteRequest{
		Prompt: "hi",
		Depth:  2,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.GetContent() != "default" {
		t.Errorf("content = %q, want %q", resp.GetContent(), "default")
	}
	if resp.GetRootModel() != "llama3.1" {
		t.Errorf("root_model = %q, want %q", resp.GetRootModel(), "llama3.1")
	}
}

func TestLMServiceCompleteBatchedPreservesOrder(t *testing.T) {
	defaultLM := &fakeLM{
		model: "llama3.1",
		fn: func(prompt any) (string, error) {
			return "Answer " + prompt.(string), nil
		},
	}

	handler := rlm.NewLMHandler(defaultLM, rlm.WithDefaultModel("llama3.1"))

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	resp, err := cli.CompleteBatched(context.Background(), &rlmv1.BatchedRequest{
		Items: []*rlmv1.CompleteRequest{
			{Prompt: "A"},
			{Prompt: "B"},
			{Prompt: "C"},
		},
	})
	if err != nil {
		t.Fatalf("CompleteBatched: %v", err)
	}

	if got := len(resp.GetResponses()); got != 3 {
		t.Fatalf("len(responses) = %d, want 3", got)
	}
	for i, want := range []string{"Answer A", "Answer B", "Answer C"} {
		if got := resp.GetResponses()[i].GetContent(); got != want {
			t.Errorf("Responses[%d].Content = %q, want %q", i, got, want)
		}
	}
}

func TestLMServiceCompleteBatchedPartialFailure(t *testing.T) {
	defaultLM := &fakeLM{
		model: "llama3.1",
		fn: func(prompt any) (string, error) {
			if prompt.(string) == "B" {
				return "", errors.New("boom")
			}
			return "Answer " + prompt.(string), nil
		},
	}

	handler := rlm.NewLMHandler(defaultLM, rlm.WithDefaultModel("llama3.1"))

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	resp, err := cli.CompleteBatched(context.Background(), &rlmv1.BatchedRequest{
		Items: []*rlmv1.CompleteRequest{
			{Prompt: "A"},
			{Prompt: "B"},
			{Prompt: "C"},
		},
	})
	if err != nil {
		t.Fatalf("CompleteBatched: %v", err)
	}

	if got := resp.GetResponses()[0].GetContent(); got != "Answer A" {
		t.Errorf("Responses[0].Content = %q, want %q", got, "Answer A")
	}
	if got := resp.GetResponses()[1].GetError(); got == "" {
		t.Errorf("Responses[1].Error = %q, want non-empty", got)
	}
	if got := resp.GetResponses()[2].GetContent(); got != "Answer C" {
		t.Errorf("Responses[2].Content = %q, want %q", got, "Answer C")
	}
}

func TestLMServiceGetUsageAggregatesClients(t *testing.T) {
	defaultLM := &fakeLM{
		model: "llama3.1",
		usage: types.ModelUsageSummary{TotalCalls: 2, TotalInputTokens: 20, TotalOutputTokens: 10},
	}
	otherLM := &fakeLM{
		model: "qwen2.5",
		usage: types.ModelUsageSummary{TotalCalls: 1, TotalInputTokens: 5, TotalOutputTokens: 5},
	}

	handler := rlm.NewLMHandler(
		defaultLM,
		rlm.WithDefaultModel("llama3.1"),
		rlm.WithOtherBackend(otherLM, "qwen2.5"),
	)

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	resp, err := cli.GetUsage(context.Background(), &rlmv1.GetUsageRequest{})
	if err != nil {
		t.Fatalf("GetUsage: %v", err)
	}

	models := resp.GetUsage().GetModelUsageSummaries()
	if got := len(models); got != 2 {
		t.Fatalf("len(models) = %d, want 2", got)
	}
	if got := models["llama3.1"].GetTotalCalls(); got != 2 {
		t.Errorf("llama3.1 calls = %d, want 2", got)
	}
	if got := models["qwen2.5"].GetTotalCalls(); got != 1 {
		t.Errorf("qwen2.5 calls = %d, want 1", got)
	}
}

func TestLMServiceCompleteBatchedRespectsConcurrencyLimit(t *testing.T) {
	const delay = 100 * time.Millisecond
	defaultLM := &fakeLM{
		model: "llama3.1",
		fn: func(prompt any) (string, error) {
			time.Sleep(delay)
			return "ok", nil
		},
	}

	handler := rlm.NewLMHandler(
		defaultLM,
		rlm.WithDefaultModel("llama3.1"),
		rlm.WithBatchConcurrency(2),
	)

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	start := time.Now()
	resp, err := cli.CompleteBatched(context.Background(), &rlmv1.BatchedRequest{
		Items: []*rlmv1.CompleteRequest{
			{Prompt: "A"},
			{Prompt: "B"},
			{Prompt: "C"},
			{Prompt: "D"},
		},
	})
	elapsed := time.Since(start)
	if err != nil {
		t.Fatalf("CompleteBatched: %v", err)
	}
	if got := len(resp.GetResponses()); got != 4 {
		t.Fatalf("len(responses) = %d, want 4", got)
	}

	// With concurrency 2 and 4 prompts each taking delay, the wall time should
	// be roughly 2*delay. Allow slack for scheduling but enforce that it did not
	// run sequentially (which would be ~4*delay).
	if elapsed < 2*delay {
		t.Errorf("elapsed = %v, expected at least 2*delay (%v)", elapsed, 2*delay)
	}
	if elapsed > 4*delay {
		t.Errorf("elapsed = %v, expected less than 4*delay (%v)", elapsed, 4*delay)
	}
}

func TestLMHandlerStartStop(t *testing.T) {
	defaultLM := &fakeLM{model: "llama3.1", fn: func(prompt any) (string, error) { return "live", nil }}
	handler := rlm.NewLMHandler(defaultLM, rlm.WithDefaultModel("llama3.1"))

	host, port, err := handler.Start()
	if err != nil {
		t.Fatalf("Start: %v", err)
	}
	defer handler.Stop()

	if host == "" {
		t.Errorf("host is empty")
	}
	if port == 0 {
		t.Errorf("port is zero")
	}

	conn, err := grpc.NewClient(
		net.JoinHostPort(host, fmt.Sprintf("%d", port)),
		grpc.WithTransportCredentials(insecure.NewCredentials()),
	)
	if err != nil {
		t.Fatalf("NewClient: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)
	resp, err := cli.Complete(context.Background(), &rlmv1.CompleteRequest{Prompt: "hi"})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.GetContent() != "live" {
		t.Errorf("content = %q, want %q", resp.GetContent(), "live")
	}
}

func TestLMServiceCompleteJSONPrompt(t *testing.T) {
	defaultLM := &fakeLM{
		model: "llama3.1",
		fn: func(prompt any) (string, error) {
			msgs, ok := prompt.([]any)
			if !ok || len(msgs) != 1 {
				return "", errors.New("expected one message")
			}
			return "got json", nil
		},
	}

	handler := rlm.NewLMHandler(defaultLM, rlm.WithDefaultModel("llama3.1"))

	lis := bufconn.Listen(bufSize)
	s := grpc.NewServer()
	rlmv1.RegisterLMServiceServer(s, handler)
	go func() {
		if err := s.Serve(lis); err != nil {
			t.Logf("server serve: %v", err)
		}
	}()
	defer s.Stop()

	conn, err := dialBufconn(lis)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	cli := rlmv1.NewLMServiceClient(conn)

	resp, err := cli.Complete(context.Background(), &rlmv1.CompleteRequest{
		Prompt:     `[{"role":"user","content":"hi"}]`,
		PromptType: rlmv1.PromptType_PROMPT_JSON,
	})
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if resp.GetContent() != "got json" {
		t.Errorf("content = %q, want %q", resp.GetContent(), "got json")
	}
}
