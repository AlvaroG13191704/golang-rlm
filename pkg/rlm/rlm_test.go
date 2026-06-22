package rlm_test

import (
	"context"
	"strings"
	"testing"

	"rlm-golang/pkg/rlm"
)

func TestNewReturnsUsableRLM(t *testing.T) {
	r, err := rlm.New()
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if r == nil {
		t.Fatal("New returned nil RLM")
	}
}

func TestNewWithModel(t *testing.T) {
	r, err := rlm.New(rlm.WithModel("llama3.1"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	// Without a real environment a completion must fail, but the RLM should
	// still be constructed with the requested model.
	_, err = r.Completion(context.Background(), "hello")
	if err == nil {
		t.Fatal("expected error without environment")
	}
}

func TestNewWithOllamaHost(t *testing.T) {
	r, err := rlm.New(
		rlm.WithModel("qwen2.5"),
		rlm.WithOllamaHost("http://localhost:9999/api"),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = r.Completion(context.Background(), "test")
	if err == nil {
		t.Fatal("expected error without environment")
	}
}

func TestCompletionRequiresEnvironment(t *testing.T) {
	r, err := rlm.New(rlm.WithModel("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = r.Completion(context.Background(), "prompt")
	if err == nil {
		t.Fatal("expected error without environment")
	}
	if !strings.Contains(err.Error(), "environment") {
		t.Errorf("error = %v, want environment-related error", err)
	}
}

func TestCompletionWithContext(t *testing.T) {
	r, err := rlm.New(rlm.WithModel("test"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = r.CompletionWithContext(context.Background(), "prompt", "context")
	if err == nil {
		t.Fatal("expected error without environment")
	}
	if !strings.Contains(err.Error(), "environment") {
		t.Errorf("error = %v, want environment-related error", err)
	}
}

func TestWithLimitsAccepted(t *testing.T) {
	r, err := rlm.New(
		rlm.WithModel("test"),
		rlm.WithMaxDepth(3),
		rlm.WithMaxIterations(10),
		rlm.WithMaxBudget(1.0),
		rlm.WithMaxTimeout(30.0),
		rlm.WithMaxTokens(1000),
		rlm.WithMaxErrors(3),
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, err = r.Completion(context.Background(), "prompt")
	if err == nil {
		t.Fatal("expected error without environment")
	}
}

func TestWithMaxIterationsZero(t *testing.T) {
	r, err := rlm.New(rlm.WithMaxIterations(0))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = r.Completion(context.Background(), "prompt")
}

func TestWithMaxDepthZero(t *testing.T) {
	r, err := rlm.New(rlm.WithMaxDepth(0))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	_, _ = r.Completion(context.Background(), "prompt")
}
