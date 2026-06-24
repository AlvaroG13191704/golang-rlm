package main

import (
	"bytes"
	"context"
	"errors"
	"os"
	"strings"
	"testing"

	"rlm-golang/pkg/rlm"
)

// mockRLM is a test double returned by the CLI's rlm factory in tests.
type mockRLM struct {
	completionFunc            func(context.Context, string) (*rlm.CompletionResult, error)
	completionWithContextFunc func(context.Context, string, any) (*rlm.CompletionResult, error)
}

func (m *mockRLM) Completion(ctx context.Context, prompt string) (*rlm.CompletionResult, error) {
	return m.completionFunc(ctx, prompt)
}

func (m *mockRLM) CompletionWithContext(ctx context.Context, prompt string, context any) (*rlm.CompletionResult, error) {
	if m.completionWithContextFunc != nil {
		return m.completionWithContextFunc(ctx, prompt, context)
	}
	return m.completionFunc(ctx, prompt)
}

func TestRunWithPromptFlag(t *testing.T) {
	var called bool
	factory := func([]rlm.Option) (rlmInterface, error) {
		return &mockRLM{completionFunc: func(_ context.Context, prompt string) (*rlm.CompletionResult, error) {
			called = true
			if prompt != "hello world" {
				t.Errorf("prompt = %q, want %q", prompt, "hello world")
			}
			return &rlm.CompletionResult{Response: "mocked result"}, nil
		}}, nil
	}

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	err := run([]string{"--prompt", "hello world"}, &bytes.Buffer{}, &stdout, &stderr, factory)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !called {
		t.Fatal("Completion was not called")
	}
	if !strings.Contains(stdout.String(), "mocked result") {
		t.Errorf("stdout = %q, want mocked result", stdout.String())
	}
}

func TestRunWithContextFlag(t *testing.T) {
	contextFile := t.TempDir() + "/context.txt"
	if err := os.WriteFile(contextFile, []byte("file context"), 0o644); err != nil {
		t.Fatalf("write context file: %v", err)
	}

	var called bool
	factory := func([]rlm.Option) (rlmInterface, error) {
		return &mockRLM{
			completionWithContextFunc: func(_ context.Context, prompt string, context any) (*rlm.CompletionResult, error) {
				called = true
				if prompt != "hello world" {
					t.Errorf("prompt = %q, want %q", prompt, "hello world")
				}
				contextStr, ok := context.(string)
				if !ok {
					t.Errorf("expected context to be string, got %T", context)
				}
				if contextStr != "file context" {
					t.Errorf("context = %q, want %q", contextStr, "file context")
				}
				return &rlm.CompletionResult{Response: "context result"}, nil
			},
		}, nil
	}

	var stdout bytes.Buffer
	err := run([]string{"--prompt", "hello world", "--context", contextFile}, &bytes.Buffer{}, &stdout, &bytes.Buffer{}, factory)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !called {
		t.Fatal("CompletionWithContext was not called")
	}
	if !strings.Contains(stdout.String(), "context result") {
		t.Errorf("stdout = %q, want context result", stdout.String())
	}
}

func TestRunWithStdin(t *testing.T) {
	stdin := bytes.NewBufferString("stdin prompt")
	factory := func([]rlm.Option) (rlmInterface, error) {
		return &mockRLM{completionFunc: func(_ context.Context, prompt string) (*rlm.CompletionResult, error) {
			if prompt != "stdin prompt" {
				t.Errorf("prompt = %q, want %q", prompt, "stdin prompt")
			}
			return &rlm.CompletionResult{Response: "ok"}, nil
		}}, nil
	}

	var stdout bytes.Buffer
	err := run([]string{}, stdin, &stdout, &bytes.Buffer{}, factory)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "ok") {
		t.Errorf("stdout = %q, want ok", stdout.String())
	}
}

func TestRunMissingPrompt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	err := run([]string{}, &bytes.Buffer{}, &stdout, &stderr, nil)
	if err == nil {
		t.Fatal("expected error for missing prompt")
	}
	if !strings.Contains(err.Error(), "prompt") {
		t.Errorf("error = %v, want prompt-related error", err)
	}
}

func TestRunCompletionError(t *testing.T) {
	factory := func([]rlm.Option) (rlmInterface, error) {
		return &mockRLM{completionFunc: func(context.Context, string) (*rlm.CompletionResult, error) {
			return nil, errors.New("completion failed")
		}}, nil
	}

	var stdout, stderr bytes.Buffer
	err := run([]string{"--prompt", "hi"}, &bytes.Buffer{}, &stdout, &stderr, factory)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "completion failed") {
		t.Errorf("error = %v, want completion failed", err)
	}
}

func TestRunWithModelFlag(t *testing.T) {
	factory := func([]rlm.Option) (rlmInterface, error) {
		return &mockRLM{completionFunc: func(_ context.Context, prompt string) (*rlm.CompletionResult, error) {
			return &rlm.CompletionResult{Response: "model-result"}, nil
		}}, nil
	}

	var stdout bytes.Buffer
	err := run([]string{"--model", "llama3.1", "--prompt", "hi"}, &bytes.Buffer{}, &stdout, &bytes.Buffer{}, factory)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "model-result") {
		t.Errorf("stdout = %q, want model-result", stdout.String())
	}
}

func TestRunWithLimits(t *testing.T) {
	var capturedOpts []rlm.Option
	factory := func(opts []rlm.Option) (rlmInterface, error) {
		capturedOpts = opts
		return &mockRLM{completionFunc: func(_ context.Context, prompt string) (*rlm.CompletionResult, error) {
			return &rlm.CompletionResult{Response: "limited"}, nil
		}}, nil
	}

	var stdout bytes.Buffer
	err := run([]string{
		"--prompt", "hi",
		"--max-iterations", "5",
		"--max-depth", "2",
		"--max-errors", "4",
	}, &bytes.Buffer{}, &stdout, &bytes.Buffer{}, factory)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "limited") {
		t.Errorf("stdout = %q, want limited", stdout.String())
	}
	if len(capturedOpts) != 5 {
		t.Errorf("len(capturedOpts) = %d, want 5", len(capturedOpts))
	}
}

func TestRunFactoryError(t *testing.T) {
	factory := func([]rlm.Option) (rlmInterface, error) {
		return nil, errors.New("factory error")
	}

	err := run([]string{"--prompt", "hi"}, &bytes.Buffer{}, &bytes.Buffer{}, &bytes.Buffer{}, factory)
	if err == nil {
		t.Fatal("expected error")
	}
	if !strings.Contains(err.Error(), "factory error") {
		t.Errorf("error = %v, want factory error", err)
	}
}

func TestRunWithOllamaHost(t *testing.T) {
	factory := func([]rlm.Option) (rlmInterface, error) {
		return &mockRLM{completionFunc: func(_ context.Context, prompt string) (*rlm.CompletionResult, error) {
			return &rlm.CompletionResult{Response: "host-ok"}, nil
		}}, nil
	}

	var stdout bytes.Buffer
	err := run([]string{
		"--prompt", "hi",
		"--ollama-host", "http://example.com/api",
	}, &bytes.Buffer{}, &stdout, &bytes.Buffer{}, factory)
	if err != nil {
		t.Fatalf("run: %v", err)
	}
	if !strings.Contains(stdout.String(), "host-ok") {
		t.Errorf("stdout = %q, want host-ok", stdout.String())
	}
}
