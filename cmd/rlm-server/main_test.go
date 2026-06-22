package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"mime/multipart"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/gofiber/fiber/v3"
	"rlm-golang/pkg/rlm"
)

// mockRunner is a test double returned by the server's RLM factory in tests.
type mockRunner struct {
	completionFunc func(context.Context, string, string) (*rlm.CompletionResult, error)
}

func (m *mockRunner) CompletionWithContext(ctx context.Context, prompt string, context string) (*rlm.CompletionResult, error) {
	return m.completionFunc(ctx, prompt, context)
}

func unexpectedFactory([]rlm.Option) (completionRunner, error) {
	return nil, errors.New("factory was not expected to be called")
}

func newMultipartRequest(t *testing.T, fields map[string]string, files map[string]fileUpload) (*http.Request, string) {
	t.Helper()
	var b bytes.Buffer
	mw := multipart.NewWriter(&b)

	for k, v := range fields {
		fw, err := mw.CreateFormField(k)
		if err != nil {
			t.Fatalf("create field %q: %v", k, err)
		}
		if _, err := fw.Write([]byte(v)); err != nil {
			t.Fatalf("write field %q: %v", k, err)
		}
	}

	for fieldName, f := range files {
		fw, err := mw.CreateFormFile(fieldName, f.filename)
		if err != nil {
			t.Fatalf("create file %q: %v", fieldName, err)
		}
		if _, err := fw.Write(f.content); err != nil {
			t.Fatalf("write file %q: %v", fieldName, err)
		}
	}

	if err := mw.Close(); err != nil {
		t.Fatalf("close multipart writer: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/api/v1/complete", &b)
	req.Header.Set("Content-Type", mw.FormDataContentType())
	return req, mw.FormDataContentType()
}

type fileUpload struct {
	filename string
	content  []byte
}

func TestHealth(t *testing.T) {
	app := newApp(unexpectedFactory)

	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	resp, err := app.Test(req, fiber.TestConfig{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("read body: %v", err)
	}

	want := `{"status":"ok"}`
	if string(body) != want {
		t.Errorf("body = %q, want %q", string(body), want)
	}
}

func TestCompleteMissingPrompt(t *testing.T) {
	app := newApp(unexpectedFactory)

	req, _ := newMultipartRequest(t, nil, nil)
	resp, err := app.Test(req, fiber.TestConfig{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusBadRequest)
	}

	var got errorResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode body: %v", err)
	}
	if got.Error == "" {
		t.Errorf("error message is empty")
	}
}

func TestCompleteWithPrompt(t *testing.T) {
	factory := func([]rlm.Option) (completionRunner, error) {
		return &mockRunner{
			completionFunc: func(_ context.Context, prompt string, context string) (*rlm.CompletionResult, error) {
				return &rlm.CompletionResult{
					Response:      "mocked answer",
					RootModel:     "llama3.1",
					ExecutionTime: 12345 * time.Millisecond,
					Metadata:      map[string]any{"iterations": 2},
					UsageSummary: rlm.UsageSummary{
						ModelUsageSummaries: map[string]rlm.ModelUsageSummary{
							"llama3.1": {
								TotalCalls:        3,
								TotalInputTokens:  150,
								TotalOutputTokens: 80,
							},
						},
					},
				}, nil
			},
		}, nil
	}

	app := newApp(factory)
	req, _ := newMultipartRequest(t, map[string]string{"prompt": "What is 2 + 2?"}, nil)

	resp, err := app.Test(req, fiber.TestConfig{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	var got completeResponse
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatalf("decode body: %v", err)
	}

	if got.Response != "mocked answer" {
		t.Errorf("response = %q, want %q", got.Response, "mocked answer")
	}
	if got.RootModel != "llama3.1" {
		t.Errorf("root_model = %q, want %q", got.RootModel, "llama3.1")
	}
	if got.ExecutionTimeMS != 12345 {
		t.Errorf("execution_time_ms = %d, want %d", got.ExecutionTimeMS, 12345)
	}
	if got.Iterations != 2 {
		t.Errorf("iterations = %d, want %d", got.Iterations, 2)
	}

	usage, ok := got.Usage.ModelUsageSummaries["llama3.1"]
	if !ok {
		t.Fatalf("missing usage for llama3.1")
	}
	if usage.TotalCalls != 3 {
		t.Errorf("total_calls = %d, want %d", usage.TotalCalls, 3)
	}
	if usage.TotalInputTokens != 150 {
		t.Errorf("total_input_tokens = %d, want %d", usage.TotalInputTokens, 150)
	}
	if usage.TotalOutputTokens != 80 {
		t.Errorf("total_output_tokens = %d, want %d", usage.TotalOutputTokens, 80)
	}
}

func TestCompleteWithContextFile(t *testing.T) {
	var gotPrompt, gotContext string
	factory := func([]rlm.Option) (completionRunner, error) {
		return &mockRunner{
			completionFunc: func(_ context.Context, prompt string, context string) (*rlm.CompletionResult, error) {
				gotPrompt = prompt
				gotContext = context
				return &rlm.CompletionResult{
					Response:  "context result",
					RootModel: "llama3.1",
				}, nil
			},
		}, nil
	}

	app := newApp(factory)
	req, _ := newMultipartRequest(
		t,
		map[string]string{"prompt": "Summarize this"},
		map[string]fileUpload{"context": {filename: "context.txt", content: []byte("file context")}},
	)

	resp, err := app.Test(req, fiber.TestConfig{Timeout: 5 * time.Second})
	if err != nil {
		t.Fatalf("app.Test: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status = %d, want %d", resp.StatusCode, http.StatusOK)
	}

	if gotPrompt != "Summarize this" {
		t.Errorf("prompt = %q, want %q", gotPrompt, "Summarize this")
	}
	if gotContext != "file context" {
		t.Errorf("context = %q, want %q", gotContext, "file context")
	}
}
