package client_test

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"rlm-golang/internal/client"
	"rlm-golang/internal/prompt"
	"rlm-golang/internal/types"
)

func ollamaGenerateHandler(response string, promptEvalCount, evalCount int, totalDur, promptEvalDur, evalDur int64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/generate" {
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req["stream"] != false {
			http.Error(w, "expected stream false", http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"model":                "llama3.1",
			"response":             response,
			"done":                 true,
			"total_duration":       totalDur,
			"prompt_eval_count":    promptEvalCount,
			"eval_count":           evalCount,
			"prompt_eval_duration": promptEvalDur,
			"eval_duration":        evalDur,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}

func ollamaChatHandler(content string, promptEvalCount, evalCount int, totalDur, promptEvalDur, evalDur int64) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			http.NotFound(w, r)
			return
		}
		body, _ := io.ReadAll(r.Body)
		var req map[string]any
		if err := json.Unmarshal(body, &req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req["stream"] != false {
			http.Error(w, "expected stream false", http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"model": "llama3.1",
			"message": map[string]any{
				"role":    "assistant",
				"content": content,
			},
			"done":                 true,
			"total_duration":       totalDur,
			"prompt_eval_count":    promptEvalCount,
			"eval_count":           evalCount,
			"prompt_eval_duration": promptEvalDur,
			"eval_duration":        evalDur,
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}
}

func TestOllamaClientGenerate(t *testing.T) {
	tests := []struct {
		name            string
		response        string
		promptEvalCount int
		evalCount       int
		totalDur        int64
		promptEvalDur   int64
		evalDur         int64
		want            string
		wantUsage       types.ModelUsageSummary
		wantPromptRate  float64
		wantEvalRate    float64
		wantTTFT        time.Duration
	}{
		{
			name:            "happy path",
			response:        "Hello back",
			promptEvalCount: 10,
			evalCount:       5,
			totalDur:        1_234_567_890,
			promptEvalDur:   100_000_000,
			evalDur:         200_000_000,
			want:            "Hello back",
			wantUsage:       types.ModelUsageSummary{TotalCalls: 1, TotalInputTokens: 10, TotalOutputTokens: 5},
			wantPromptRate:  100.0,
			wantEvalRate:    25.0,
			wantTTFT:        100 * time.Millisecond,
		},
		{
			name:            "zero durations guard",
			response:        "OK",
			promptEvalCount: 4,
			evalCount:       0,
			totalDur:        50_000_000,
			promptEvalDur:   0,
			evalDur:         0,
			want:            "OK",
			wantUsage:       types.ModelUsageSummary{TotalCalls: 1, TotalInputTokens: 4, TotalOutputTokens: 0},
			wantPromptRate:  0,
			wantEvalRate:    0,
			wantTTFT:        0,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(ollamaGenerateHandler(tt.response, tt.promptEvalCount, tt.evalCount, tt.totalDur, tt.promptEvalDur, tt.evalDur))
			defer srv.Close()

			c, err := client.NewOllamaClient("llama3.1", srv.URL+"/api", http.DefaultClient)
			if err != nil {
				t.Fatalf("NewOllamaClient: %v", err)
			}

			got, err := c.Completion(context.Background(), "Hello")
			if err != nil {
				t.Fatalf("Completion: %v", err)
			}
			if got != tt.want {
				t.Errorf("Completion() = %q, want %q", got, tt.want)
			}

			usage := c.GetUsageSummary()
			if usage.TotalCalls != tt.wantUsage.TotalCalls {
				t.Errorf("GetUsageSummary().TotalCalls = %d, want %d", usage.TotalCalls, tt.wantUsage.TotalCalls)
			}
			if usage.TotalInputTokens != tt.wantUsage.TotalInputTokens {
				t.Errorf("GetUsageSummary().TotalInputTokens = %d, want %d", usage.TotalInputTokens, tt.wantUsage.TotalInputTokens)
			}
			if usage.TotalOutputTokens != tt.wantUsage.TotalOutputTokens {
				t.Errorf("GetUsageSummary().TotalOutputTokens = %d, want %d", usage.TotalOutputTokens, tt.wantUsage.TotalOutputTokens)
			}
			if usage.TotalCost != nil {
				t.Errorf("GetUsageSummary().TotalCost = %v, want nil", *usage.TotalCost)
			}

			last := c.GetLastUsage()
			if last.TotalInputTokens != tt.wantUsage.TotalInputTokens {
				t.Errorf("GetLastUsage().TotalInputTokens = %d, want %d", last.TotalInputTokens, tt.wantUsage.TotalInputTokens)
			}

			m := c.Metrics()
			if m.TotalDuration != time.Duration(tt.totalDur) {
				t.Errorf("Metrics().TotalDuration = %v, want %v", m.TotalDuration, time.Duration(tt.totalDur))
			}
			if m.TTFT != tt.wantTTFT {
				t.Errorf("Metrics().TTFT = %v, want %v", m.TTFT, tt.wantTTFT)
			}
			if m.PromptEvalRate != tt.wantPromptRate {
				t.Errorf("Metrics().PromptEvalRate = %v, want %v", m.PromptEvalRate, tt.wantPromptRate)
			}
			if m.EvalRate != tt.wantEvalRate {
				t.Errorf("Metrics().EvalRate = %v, want %v", m.EvalRate, tt.wantEvalRate)
			}
		})
	}
}

func TestOllamaClientChat(t *testing.T) {
	srv := httptest.NewServer(ollamaChatHandler("Chat reply", 3, 2, 500_000_000, 50_000_000, 100_000_000))
	defer srv.Close()

	c, err := client.NewOllamaClient("llama3.1", srv.URL+"/api", http.DefaultClient)
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	msgs := []prompt.Message{{Role: "user", Content: "Hi"}}
	got, err := c.Completion(context.Background(), msgs)
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if got != "Chat reply" {
		t.Errorf("Completion() = %q, want %q", got, "Chat reply")
	}

	usage := c.GetUsageSummary()
	if usage.TotalCalls != 1 || usage.TotalInputTokens != 3 || usage.TotalOutputTokens != 2 {
		t.Errorf("unexpected usage: %+v", usage)
	}

	m := c.Metrics()
	if m.PromptEvalRate != 60.0 {
		t.Errorf("Metrics().PromptEvalRate = %v, want 60", m.PromptEvalRate)
	}
	if m.EvalRate != 20.0 {
		t.Errorf("Metrics().EvalRate = %v, want 20", m.EvalRate)
	}
}

func TestOllamaClientNon2xx(t *testing.T) {
	tests := []struct {
		name        string
		status      int
		body        string
		errContains []string
	}{
		{
			name:        "not found",
			status:      http.StatusNotFound,
			body:        "model not found",
			errContains: []string{"404", "model not found"},
		},
		{
			name:        "server error",
			status:      http.StatusInternalServerError,
			body:        "something went wrong",
			errContains: []string{"500", "something went wrong"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				w.WriteHeader(tt.status)
				_, _ = w.Write([]byte(tt.body))
			}))
			defer srv.Close()

			c, err := client.NewOllamaClient("llama3.1", srv.URL+"/api", http.DefaultClient)
			if err != nil {
				t.Fatalf("NewOllamaClient: %v", err)
			}

			_, err = c.Completion(context.Background(), "Hello")
			if err == nil {
				t.Fatalf("expected error")
			}
			errStr := err.Error()
			for _, want := range tt.errContains {
				if !strings.Contains(errStr, want) {
					t.Errorf("error %q does not contain %q", errStr, want)
				}
			}
		})
	}
}

func TestOllamaClientInvalidJSON(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		_, _ = w.Write([]byte("not json"))
	}))
	defer srv.Close()

	c, err := client.NewOllamaClient("llama3.1", srv.URL+"/api", http.DefaultClient)
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	_, err = c.Completion(context.Background(), "Hello")
	if err == nil {
		t.Fatalf("expected error")
	}
	if !strings.Contains(err.Error(), "unmarshal") && !strings.Contains(err.Error(), "decode") {
		t.Errorf("error does not mention JSON: %v", err)
	}
}

func TestOllamaClientMissingModel(t *testing.T) {
	_, err := client.NewOllamaClient("", "http://localhost:11434/api", http.DefaultClient)
	if err == nil {
		t.Fatalf("expected error for missing model")
	}
	if !strings.Contains(err.Error(), "model") {
		t.Errorf("error does not mention model: %v", err)
	}
}

func TestOllamaClientMalformedBaseURL(t *testing.T) {
	_, err := client.NewOllamaClient("llama3.1", "://not-a-url", http.DefaultClient)
	if err == nil {
		t.Fatalf("expected error for malformed base URL")
	}
}

func TestOllamaClientDefaultBaseURLEnv(t *testing.T) {
	srv := httptest.NewServer(ollamaGenerateHandler("env", 1, 1, 1_000_000, 100_000, 200_000))
	defer srv.Close()

	t.Setenv("OLLAMA_HOST", srv.URL+"/api")

	c, err := client.NewOllamaClient("llama3.1", "", http.DefaultClient)
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	got, err := c.Completion(context.Background(), "Hello")
	if err != nil {
		t.Fatalf("Completion: %v", err)
	}
	if got != "env" {
		t.Errorf("Completion() = %q, want %q", got, "env")
	}

	if err := os.Unsetenv("OLLAMA_HOST"); err != nil {
		t.Fatalf("Unsetenv: %v", err)
	}
}

func TestOllamaClientUsageAggregation(t *testing.T) {
	srv := httptest.NewServer(ollamaGenerateHandler("aggregated", 5, 3, 1_000_000, 100_000, 200_000))
	defer srv.Close()

	c, err := client.NewOllamaClient("llama3.1", srv.URL+"/api", http.DefaultClient)
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	for i := 0; i < 3; i++ {
		if _, err := c.Completion(context.Background(), fmt.Sprintf("call %d", i)); err != nil {
			t.Fatalf("Completion %d: %v", i, err)
		}
	}

	usage := c.GetUsageSummary()
	if usage.TotalCalls != 3 {
		t.Errorf("TotalCalls = %d, want 3", usage.TotalCalls)
	}
	if usage.TotalInputTokens != 15 {
		t.Errorf("TotalInputTokens = %d, want 15", usage.TotalInputTokens)
	}
	if usage.TotalOutputTokens != 9 {
		t.Errorf("TotalOutputTokens = %d, want 9", usage.TotalOutputTokens)
	}
}

func TestOllamaClientStreamFalse(t *testing.T) {
	var gotBody map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		if err := json.Unmarshal(body, &gotBody); err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		resp := map[string]any{
			"response":          "ok",
			"done":              true,
			"prompt_eval_count": 1,
			"eval_count":        1,
		}
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer srv.Close()

	c, err := client.NewOllamaClient("llama3.1", srv.URL+"/api", http.DefaultClient)
	if err != nil {
		t.Fatalf("NewOllamaClient: %v", err)
	}

	if _, err := c.Completion(context.Background(), "Hello"); err != nil {
		t.Fatalf("Completion: %v", err)
	}

	if gotBody["stream"] != false {
		t.Errorf("stream = %v, want false", gotBody["stream"])
	}
}
