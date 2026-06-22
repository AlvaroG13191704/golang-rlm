package main

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/gofiber/fiber/v3"
	"github.com/valyala/fasthttp"
	"rlm-golang/pkg/rlm"
)

// completionRunner is the subset of *rlm.RLM used by the HTTP server. It is
// extracted so tests can inject a mock.
type completionRunner interface {
	CompletionWithContext(ctx context.Context, prompt string, context string) (*rlm.CompletionResult, error)
}

// server holds the HTTP handlers and the factory used to create a completion
// runner per request.
type server struct {
	factory func([]rlm.Option) (completionRunner, error)
}

// completeRequest carries the non-file fields of the multipart completion form.
type completeRequest struct {
	Prompt        string `form:"prompt"`
	Model         string `form:"model"`
	MaxIterations int    `form:"max_iterations"`
	MaxDepth      int    `form:"max_depth"`
	OllamaHost    string `form:"ollama_host"`
}

// completeResponse is the JSON envelope returned on a successful completion.
type completeResponse struct {
	Response        string       `json:"response"`
	RootModel       string       `json:"root_model"`
	ExecutionTimeMS int64        `json:"execution_time_ms"`
	Iterations      int          `json:"iterations"`
	Usage           usageSummary `json:"usage"`
}

// usageSummary mirrors rlm.UsageSummary with explicit JSON tags.
type usageSummary struct {
	ModelUsageSummaries map[string]modelUsageSummary `json:"model_usage_summaries"`
}

// modelUsageSummary mirrors rlm.ModelUsageSummary with explicit JSON tags.
type modelUsageSummary struct {
	TotalCalls        int `json:"total_calls"`
	TotalInputTokens  int `json:"total_input_tokens"`
	TotalOutputTokens int `json:"total_output_tokens"`
}

// errorResponse is the JSON envelope returned on request or runtime errors.
type errorResponse struct {
	Error string `json:"error"`
}

func main() {
	initLogger()

	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	app := newApp(defaultFactory)
	slog.Info("RLM server starting", "port", port)
	if err := app.Listen(":" + port); err != nil {
		slog.Error("server failed", "error", err)
		os.Exit(1)
	}
}

func defaultFactory(opts []rlm.Option) (completionRunner, error) {
	r, err := rlm.New(opts...)
	if err != nil {
		return nil, err
	}
	return r, nil
}

func initLogger() {
	level := slog.LevelInfo
	switch strings.ToLower(os.Getenv("LOG_LEVEL")) {
	case "debug":
		level = slog.LevelDebug
	case "info", "":
		level = slog.LevelInfo
	case "warn":
		level = slog.LevelWarn
	case "error":
		level = slog.LevelError
	default:
		slog.Warn("unknown LOG_LEVEL, defaulting to info", "LOG_LEVEL", os.Getenv("LOG_LEVEL"))
	}

	handler := slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level})
	slog.SetDefault(slog.New(handler))
}

func newApp(factory func([]rlm.Option) (completionRunner, error)) *fiber.App {
	app := fiber.New()

	s := &server{factory: factory}
	app.Get("/health", s.handleHealth)
	app.Post("/api/v1/complete", s.handleComplete)

	return app
}

func (s *server) handleHealth(c fiber.Ctx) error {
	return c.Status(http.StatusOK).JSON(fiber.Map{"status": "ok"})
}

func (s *server) handleComplete(c fiber.Ctx) error {
	var req completeRequest
	if err := c.Bind().WithoutAutoHandling().Body(&req); err != nil {
		return c.Status(http.StatusBadRequest).JSON(errorResponse{Error: "invalid request: " + err.Error()})
	}

	if strings.TrimSpace(req.Prompt) == "" {
		return c.Status(http.StatusBadRequest).JSON(errorResponse{Error: "prompt is required"})
	}

	fileHeader, err := c.FormFile("context")
	if err != nil && !errors.Is(err, fasthttp.ErrMissingFile) {
		return c.Status(http.StatusBadRequest).JSON(errorResponse{Error: "invalid context file: " + err.Error()})
	}

	var contextStr string
	if fileHeader != nil {
		ext := strings.ToLower(filepath.Ext(fileHeader.Filename))
		if ext != ".txt" {
			return c.Status(http.StatusBadRequest).JSON(errorResponse{Error: "context file must be a .txt file"})
		}

		file, err := fileHeader.Open()
		if err != nil {
			return c.Status(http.StatusBadRequest).JSON(errorResponse{Error: "cannot open context file: " + err.Error()})
		}
		defer file.Close()

		data, err := io.ReadAll(file)
		if err != nil {
			return c.Status(http.StatusBadRequest).JSON(errorResponse{Error: "cannot read context file: " + err.Error()})
		}
		contextStr = string(data)
	}

	model := req.Model
	if model == "" {
		model = "llama3.1"
	}
	maxIterations := req.MaxIterations
	if maxIterations <= 0 {
		maxIterations = 30
	}
	maxDepth := req.MaxDepth
	if maxDepth <= 0 {
		maxDepth = 2
	}
	ollamaHost := req.OllamaHost
	if ollamaHost == "" {
		ollamaHost = os.Getenv("OLLAMA_HOST")
	}

	opts := []rlm.Option{
		rlm.WithModel(model),
		rlm.WithMaxIterations(maxIterations),
		rlm.WithMaxDepth(maxDepth),
		rlm.WithDockerREPL(),
	}
	if ollamaHost != "" {
		opts = append(opts, rlm.WithOllamaHost(ollamaHost))
	}

	runner, err := s.factory(opts)
	if err != nil {
		slog.Error("create RLM", "error", err)
		return c.Status(http.StatusInternalServerError).JSON(errorResponse{Error: "failed to create RLM: " + err.Error()})
	}

	result, err := runner.CompletionWithContext(c.Context(), req.Prompt, contextStr)
	if err != nil {
		slog.Error("completion failed", "error", err)
		return c.Status(http.StatusInternalServerError).JSON(errorResponse{Error: "completion failed: " + err.Error()})
	}

	return c.Status(http.StatusOK).JSON(completeResponse{
		Response:        result.Response,
		RootModel:       result.RootModel,
		ExecutionTimeMS: result.ExecutionTime.Milliseconds(),
		Iterations:      iterationsFromResult(result),
		Usage:           toUsageResponse(result.UsageSummary),
	})
}

func iterationsFromResult(result *rlm.CompletionResult) int {
	if result == nil || result.Metadata == nil {
		return 0
	}
	switch v := result.Metadata["iterations"].(type) {
	case int:
		return v
	case int64:
		return int(v)
	case float64:
		return int(v)
	case string:
		if n, err := strconv.Atoi(v); err == nil {
			return n
		}
	}
	return 0
}

func toUsageResponse(u rlm.UsageSummary) usageSummary {
	summaries := make(map[string]modelUsageSummary, len(u.ModelUsageSummaries))
	for name, s := range u.ModelUsageSummaries {
		summaries[name] = modelUsageSummary{
			TotalCalls:        s.TotalCalls,
			TotalInputTokens:  s.TotalInputTokens,
			TotalOutputTokens: s.TotalOutputTokens,
		}
	}
	return usageSummary{ModelUsageSummaries: summaries}
}
