package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"io"
	"log/slog"
	"os"
	"strings"

	"rlm-golang/internal/logger"
	"rlm-golang/pkg/rlm"
)

// rlmInterface is the subset of *rlm.RLM used by the CLI. It is extracted so
// tests can inject a mock.
type rlmInterface interface {
	Completion(ctx context.Context, prompt string) (*rlm.CompletionResult, error)
	CompletionWithContext(ctx context.Context, prompt string, context any) (*rlm.CompletionResult, error)
}

func main() {
	if err := run(os.Args[1:], os.Stdin, os.Stdout, os.Stderr, defaultFactory); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func defaultFactory(opts []rlm.Option) (rlmInterface, error) {
	return rlm.New(opts...)
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

	handler := logger.NewColorHandler(os.Stderr, slog.HandlerOptions{Level: level})
	slog.SetDefault(slog.New(handler))
}

func run(args []string, stdin io.Reader, stdout, stderr io.Writer, factory func([]rlm.Option) (rlmInterface, error)) error {
	initLogger()

	fs := flag.NewFlagSet("rlm", flag.ContinueOnError)
	fs.SetOutput(stderr)

	model := fs.String("model", "nemotron-3-ultra:cloud", "Ollama model name")
	promptFlag := fs.String("prompt", "", "Prompt text (alternative: pipe to stdin)")
	contextFlag := fs.String("context", "", "Path to a file containing context text")
	maxIterations := fs.Int("max-iterations", 30, "Maximum REPL iterations")
	maxDepth := fs.Int("max-depth", 2, "Maximum recursion depth")
	maxErrors := fs.Int("max-errors", 3, "Maximum consecutive execution errors before aborting (0 to disable)")
	ollamaHost := fs.String("ollama-host", "", "Ollama host URL")

	if err := fs.Parse(args); err != nil {
		return err
	}

	prompt := *promptFlag
	if prompt == "" {
		data, err := io.ReadAll(stdin)
		if err != nil {
			return fmt.Errorf("read stdin: %w", err)
		}
		prompt = strings.TrimSpace(string(data))
	}

	if prompt == "" {
		return errors.New("prompt is required (use --prompt or pipe to stdin)")
	}

	var contextPayload any
	if *contextFlag != "" {
		info, err := os.Stat(*contextFlag)
		if err != nil {
			return fmt.Errorf("stat context path: %w", err)
		}
		if info.IsDir() {
			contextPayload = rlm.DirectoryContext{Path: *contextFlag}
		} else {
			data, err := os.ReadFile(*contextFlag)
			if err != nil {
				return fmt.Errorf("read context file: %w", err)
			}
			contextPayload = string(data)
		}
	}

	opts := []rlm.Option{
		rlm.WithModel(*model),
		rlm.WithMaxIterations(*maxIterations),
		rlm.WithMaxDepth(*maxDepth),
		rlm.WithDockerREPL(),
		rlm.WithMaxErrors(*maxErrors),
	}
	if *ollamaHost != "" {
		opts = append(opts, rlm.WithOllamaHost(*ollamaHost))
	}

	r, err := factory(opts)
	if err != nil {
		return fmt.Errorf("create RLM: %w", err)
	}

	var result *rlm.CompletionResult
	if contextPayload != nil {
		result, err = r.CompletionWithContext(context.Background(), prompt, contextPayload)
	} else {
		result, err = r.Completion(context.Background(), prompt)
	}
	if err != nil {
		return fmt.Errorf("completion: %w", err)
	}

	fmt.Fprintln(stdout, result.Response)

	var totalCalls, totalInput, totalOutput int
	for _, m := range result.UsageSummary.ModelUsageSummaries {
		totalCalls += m.TotalCalls
		totalInput += m.TotalInputTokens
		totalOutput += m.TotalOutputTokens
	}

	fmt.Fprintln(stdout, "---")
	fmt.Fprintln(stdout, "Token Usage:")
	fmt.Fprintf(stdout, "  Total Calls:         %d\n", totalCalls)
	fmt.Fprintf(stdout, "  Total Input Tokens:  %d\n", totalInput)
	fmt.Fprintf(stdout, "  Total Output Tokens: %d\n", totalOutput)
	fmt.Fprintf(stdout, "  Total Tokens:        %d\n", totalInput+totalOutput)
	return nil
}

