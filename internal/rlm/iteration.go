package rlm

import (
	"fmt"
	"regexp"
	"strings"
	"time"

	"rlm-golang/internal/prompt"
)

const defaultMaxIterationChars = 2000

// CodeBlock pairs model-emitted code with its execution result.
type CodeBlock struct {
	Code   string
	Result REPLResult
}

// RLMIteration captures one prompt → response → execute cycle.
type RLMIteration struct {
	Prompt        any
	Response      string
	CodeBlocks    []CodeBlock
	IterationTime time.Duration
	FinalAnswer   string
}

var replCodeBlockRE = regexp.MustCompile("(?s)```repl\\s*\\n(.*?)\\n```")

// FindCodeBlocks extracts code wrapped in ```repl blocks. It returns the raw
// code content of each block in the order they appear.
func FindCodeBlocks(response string) []string {
	matches := replCodeBlockRE.FindAllStringSubmatch(response, -1)
	blocks := make([]string, 0, len(matches))
	for _, m := range matches {
		if len(m) > 1 {
			blocks = append(blocks, strings.TrimSpace(m[1]))
		}
	}
	return blocks
}

// FormatIteration converts an iteration into messages that extend the chat
// history. It returns a single assistant message when no code was run, or an
// assistant message followed by a user message summarising REPL outputs.
func FormatIteration(iter RLMIteration, maxChars int) []prompt.Message {
	if maxChars <= 0 {
		maxChars = defaultMaxIterationChars
	}

	messages := []prompt.Message{{Role: "assistant", Content: iter.Response}}
	if len(iter.CodeBlocks) == 0 {
		return messages
	}

	parts := make([]string, 0, len(iter.CodeBlocks))
	multi := len(iter.CodeBlocks) > 1
	for i, block := range iter.CodeBlocks {
		result := formatExecutionResult(block.Result, maxChars)
		header := "REPL output:"
		if multi {
			header = fmt.Sprintf("REPL output (block %d):", i+1)
		}
		parts = append(parts, header+"\n"+result)
	}

	messages = append(messages, prompt.Message{
		Role:    "user",
		Content: strings.Join(parts, "\n\n"),
	})
	return messages
}

func formatExecutionResult(result REPLResult, maxChars int) string {
	var parts []string
	if strings.TrimSpace(result.Stdout) != "" {
		if len(result.Stdout) > maxChars {
			preview := result.Stdout[:maxChars]
			warning := fmt.Sprintf("\n\n... [Output truncated. Total length: %d chars. The output is too large to fit in the prompt history. Please store this data in a REPL variable and use 'llm_query' or 'rlm_query' to analyze it programmatically instead of printing it.]", len(result.Stdout))
			parts = append(parts, preview+warning)
		} else {
			parts = append(parts, result.Stdout)
		}
	}
	if strings.TrimSpace(result.Stderr) != "" {
		parts = append(parts, result.Stderr)
	}

	var keys []string
	for k := range result.Locals {
		if !strings.HasPrefix(k, "_") && k != "__builtins__" && k != "__name__" && k != "__doc__" {
			keys = append(keys, k)
		}
	}
	if len(keys) > 0 {
		parts = append(parts, "REPL variables: "+strings.Join(keys, ", "))
	}

	if len(parts) == 0 {
		return "No output"
	}
	return strings.Join(parts, "\n\n")
}
