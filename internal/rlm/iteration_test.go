package rlm_test

import (
	"strings"
	"testing"
	"time"

	"rlm-golang/internal/rlm"
)

func TestFindCodeBlocksExtractsReplBlocks(t *testing.T) {
	response := "Let's inspect the context.\n\n```repl\nprint(context[:20])\n```\n\nNow compute:\n\n```repl\nx = 42\nprint(x)\n```\n"
	blocks := rlm.FindCodeBlocks(response)
	if len(blocks) != 2 {
		t.Fatalf("len(blocks) = %d, want 2", len(blocks))
	}
	if got := strings.TrimSpace(blocks[0]); got != "print(context[:20])" {
		t.Errorf("block[0] = %q, want %q", got, "print(context[:20])")
	}
	if got := strings.TrimSpace(blocks[1]); got != "x = 42\nprint(x)" {
		t.Errorf("block[1] = %q, want %q", got, "x = 42\nprint(x)")
	}
}

func TestFindCodeBlocksReturnsEmptyWhenNone(t *testing.T) {
	blocks := rlm.FindCodeBlocks("No code here.")
	if len(blocks) != 0 {
		t.Fatalf("len(blocks) = %d, want 0", len(blocks))
	}
}

func TestFormatIterationNoCodeBlocks(t *testing.T) {
	iter := rlm.RLMIteration{
		Response: "The answer is 42.",
	}
	msgs := rlm.FormatIteration(iter, 20000)
	if len(msgs) != 1 {
		t.Fatalf("len(msgs) = %d, want 1", len(msgs))
	}
	if msgs[0].Role != "assistant" {
		t.Errorf("role = %q, want assistant", msgs[0].Role)
	}
	if msgs[0].Content != iter.Response {
		t.Errorf("content = %q, want %q", msgs[0].Content, iter.Response)
	}
}

func TestFormatIterationWithCodeBlocks(t *testing.T) {
	iter := rlm.RLMIteration{
		Response: "```repl\nprint(42)\n```",
		CodeBlocks: []rlm.CodeBlock{
			{
				Code: "print(42)",
				Result: rlm.REPLResult{
					Stdout: "42\n",
					Locals: map[string]string{"x": "42"},
				},
			},
		},
		IterationTime: time.Millisecond,
	}
	msgs := rlm.FormatIteration(iter, 20000)
	if len(msgs) != 2 {
		t.Fatalf("len(msgs) = %d, want 2", len(msgs))
	}
	if msgs[0].Role != "assistant" {
		t.Errorf("first role = %q, want assistant", msgs[0].Role)
	}
	if msgs[1].Role != "user" {
		t.Errorf("second role = %q, want user", msgs[1].Role)
	}
	if !strings.Contains(msgs[1].Content, "42") {
		t.Errorf("user content missing output: %q", msgs[1].Content)
	}
}

func TestFormatIterationTruncatesLongOutput(t *testing.T) {
	long := strings.Repeat("a", 100)
	iter := rlm.RLMIteration{
		Response: "```repl\nprint(x)\n```",
		CodeBlocks: []rlm.CodeBlock{
			{
				Code:   "print(x)",
				Result: rlm.REPLResult{Stdout: long},
			},
		},
	}
	msgs := rlm.FormatIteration(iter, 20)
	if !strings.Contains(msgs[1].Content, "Output truncated") {
		t.Errorf("long output was not truncated: %q", msgs[1].Content)
	}
	if strings.Count(msgs[1].Content, "a") > 50 {
		t.Errorf("too many 'a' characters preserved: %q", msgs[1].Content)
	}
}
