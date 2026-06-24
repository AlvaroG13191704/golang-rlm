package prompt_test

import (
	"strings"
	"testing"

	"rlm-golang/internal/prompt"
)

func TestNewQueryMetadata(t *testing.T) {
	tests := []struct {
		name        string
		prompt      any
		wantType    string
		wantLengths []int
		wantTotal   int
		wantKeys    []string
		wantErr     bool
	}{
		{
			name:        "string",
			prompt:      "hello",
			wantType:    "str",
			wantLengths: []int{5},
			wantTotal:   5,
		},
		{
			name:        "dict",
			prompt:      map[string]any{"a": "foo", "b": "bar"},
			wantType:    "dict",
			wantLengths: []int{3, 3},
			wantTotal:   6,
			wantKeys:    []string{"a", "b"},
		},
		{
			name:        "list of strings",
			prompt:      []any{"a", "bb"},
			wantType:    "list",
			wantLengths: []int{1, 2},
			wantTotal:   3,
		},
		{
			name:        "empty list",
			prompt:      []any{},
			wantType:    "list",
			wantLengths: []int{0},
			wantTotal:   0,
		},
		{
			name:        "list of content dicts",
			prompt:      []any{map[string]any{"content": "hello"}, map[string]any{"content": "world"}},
			wantType:    "list",
			wantLengths: []int{5, 5},
			wantTotal:   10,
		},
		{
			name:        "dict with non-string value",
			prompt:      map[string]any{"x": 123},
			wantType:    "dict",
			wantLengths: []int{3},
			wantTotal:   3,
			wantKeys:    []string{"x"},
		},
		{
			name:    "unsupported type",
			prompt:  42,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			meta, err := prompt.NewQueryMetadata(tt.prompt)
			if tt.wantErr {
				if err == nil {
					t.Fatalf("expected error")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if meta.ContextType != tt.wantType {
				t.Errorf("ContextType = %q, want %q", meta.ContextType, tt.wantType)
			}
			if len(meta.ContextLengths) != len(tt.wantLengths) {
				t.Fatalf("ContextLengths = %v, want %v", meta.ContextLengths, tt.wantLengths)
			}
			for i := range tt.wantLengths {
				if meta.ContextLengths[i] != tt.wantLengths[i] {
					t.Errorf("ContextLengths[%d] = %d, want %d", i, meta.ContextLengths[i], tt.wantLengths[i])
				}
			}
			if meta.ContextTotalLength != tt.wantTotal {
				t.Errorf("ContextTotalLength = %d, want %d", meta.ContextTotalLength, tt.wantTotal)
			}
			if len(meta.Keys) != len(tt.wantKeys) {
				t.Fatalf("Keys = %v, want %v", meta.Keys, tt.wantKeys)
			}
			for i := range tt.wantKeys {
				if meta.Keys[i] != tt.wantKeys[i] {
					t.Errorf("Keys[%d] = %q, want %q", i, meta.Keys[i], tt.wantKeys[i])
				}
			}
		})
	}
}

func TestBuildSystemPrompt(t *testing.T) {
	t.Run("with orchestrator", func(t *testing.T) {
		meta, err := prompt.NewQueryMetadata(strings.Repeat("a", 100))
		if err != nil {
			t.Fatalf("NewQueryMetadata: %v", err)
		}

		msgs, err := prompt.BuildSystemPrompt(prompt.RLM_SYSTEM_PROMPT, meta, nil, "", true)
		if err != nil {
			t.Fatalf("BuildSystemPrompt: %v", err)
		}
		if len(msgs) != 2 {
			t.Fatalf("expected 2 messages, got %d", len(msgs))
		}
		if msgs[0].Role != "system" {
			t.Errorf("first role = %q, want system", msgs[0].Role)
		}
		if msgs[1].Role != "user" {
			t.Errorf("second role = %q, want user", msgs[1].Role)
		}
		if !strings.Contains(msgs[0].Content, "Recursive Language Model") {
			t.Errorf("system prompt missing identity")
		}
		if !strings.Contains(msgs[0].Content, "act as an orchestrator") {
			t.Errorf("system prompt missing orchestrator addendum")
		}
		if !strings.Contains(msgs[1].Content, "Your context is a str of 100 total characters") {
			t.Errorf("user prompt missing metadata")
		}
	})

	t.Run("with root prompt", func(t *testing.T) {
		meta, _ := prompt.NewQueryMetadata("ctx")
		msgs, err := prompt.BuildSystemPrompt(prompt.RLM_SYSTEM_PROMPT, meta, nil, "summarize", true)
		if err != nil {
			t.Fatalf("BuildSystemPrompt: %v", err)
		}
		wantPrefix := "Answer the following: summarize\n\nYour context is a str"
		if !strings.HasPrefix(msgs[1].Content, wantPrefix) {
			t.Errorf("user content = %q, want prefix %q", msgs[1].Content, wantPrefix)
		}
	})

	t.Run("with dict context keys", func(t *testing.T) {
		meta, err := prompt.NewQueryMetadata(map[string]any{"file1.txt": "foo", "file2.txt": "bar"})
		if err != nil {
			t.Fatalf("NewQueryMetadata: %v", err)
		}
		msgs, err := prompt.BuildSystemPrompt(prompt.RLM_SYSTEM_PROMPT, meta, nil, "", true)
		if err != nil {
			t.Fatalf("BuildSystemPrompt: %v", err)
		}
		wantContains := "Available context keys (files): file1.txt, file2.txt"
		if !strings.Contains(msgs[1].Content, wantContains) {
			t.Errorf("user content = %q, want it to contain %q", msgs[1].Content, wantContains)
		}
	})

	t.Run("without orchestrator", func(t *testing.T) {
		meta, _ := prompt.NewQueryMetadata("ctx")
		msgs, err := prompt.BuildSystemPrompt(prompt.RLM_SYSTEM_PROMPT, meta, nil, "", false)
		if err != nil {
			t.Fatalf("BuildSystemPrompt: %v", err)
		}
		if strings.Contains(msgs[0].Content, "act as an orchestrator") {
			t.Errorf("system prompt should not contain orchestrator addendum")
		}
	})

	t.Run("with custom tools", func(t *testing.T) {
		meta, _ := prompt.NewQueryMetadata("ctx")
		tools := map[string]any{
			"search": map[string]any{"tool": "dummy", "description": "Search the web"},
			"calc":   func() {},
		}
		msgs, err := prompt.BuildSystemPrompt(prompt.RLM_SYSTEM_PROMPT, meta, tools, "", false)
		if err != nil {
			t.Fatalf("BuildSystemPrompt: %v", err)
		}
		if !strings.Contains(msgs[0].Content, "Custom tools and data available in the REPL") {
			t.Errorf("system prompt missing custom tools header")
		}
		if !strings.Contains(msgs[0].Content, "- `search`: Search the web") {
			t.Errorf("system prompt missing search tool")
		}
		if !strings.Contains(msgs[0].Content, "- `calc`: A custom function") {
			t.Errorf("system prompt missing calc tool")
		}
	})
}

func TestBuildUserPrompt(t *testing.T) {
	tests := []struct {
		name          string
		iteration     int
		contextCount  int
		historyCount  int
		maxIterations int
		wantContains  []string
		wantMissing   []string
	}{
		{
			name:          "first turn safeguard",
			iteration:     0,
			contextCount:  1,
			historyCount:  0,
			maxIterations: 5,
			wantContains:  []string{"You have not interacted with the REPL", "Turn 1/5"},
			wantMissing:   []string{"context_0"},
		},
		{
			name:          "later turn no safeguard",
			iteration:     2,
			contextCount:  1,
			historyCount:  0,
			maxIterations: 5,
			wantContains:  []string{"Turn 3/5"},
			wantMissing:   []string{"You have not interacted"},
		},
		{
			name:          "multiple contexts",
			iteration:     0,
			contextCount:  2,
			historyCount:  0,
			maxIterations: 5,
			wantContains:  []string{"You have 2 contexts available (context_0 through context_1)."},
		},
		{
			name:          "single history",
			iteration:     0,
			contextCount:  1,
			historyCount:  1,
			maxIterations: 5,
			wantContains:  []string{"You have 1 prior conversation history available in the `history` variable."},
		},
		{
			name:          "multiple histories",
			iteration:     0,
			contextCount:  1,
			historyCount:  3,
			maxIterations: 5,
			wantContains:  []string{"You have 3 prior conversation histories available (history_0 through history_2)."},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			msg := prompt.BuildUserPrompt("", tt.iteration, tt.contextCount, tt.historyCount, tt.maxIterations)
			if msg.Role != "user" {
				t.Errorf("Role = %q, want user", msg.Role)
			}
			for _, want := range tt.wantContains {
				if !strings.Contains(msg.Content, want) {
					t.Errorf("content missing %q: %q", want, msg.Content)
				}
			}
			for _, miss := range tt.wantMissing {
				if strings.Contains(msg.Content, miss) {
					t.Errorf("content should not contain %q: %q", miss, msg.Content)
				}
			}
		})
	}
}
