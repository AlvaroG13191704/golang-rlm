// Package prompt builds system prompts, per-turn user prompts, and query
// metadata for the RLM orchestrator.
package prompt

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"reflect"
	"sort"
	"strings"

	"rlm-golang/internal/types"
)

// Message is a single chat message with a role and content.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// QueryMetadata describes the shape and size of the prompt context.
type QueryMetadata struct {
	ContextLengths     []int
	ContextTotalLength int
	ContextType        string
	Keys               []string
	Previews           []string
}

// RLM_SYSTEM_PROMPT is the default system prompt for the RLM. It mirrors the
// Python reference in rlm/utils/prompts.py. The literal substring
// "{custom_tools_section}" is replaced by BuildSystemPrompt when tools are
// present.
const RLM_SYSTEM_PROMPT = `You are a Recursive Language Model (RLM): a language model with a prompt, and a very important context stored in a Python REPL related to that prompt.
You can iteratively interact with the a Python REPL, which has access to LLM calls as a function. You will be queried turn-by-turn until you have an answer to the query.

To use the REPL, you need to write code in ` + "```repl```" + ` blocks; the REPL persists across turns. Available in the REPL:
- ` + "`context`" + `: the important, potentially very long information related to the prompt (typically ` + "`str`" + ` or ` + "`list[str]`" + `).
- ` + "`df`" + `: if the context is a CSV/TSV file, it is automatically loaded as a pandas DataFrame (pandas is imported as ` + "`pd`" + `, numpy as ` + "`np`" + `).
- ` + "`llm_query(prompt: str, model: str | None = None) -> str`" + `: a single sub-LLM completion. Use for extraction, summarization, or Q&A over a chunk of text. Sub-LLM context window ≈ 500K chars.
- ` + "`llm_query_batched(prompts: list[str], model=None) -> list[str]`" + `: concurrently call several LLM calls in parallel over a list of prompts; same order out as in.
- ` + "`rlm_query(prompt, model=None)`" + ` / ` + "`rlm_query_batched(prompts, model=None)`" + `: recursive RLM sub-calls. Fall back to ` + "`llm_query`" + ` / ` + "`llm_query_batched`" + ` when recursion is disabled.
- ` + "`SHOW_VARS() -> str`" + `: list every variable currently in the REPL.
- ` + "`answer`" + `: dict initialized to ` + "`{\"content\": \"\", \"ready\": False}`" + `. To submit, set ` + "`answer[\"content\"]`" + ` to the final answer and ` + "`answer[\"ready\"] = True`" + ` inside a ` + "```repl```" + ` block.
{custom_tools_section}

REPL outputs over ~20K characters are truncated, so for longer payloads slice ` + "`context`" + ` and pass slices through ` + "`llm_query`" + ` rather than ` + "`print`" + `-ing them whole. The REPL is NOT a Jupyter cell — only ` + "`print(...)`" + ` output (stdout) is shown back to you between turns; a bare expression on the last line is silently discarded. Always wrap inspections in ` + "`print(...)`" + `.

As a general strategy, you should start by probing your context to understand it better (e.g. print a few lines, count them, etc.). Then, use the REPL to build up an answer to the query.

An example strategy is to first look at the context and figure out a chunking strategy, then break up the context into chunks, query an LLM per chunk, and save the answers to a buffer. For example:
` + "```repl\n" + `chunk = context[:10000]
answer = llm_query(f"What is the magic number in the context? Here is the chunk: {chunk}")
print(answer)
` + "```\n\n" + `As another example, suppose you are iterating through a list of sections:
` + "```repl\n" + `query = "did Gryffindor win?"
for i, section in enumerate(context):
    if i == len(context) - 1:
        buffer = llm_query(f"You are on the last section. Gather from this to answer {query}. Section: {section}")
        print(f"Based on reading iteratively, the answer is: {buffer}")
    else:
        buffer = llm_query(f"You are iteratively looking through sections. Section {i} of {len(context)}. Gather info to answer {query}. Section: {section}")
        print(f"After section {i}, tracked: {buffer}")
` + "```\n\n" + `Plan in prose, then execute one ` + "```repl```" + ` block every turn, get feedback from the output, then continue on the next turn. Do not flip ` + "`answer[\"ready\"] = True`" + ` on turn 1 without first inspecting ` + "`context`" + `.`

// ORCHESTRATOR_ADDENDUM is appended to the system prompt when the orchestrator
// flag is true. It mirrors the Python reference.
const ORCHESTRATOR_ADDENDUM = `As an RLM, you should act as an orchestrator, not a solver.

Directly after you probe the ` + "`context`" + ` and understand your task, pause and plan: state explicitly how the task decomposes into sub-LLM / REPL steps, and sketch the concrete sequence of turns — what each turn computes and which sub-LLM call (if any) it issues — like a condensed trajectory, before you execute them. Then execute one turn at a time: after each step ` + "`print`" + ` a small sample of the result, verify it looks right, and only flip ` + "`answer[\"ready\"] = True`" + ` once you have actually printed the candidate answer. If you are running out of turns without a confirmed answer, submit your best inference rather than letting the rollout terminate unsubmitted.

Your own context window is small. Push every long-context operation that would not fit comfortably in your own working window — reading, summarizing, classifying, verifying, answering sub-questions, even recapping your own progress — into ` + "`llm_query`" + ` / ` + "`llm_query_batched`" + ` calls instead of pulling that text into your own message stream. (Conversely: if a Python keyword / regex search over ` + "`context`" + ` would already pin the answer, or if a single visible passage already contains it, just read it directly — sub-LMs are for when the raw text won't fit or the question needs semantic interpretation.) Long REPL stdout pollutes history the same way raw ` + "`context`" + ` does: if you want a recap, ask ` + "`llm_query`" + ` for a 1–2 sentence summary and ` + "`print`" + ` only that. Aggregate the small results back in the REPL.

Sub-LLMs have no REPL; they only see the prompt and the ` + "`context`" + ` slice you pass them. Hand them clean, focused inputs and explicitly instruct them to be extremely concise. Sub-LLMs tend to regurgitate context if not strictly constrained, which wastes budget and causes timeouts. Ask for terse, structured outputs you can manipulate programmatically.

***Choosing between 'llm_query' and 'rlm_query'***:
- Use ` + "`llm_query(prompt)`" + ` for straightforward sub-tasks like summarization, extraction, or answering a question about a chunk.
- Use ` + "`rlm_query(prompt)`" + ` for **complex sub-tasks** that benefit from iterative, multi-step reasoning. This spawns a full RLM_REPL loop (with its own REPL environment, sub-LLM calls, and iterative code execution) to analyze the given context. Use this when a sub-task itself requires chunking, aggregation, or multi-step analysis.

IMPORTANT: Be very careful about using 'llm_query' as it incurs high runtime costs. Always batch as much information as reasonably possible into each call (aim for around 100K-200K characters per call). For example, if you have 1000 lines of information to process, it's much better to split into chunks of 5 and call 'llm_query' on each chunk rather than making 1000 individual calls. Minimize the number of 'llm_query' calls by batching related information together.

Sub-call budget is finite on two independent axes, and ` + "`llm_query_batched`" + ` only parallelizes — it does not relax either. (1) Per-prompt capacity: a single sub-call answers well only when its input stays modestly sized — a useful rough ceiling is ~100K characters per prompt, less when the text is dense. Pack each prompt close to that capacity (a chunk of many items, a whole document) so one call accomplishes a lot of work. (2) Per-batch fan-out: ` + "`llm_query_batched`" + ` concurrency is bounded too — a useful rough ceiling is ~20 prompts per batch. Tiny-prompt mega-batches (hundreds or thousands of single-item prompts) are the anti-pattern; fat-prompt small batches are correct. For many independent units, use several ~20-wide batches of full-capacity prompts in sequence, not one mega-batch of tiny prompts. When the work can be expressed either as a sequential loop of ` + "`llm_query`" + `s or as one comparably-sized batched call, prefer batched — same total work, far fewer turns burned. After Python-side filtering has narrowed the candidate set, batch-extract the survivors rather than reading them by hand. If the raw workload exceeds both budgets at once (e.g. a context far larger than ~20 × 100K chars), don't brute-force it: filter aggressively in Python first to a tractable subset, or stage the task — a cheap coarse pass narrows candidates, then a targeted second pass extracts from the survivors.

Reserve your own tokens for high-level decisions: what to ask next, how to combine sub-LM outputs, when to finalize. Delegate everything else.`

const defaultMaxIterations = 30

// NewQueryMetadata computes the type and length metadata for a prompt. It
// supports strings, maps, and lists, matching the Python reference behavior.
func NewQueryMetadata(prompt any) (QueryMetadata, error) {
	var meta QueryMetadata

	switch p := prompt.(type) {
	case types.DirectoryContext:
		meta.ContextType = "directory"
		err := filepath.WalkDir(p.Path, func(path string, d os.DirEntry, err error) error {
			if err != nil {
				return err
			}
			if d.IsDir() {
				return nil
			}
			info, err := d.Info()
			if err != nil {
				return err
			}
			relPath, err := filepath.Rel(p.Path, path)
			if err != nil {
				return err
			}
			meta.Keys = append(meta.Keys, relPath)
			meta.ContextLengths = append(meta.ContextLengths, int(info.Size()))

			data, err := os.ReadFile(path)
			var preview string
			if err == nil {
				preview = string(data)
				if len(preview) > 100 {
					preview = preview[:100] + "..."
				}
				preview = strings.ReplaceAll(preview, "\n", " ")
				preview = strings.ReplaceAll(preview, "\r", " ")
			}
			meta.Previews = append(meta.Previews, preview)

			return nil
		})
		if err != nil {
			return meta, fmt.Errorf("read directory context metadata: %w", err)
		}
	case string:
		meta.ContextType = "str"
		meta.ContextLengths = []int{len(p)}
	case map[string]any:
		meta.ContextType = "dict"
		keys := make([]string, 0, len(p))
		for k := range p {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		meta.Keys = keys
		for _, k := range keys {
			meta.ContextLengths = append(meta.ContextLengths, lengthOf(p[k]))

			preview := toString(p[k])
			if len(preview) > 100 {
				preview = preview[:100] + "..."
			}
			preview = strings.ReplaceAll(preview, "\n", " ")
			preview = strings.ReplaceAll(preview, "\r", " ")
			meta.Previews = append(meta.Previews, preview)
		}
	case []any:
		meta.ContextType = "list"
		if len(p) == 0 {
			meta.ContextLengths = []int{0}
			break
		}
		if first, ok := p[0].(map[string]any); ok && hasContentKey(first) {
			for _, item := range p {
				if m, ok := item.(map[string]any); ok {
					meta.ContextLengths = append(meta.ContextLengths, len(toString(m["content"])))
				} else {
					meta.ContextLengths = append(meta.ContextLengths, lengthOf(item))
				}
			}
		} else if _, ok := p[0].(map[string]any); ok {
			for _, item := range p {
				meta.ContextLengths = append(meta.ContextLengths, lengthOf(item))
			}
		} else {
			for _, item := range p {
				meta.ContextLengths = append(meta.ContextLengths, len(toString(item)))
			}
		}
	default:
		return meta, fmt.Errorf("unsupported prompt type %T", prompt)
	}

	for _, n := range meta.ContextLengths {
		meta.ContextTotalLength += n
	}
	return meta, nil
}

// BuildSystemPrompt combines the system prompt template, optional custom tools,
// the orchestrator addendum, and query metadata into an initial message list.
func BuildSystemPrompt(systemPrompt string, meta QueryMetadata, customTools map[string]any, rootPrompt string, orchestrator bool) ([]Message, error) {
	toolsSection := formatToolsForPrompt(customTools)
	finalSystemPrompt := strings.Replace(systemPrompt, "{custom_tools_section}", toolsSection, 1)
	if orchestrator {
		finalSystemPrompt = finalSystemPrompt + "\n\n" + ORCHESTRATOR_ADDENDUM
	}

	var metadataBody string
	if meta.ContextType == "directory" {
		metadataBody = fmt.Sprintf(
			"Your context is mounted as a directory at '/workspace/context/' containing %d total bytes. Each sub-LLM call can handle roughly ~100k tokens at once.",
			meta.ContextTotalLength,
		)
	} else {
		metadataBody = fmt.Sprintf(
			"Your context is a %s of %d total characters. Each sub-LLM call can handle roughly ~100k tokens at once.",
			meta.ContextType, meta.ContextTotalLength,
		)
	}
	if len(meta.Keys) > 0 {
		var fileList []string
		for i, key := range meta.Keys {
			if len(meta.Previews) > i && meta.Previews[i] != "" {
				fileList = append(fileList, fmt.Sprintf("- %s (preview: %q)", key, meta.Previews[i]))
			} else {
				fileList = append(fileList, fmt.Sprintf("- %s", key))
			}
		}
		if meta.ContextType == "directory" {
			metadataBody += fmt.Sprintf("\nAvailable files mounted at /workspace/context/:\n%s", strings.Join(fileList, "\n"))
		} else {
			metadataBody += fmt.Sprintf("\nAvailable context keys:\n%s", strings.Join(fileList, "\n"))
		}
	}
	var metadataPrompt string
	if rootPrompt != "" {
		metadataPrompt = fmt.Sprintf("Answer the following: %s\n\n%s", rootPrompt, metadataBody)
	} else {
		metadataPrompt = metadataBody
	}

	return []Message{
		{Role: "system", Content: finalSystemPrompt},
		{Role: "user", Content: metadataPrompt},
	}, nil
}

// BuildUserPrompt creates a per-turn user message. The first turn includes a
// safeguard reminding the model to inspect the context before answering.
func BuildUserPrompt(rootPrompt string, iteration, contextCount, historyCount, maxIterations int) Message {
	iter := iteration + 1
	body := fmt.Sprintf("Turn %d/%d:", iter, maxIterations)

	var content string
	if iteration == 0 {
		safeguard := "You have not interacted with the REPL environment or seen your prompt / context yet. Look at the context first; do not provide a final answer yet.\n\n"
		content = safeguard + body
	} else {
		content = body
	}

	if contextCount > 1 {
		content += fmt.Sprintf("\n\nNote: You have %d contexts available (context_0 through context_%d).", contextCount, contextCount-1)
	}
	if historyCount > 0 {
		if historyCount == 1 {
			content += "\n\nNote: You have 1 prior conversation history available in the `history` variable."
		} else {
			content += fmt.Sprintf("\n\nNote: You have %d prior conversation histories available (history_0 through history_%d).", historyCount, historyCount-1)
		}
	}

	return Message{Role: "user", Content: content}
}

func formatToolsForPrompt(customTools map[string]any) string {
	if len(customTools) == 0 {
		return ""
	}

	names := make([]string, 0, len(customTools))
	for name := range customTools {
		names = append(names, name)
	}
	sort.Strings(names)

	var lines []string
	for _, name := range names {
		value, description := parseToolEntry(customTools[name])
		if description != "" {
			lines = append(lines, fmt.Sprintf("- `%s`: %s", name, description))
			continue
		}
		if isCallable(value) {
			lines = append(lines, fmt.Sprintf("- `%s`: A custom function", name))
		} else {
			lines = append(lines, fmt.Sprintf("- `%s`: A custom %T value", name, value))
		}
	}

	if len(lines) == 0 {
		return ""
	}
	return "\n6. Custom tools and data available in the REPL:\n" + strings.Join(lines, "\n")
}

func parseToolEntry(entry any) (value any, description string) {
	if m, ok := entry.(map[string]any); ok {
		if v, has := m["tool"]; has {
			if d, ok := m["description"].(string); ok {
				return v, d
			}
			return v, ""
		}
	}
	return entry, ""
}

func isCallable(v any) bool {
	if v == nil {
		return false
	}
	return reflect.TypeOf(v).Kind() == reflect.Func
}

func hasContentKey(m map[string]any) bool {
	_, ok := m["content"]
	return ok
}

func lengthOf(v any) int {
	if s, ok := v.(string); ok {
		return len(s)
	}
	b, err := json.Marshal(v)
	if err != nil {
		return len(toString(v))
	}
	return len(b)
}

func toString(v any) string {
	if s, ok := v.(string); ok {
		return s
	}
	if v == nil {
		return ""
	}
	return fmt.Sprint(v)
}
