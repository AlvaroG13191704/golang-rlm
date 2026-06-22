# Understanding `rlm-golang`: A Beginner's Guide to Recursive Language Models

This guide explains how `rlm-golang` works from the ground up. It is written for anyone learning about LLM agents, tool use, or recursive reasoning. No prior knowledge of the original Python `rlm` project is required.

## What this project does

`rlm-golang` is a Go implementation of a **Recursive Language Model (RLM)** runtime. In plain English:

> It lets a language model answer questions by writing and running Python code in a sandbox, inspecting the result, and deciding what to do next — repeatedly, up to a configured limit.

The key idea is that the model is **not** given the full answer in its prompt. Instead, it is given:

- A **question** (the prompt).
- A **context** (a document, a dataset, etc.).
- A **Python REPL** running inside a Docker container.
- A set of helper functions: `llm_query`, `rlm_query`, and `answer`.

The model must write Python code to explore the context, call sub-models when needed, and finally submit an answer.

## Why this matters

Most LLM applications today fit one of two patterns:

1. **Single-shot prompting**: ask once, get one answer. Good for short, self-contained questions.
2. **Agent loops**: the model can call tools, observe results, and act again. This is the foundation of systems like this one.

An RLM is a specific kind of agent loop where the "tool" is a full Python REPL with sub-model access. This lets the model:

- Process documents larger than its own context window.
- Verify facts by running code instead of guessing.
- Break hard problems into smaller sub-questions answered by child LLM calls.

## Core concepts

| Concept | Meaning |
|--------|---------|
| **Root prompt** | The user's question. |
| **Context payload** | The document or data the model must reason about. It becomes the `context` variable in the REPL. |
| **REPL** | Read-Eval-Print Loop. A Python interpreter that persists state between turns. |
| **Turn / iteration** | One cycle of: model writes code → code runs → model sees output → model decides next step. |
| **Code block** | A fenced block (`` ```repl ``) in the model's response that contains executable Python code. |
| **Final answer** | When the model sets `answer["content"] = ...` and `answer["ready"] = True`, the loop ends. |
| **Sub-call** | A call to `llm_query` or `rlm_query` from inside the REPL. Lets the model delegate work to another LLM. |
| **Recursion depth** | How many levels of nested RLM calls are allowed. Depth 0 is the outermost call. |

## End-to-end flow

Here is what happens when you run:

```bash
go run ./cmd/rlm \
  --model llama3.1 \
  --context context.txt \
  --prompt "What is the main idea?"
```

### 1. CLI parses flags

`cmd/rlm/main.go` reads `--model`, `--context`, `--prompt`, and other flags. It builds an `rlm.RLM` value through the public API in `pkg/rlm/rlm.go`.

### 2. Public API configures the runtime

`pkg/rlm.New(...)` wires together:

- An **Ollama client** factory.
- A **Docker REPL** factory.
- Limits: max iterations, max depth, etc.

The actual Docker container and LLM handler are not created yet; factories are invoked later.

### 3. Orchestrator starts

`RLM.CompletionWithContext(ctx, prompt, context)` calls the internal orchestrator in `internal/rlm/orchestrator.go`.

The orchestrator:

1. Starts an **LMHandler** (a gRPC server that forwards LLM requests to the Ollama client).
2. Creates a temporary workspace directory on the host.
3. Starts a **Docker container** from the `rlm-sandbox` image.
4. Loads the context file into the container as `/workspace/context.txt` and as the Python variable `context`.

### 4. Iteration loop

For each turn (up to `--max-iterations`):

1. **Build prompt**: the orchestrator builds a system message plus a user message. The system message explains the REPL tools. The user message repeats the question and reminds the model to inspect `context` first.
2. **Call LLM**: the orchestrator sends the message history to the Ollama client (`internal/client/ollama.go`), which calls `/api/chat` on the Ollama server.
3. **Extract code**: the response is scanned for `` ```repl `` blocks.
4. **Run code**: each block is executed inside the Docker container via `docker exec`. The script inside the container is `container/exec_script.py`.
5. **Observe output**: stdout, stderr, and the final `answer` dict are captured.
6. **Check for answer**: if `answer["ready"]` is true, the loop ends and the answer is returned.
7. **Append to history**: the code and its output are added to the conversation history, and the next turn begins.

### 5. Cleanup

When the loop ends (success, max iterations, error, etc.), the orchestrator stops the container and deletes the temporary workspace.

```text
+--------+     +------------------+     +------------------+
|  User  | --> |      CLI         | --> |  pkg/rlm public  |
|        |     |  cmd/rlm/main.go |     |     API          |
+--------+     +------------------+     +------------------+
                                                |
                                                v
                                       +------------------+
                                       |  Orchestrator    |
                                       | internal/rlm/... |
                                       +------------------+
                                                |
                       +------------------------+------------------------+
                       |                                                 |
                       v                                                 v
              +------------------+                              +------------------+
              |   LMHandler      |                              |   Docker REPL    |
              | gRPC -> Ollama   |                              | Python sandbox   |
              +------------------+                              +------------------+
                       |                                                 |
                       v                                                 v
              +------------------+                              +------------------+
              |  Ollama server   |                              |  context var     |
              |  (local/cloud)   |                              |  llm_query()     |
              +------------------+                              |  rlm_query()     |
                                                                |  answer{}        |
                                                                +------------------+
```

## What the model sees

The model does not receive the full context text (unless it is tiny). It receives a system prompt like this (greatly abbreviated):

> You are a Recursive Language Model (RLM). You have access to a Python REPL with these variables and functions:
> - `context`: the important, potentially very long information related to the prompt.
> - `llm_query(prompt)`: a sub-LLM call for extraction, summarization, or Q&A over a chunk.
> - `rlm_query(prompt)`: a recursive child RLM call.
> - `answer`: a dict; set `answer["content"]` and `answer["ready"] = True` to finish.
>
> Your own context window is small. Push long-context work into `llm_query` instead of pulling it into your own message stream.

Then a user message:

> Answer the following: What is the main idea?
>
> Your context is a str of 4,076,632 total characters. Each sub-LLM call can handle roughly ~100k tokens at once.

The model must decide how to explore `context` without reading it all at once.

## How files are organized

```text
rlm-golang/
├── cmd/rlm/main.go              # CLI entry point
├── pkg/rlm/rlm.go               # Public library API
├── internal/
│   ├── client/
│   │   └── ollama.go            # Ollama HTTP client
│   ├── prompt/
│   │   └── prompts.go           # System/user prompt construction
│   ├── rlm/
│   │   ├── orchestrator.go      # Main iteration loop
│   │   ├── handler.go           # LMHandler gRPC server
│   │   └── iteration.go         # Code-block extraction, iteration result types
│   ├── server/
│   │   └── server.go            # gRPC service implementations
│   ├── environment/
│   │   ├── docker.go            # Docker REPL lifecycle
│   │   └── execscript.go        # Inline Python exec script template
│   └── types/
│       └── ...                  # Shared domain types
├── container/
│   ├── Dockerfile               # Docker image for the Python sandbox
│   └── exec_script.py           # Canonical Python exec script
├── proto/rlm/v1/                # gRPC proto definitions
└── gen/rlm/v1/                  # Generated Go gRPC code
```

### File-by-file purpose

| File | Responsibility |
|------|----------------|
| `cmd/rlm/main.go` | Parses flags, initializes logging, calls the public API, prints the answer. |
| `pkg/rlm/rlm.go` | Stable public API. Hides internal wiring. Supports `Completion` and `CompletionWithContext`. |
| `internal/client/ollama.go` | Talks to Ollama's `/api/generate` or `/api/chat`. Tracks token usage. |
| `internal/prompt/prompts.go` | Builds the system prompt and per-turn user messages. Computes context metadata. |
| `internal/rlm/orchestrator.go` | Owns the top-level loop: start services, run iterations, handle answers, clean up. |
| `internal/rlm/handler.go` | Creates and runs the gRPC LMHandler so the REPL can request LLM completions. |
| `internal/rlm/iteration.go` | Parses `` ```repl `` blocks, represents one iteration's result. |
| `internal/server/server.go` | gRPC handlers that receive `LMRequest`/`RLMRequest` from the REPL and forward them. |
| `internal/environment/docker.go` | Starts/stops the Docker container, loads context, executes code blocks. |
| `internal/environment/execscript.go` | Go string/template containing the Python script run inside the container. |
| `container/exec_script.py` | Standalone copy of the same script, used to build the Docker image. |
| `container/Dockerfile` | Multi-stage build: generates gRPC Python stubs, installs runtime deps. |

## The Python sandbox

When the container starts, it runs `exec_script.py`. This script:

1. Starts a small gRPC client that connects back to the host's `LMHandler`.
2. Defines `llm_query(prompt, model=None)` and `rlm_query(prompt, model=None)`.
3. Defines `context`, `answer`, and `SHOW_VARS()`.
4. Reads `/workspace/context.txt` into the `context` variable.
5. Enters a loop waiting for code to execute.

When the Go side runs `docker exec` with a code block, the script executes it in a shared namespace, captures stdout/stderr, and returns a JSON result.

## Communication between Go and Python

There are two channels:

1. **Host → container**: `docker exec` runs Python code and reads JSON from stdout.
2. **Container → host**: a gRPC connection from the Python script to the host's `LMHandler`.

This hybrid approach avoids running a full gRPC server inside the container. The container only needs to act as a gRPC client.

```text
Host machine                              Docker container
+-------------+                           +------------------+
|  RLM main   |                           |  exec_script.py  |
|  (Go)       |                           |  (Python)        |
+------+------+                           +---------+--------+
       |                                            |
       | docker exec                                | gRPC client
       v                                            v
+------+------+                           +---------+--------+
|  Docker     |                           |  LMHandler       |
|  daemon     |                           |  (host gRPC)     |
+-------------+                           +---------+--------+
                                                    |
                                                    | HTTP
                                                    v
                                           +------------------+
                                           |  Ollama server   |
                                           +------------------+
```

## Recursion

`rlm_query` inside the REPL triggers a child RLM call. The child:

- Runs in a new Docker container (a "sibling" container, not nested inside the parent).
- Has its own workspace.
- Gets `max_depth - 1` remaining depth.
- Can itself call `rlm_query` until depth reaches zero, at which point it falls back to `llm_query`.

This is useful when a sub-task itself benefits from multi-turn reasoning.

```text
Depth 0 RLM
   └── calls rlm_query("summarize section A")  -> Depth 1 RLM
          └── calls rlm_query("extract dates") -> Depth 2 RLM
                 └── max_depth reached, uses llm_query instead
```

## Logging

Set `LOG_LEVEL=debug` to see every step. At `info` you get high-level events:

- Container start/stop.
- Each iteration number.
- Code execution success or failure.
- Final answer submission.

At `debug` you also get truncated previews of prompts, code blocks, and responses.

## Common failure modes

| Symptom | Likely cause | Where to look |
|---------|--------------|---------------|
| `no EnvFactory configured` | Public API created without `WithDockerREPL()`. | `pkg/rlm/rlm.go`, `cmd/rlm/main.go` |
| `ModuleNotFoundError: No module named 'google'` | Missing `protobuf` in Docker image. | `container/Dockerfile` |
| `UnicodeDecodeError` | Context file is not valid UTF-8. | `container/exec_script.py` encoding fallback |
| `prompt too long` | Model context window smaller than the system prompt. | Try a model with a larger context window. |
| Container fails to start | Docker not running, or `rlm-sandbox` image not built. | `docker images`, `docker ps` |

## Try it yourself

Build the sandbox image once:

```bash
docker build -t rlm-sandbox -f container/Dockerfile .
```

Run a simple question:

```bash
go run ./cmd/rlm --model llama3.1 --prompt "What is 2 + 2?"
```

Run with a context file:

```bash
echo "The quick brown fox jumps over the lazy dog." > context.txt
go run ./cmd/rlm \
  --model llama3.1 \
  --context context.txt \
  --prompt "What animal is mentioned?"
```

Run with debug logs:

```bash
LOG_LEVEL=debug go run ./cmd/rlm --model llama3.1 --prompt "What is 2 + 2?"
```

## Further reading

- Original RLM paper and Python implementation: see the `rlm/` sibling project.
- Ollama API docs: https://github.com/ollama/ollama/blob/main/docs/api.md
- Go `log/slog`: standard structured logging used throughout this project.
- gRPC in Go: the `gen/rlm/v1` package is generated with `protoc-gen-go-grpc`.

## Summary

`rlm-golang` demonstrates a powerful agent pattern: give a language model a sandbox, a context variable, and the ability to call sub-models, then let it reason step by step. The Go side orchestrates the loop; the Python side executes the model's code; Ollama provides the LLM completions. Everything is wired together with gRPC and Docker.
