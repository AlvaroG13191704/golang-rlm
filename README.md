# rlm-golang

Go port of the **Recursive Language Model (RLM)** inference runtime. A single `Completion()` call spawns a Docker-based Python REPL sandbox, executes an iterative LLM-driven loop, runs Python code dynamically to explore contexts, and returns the final answer produced by the model.

> **Credits / Acknowledgments:** This project is a Go-based implementation inspired by and based on the original [RLM paper and repository](https://github.com/alexzhang13/rlm). All core conceptual credits belong to the original authors.

## Features

- **Sandboxed Python REPL:** Runs code blocks (` ```repl `) securely inside a Docker container.
- **Recursive Reasoning:** Allows the model to spawn child RLM runs (`rlm_query`) or plain LLM queries (`llm_query`) from inside the sandbox to delegate sub-tasks.
- **Rich Context Analytics:** Automatically detects and loads single `.csv`, `.tsv`, or semicolon-separated files as a `pandas.DataFrame` (`df`) inside the REPL, enabling high-performance analysis on large datasets.
- **Directory Context Mounting:** You can mount entire directories into the sandbox. The LLM receives efficient file-previews (the first 100 characters of each file) in its prompt to locate what it needs without token saturation, allowing it to read raw files natively.
- **Structured Logging:** Includes clean structured logging with ANSI colors and automatic truncation of large prompt payloads to prevent terminal saturation.

## Requirements

- **Go 1.26.4+**
- **Docker Engine**
- **Ollama** (running locally or accessible via URL)

## Installation

Clone the repository and build the CLI binary:

```bash
git clone https://github.com/AlvaroG13191704/golang-rlm.git
cd golang-rlm
go build ./...
```

Build the sandbox Docker image once before running:

```bash
docker build -t rlm-sandbox -f container/Dockerfile .
```

## CLI Usage

Run a completion with a prompt:

```bash
go run ./cmd/rlm --model llama3.1 --prompt "What is 2 + 2?"
```

Pipe a prompt from stdin:

```bash
echo "Explain recursion in three sentences." | go run ./cmd/rlm --model llama3.1
```

### Loading Context

You can pass a single file (`.txt`, `.md`, `.csv`) OR an entire directory using the `--context` flag:

```bash
go run ./cmd/rlm \
  --model gemma4:31b-cloud \
  --context my-datasets/ \
  --prompt "Load the fifa dataset and calculate the average player age."
```

*Note: If a single CSV file is supplied, it is pre-loaded into the Python environment as a pandas DataFrame named `df`. If a directory is supplied, its contents are mounted into the sandbox at `/workspace/context/` and the LLM receives file-previews to efficiently locate what it needs to read natively via `open()` or `pd.read_csv()`.*

### CLI Flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `nemotron-3-ultra:cloud` | LLM model name |
| `--prompt` | `""` | Prompt text (alternative: pipe to stdin) |
| `--context` | `""` | Path to a file or directory containing context |
| `--max-iterations` | `30` | Maximum REPL iterations allowed |
| `--max-depth` | `2` | Maximum recursion depth |
| `--ollama-host` | `""` | Ollama base URL |

---

## Library Usage

You can import `rlm-golang` as a library in your Go projects:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "rlm-golang/pkg/rlm"
)

func main() {
    r, err := rlm.New(
        rlm.WithModel("llama3.1"),
        rlm.WithMaxIterations(10),
        rlm.WithMaxDepth(2),
        rlm.WithDockerREPL(),
    )
    if err != nil {
        log.Fatal(err)
    }

    result, err := r.Completion(context.Background(), "What is 2 + 2?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Println(result.Response)
}
```

---

## Architecture

The project is structured into the following layers:

- `cmd/rlm` — Command-line interface.
- `pkg/rlm` — Stable public API wrapper.
- `internal/rlm` — Orchestrator coordination loop and LM handler.
- `internal/client` — Ollama HTTP client backend.
- `internal/environment` — Docker REPL lifecycle and exec script generation.
- `internal/prompt` — Prompt engineering and system/user prompt builders.
- `internal/server` — Host-side gRPC services (`LMService`, `RLMService`) to receive requests from the sandbox.
- `internal/logger` — Custom ANSI colored structured logger.
- `container` — Sandbox Dockerfile and Python execution script.

---

## Testing

Run all unit tests:

```bash
go test -race ./...
```
