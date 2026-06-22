# rlm-golang

Go port of the Recursive Language Model (RLM) inference runtime. A single
`Completion()` call spawns a Docker-based Python REPL, runs an iterative LLM
loop, and returns the final answer produced by the model inside the REPL.

## Requirements

- Go 1.26.4+
- Docker Engine
- Ollama running locally (default) or accessible via `--ollama-host`

## Installation

```bash
git clone https://github.com/AlvaroG13191704/golang-rlm.git
cd golang-rlm
go build ./...
```

## Library usage

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

## CLI usage

Run a completion with a prompt flag:

```bash
go run ./cmd/rlm --model llama3.1 --prompt "What is 2 + 2?"
```

Pipe a prompt from stdin:

```bash
echo "Summarize the Go runtime" | go run ./cmd/rlm --model llama3.1
```

Use a remote Ollama host:

```bash
go run ./cmd/rlm \
  --model qwen2.5 \
  --ollama-host http://localhost:11434/api \
  --prompt "Explain recursion"
```

### CLI flags

| Flag | Default | Description |
|---|---|---|
| `--model` | `llama3.1` | Ollama model name |
| `--prompt` | "" | Prompt text (or pipe to stdin) |
| `--max-iterations` | `30` | Maximum REPL iterations |
| `--max-depth` | `2` | Maximum recursion depth |
| `--ollama-host` | "" | Ollama base URL |

## HTTP Server

Run the RLM runtime as an HTTP service:

```bash
go run ./cmd/rlm-server
```

Configuration via environment variables:

| Variable | Default | Description |
|---|---|---|
| `PORT` | `8080` | Server port |
| `LOG_LEVEL` | `info` | Log level: debug, info, warn, error |
| `OLLAMA_HOST` | "" | Default Ollama base URL |

### Endpoints

`GET /health` — health check.

```bash
curl http://localhost:8080/health
```

`POST /api/v1/complete` — run a completion. Accepts `multipart/form-data` with the following fields:

| Field | Required | Default | Description |
|---|---|---|---|
| `prompt` | yes | — | The question or task |
| `context` | no | — | A `.txt` file to load as REPL context |
| `model` | no | `llama3.1` | Ollama model name |
| `max_iterations` | no | `30` | Maximum REPL iterations |
| `max_depth` | no | `2` | Maximum recursion depth |
| `ollama_host` | no | `OLLAMA_HOST` env | Ollama base URL |

Example request with a context file:

```bash
curl -X POST http://localhost:8080/api/v1/complete \
  --form 'prompt="Summarize the attached context"' \
  --form 'context=@context.txt' \
  --form 'model=llama3.1' \
  --form 'max_iterations=30' \
  --form 'max_depth=2'
```

Example response:

```json
{
  "response": "The final answer produced by the model",
  "root_model": "llama3.1",
  "execution_time_ms": 12345,
  "iterations": 2,
  "usage": {
    "model_usage_summaries": {
      "llama3.1": {
        "total_calls": 3,
        "total_input_tokens": 150,
        "total_output_tokens": 80
      }
    }
  }
}
```

The response includes the model's final answer, the model that produced it, the
execution time in milliseconds, the number of iterations, and per-model usage
metrics (total calls, input tokens, and output tokens).

## Testing

```bash
go test -race ./...
```

Docker-dependent integration tests are skipped unless `-short` is omitted and a
Docker daemon is available.

## Architecture

The runtime is split across these layers:

- `cmd/rlm` — command-line interface
- `cmd/rlm-server` — HTTP server built on Fiber
- `pkg/rlm` — public API wrapper
- `internal/rlm` — orchestrator and LM handler
- `internal/client` — LM backends (Ollama)
- `internal/environment` — Docker Python REPL
- `internal/prompt` — system/user prompt builders
- `internal/server` — gRPC services for container ↔ host communication

Container-side code calls the host over gRPC (`LMService`, `RLMService`), while
the host drives the container via `docker exec` and parses JSON from stdout.
