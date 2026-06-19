# Design: RLM Go Core

## Technical Approach

Port the Python RLM inference loop to Go using a **hybrid gRPC** architecture: the container calls the host over gRPC for `llm_query`/`rlm_query`, while the host drives the container via `docker exec` and parses JSON from stdout. A single `RLM.Completion()` call creates a fresh `LMHandler` gRPC server, spawns a Docker Python REPL, runs the prompt → code → execute → check loop, and returns the final answer. The design mirrors the Python reference (`rlm/rlm/core/rlm.py`, `lm_handler.py`, `environments/docker_repl.py`) but replaces sockets and HTTP with gRPC and Go concurrency primitives.

## Architecture Decisions

| Decision | Choice | Alternatives Rejected | Rationale |
|----------|--------|----------------------|-----------|
| Prompt payload in gRPC | `string prompt` + `PromptType` enum (`RAW`, `JSON`) | `google.protobuf.Struct` / `Value` | Keeps the Python exec-script client simple; host unmarshals JSON into `any` so str/dict/list all work; discriminator removes ambiguity. |
| Container → host | gRPC `LMService` / `RLMService` | HTTP broker / raw sockets | Type-safe, multiplexed, matches locked hybrid decision. |
| Host → container | `docker exec` with JSON stdout | Container-side gRPC server | No server inside container; smaller image; same pattern as Python reference. |
| Recursion | Sibling containers sharing workspace; depth-prefixed state files | Nested exec in same container | Matches reference behavior; clean isolation; avoids state collision. |
| Batched/subcall concurrency | `errgroup.WithContext` + `SetLimit` | Unbounded goroutines / `sync.WaitGroup` | Structured concurrency with cancellation and backpressure. |
| LM client (v1) | Ollama HTTP client only | Multi-provider registry | Locked decision; keeps scope demo-sized. |
| State serialization | `dill` (pickle fallback) in container | Go parsing dill | Go never touches dill bytes; contract stays identical to Python. |
| Public API | `pkg/rlm/rlm.go` wraps internal orchestrator | Expose all internals | Library-first priority; minimal surface. |

## Data Flow

### Single completion

```
User → pkg/rlm.Completion()
        │
        ▼
  internal/rlm.RLM
        │
        ├──► internal/rlm.LMHandler (gRPC server, ephemeral port)
        │
        └──► internal/environment.DockerREPL
                 │
                 ├─ docker run ─► Container (python:3.11-slim)
                 │
                 ├─ docker exec ─► exec script loads state, runs code
                 │         │
                 │         └─ llm_query ──gRPC──► LMService.Complete
                 │         │
                 │         └─ rlm_query ──gRPC──► RLMService.Subcall
                 │                  │
                 │                  └─ host spawns child RLM + sibling container
                 │
                 └─ stdout JSON ◄── REPLResult
```

### Batched completion

```
LMService.CompleteBatched
   │
   ├─ errgroup.SetLimit(batchConcurrency)
   ├─ goroutine per prompt → BaseLM.Completion
   └─ results ordered by index
```

## File Changes

| File | Action | Description |
|------|--------|-------------|
| `proto/rlm/v1/types.proto` | Create | Shared messages: `ModelUsageSummary`, `UsageSummary`, `RLMChatCompletion`. |
| `proto/rlm/v1/lm_service.proto` | Create | `LMService`: `Complete`, `CompleteBatched`, `GetUsage`. |
| `proto/rlm/v1/rlm_service.proto` | Create | `RLMService`: `Subcall`, `SubcallBatched`. |
| `gen/rlm/v1/*.go` | Create | Generated protobuf/gRPC code (committed). |
| `pkg/rlm/rlm.go` | Create | Public API: `New`, `Completion`, config builder. |
| `internal/rlm/types.go` | Create | Go domain types: `REPLResult`, `RLMChatCompletion`, `UsageSummary`, limit errors. |
| `internal/rlm/orchestrator.go` | Create | `RLM` loop, limit enforcement, child RLM spawning. |
| `internal/rlm/handler.go` | Create | `LMHandler`: client registry, depth routing, batched fan-out. |
| `internal/rlm/iteration.go` | Create | Code-block parsing and `RLMIteration` formatting. |
| `internal/client/base.go` | Create | `BaseLM` interface. |
| `internal/client/ollama.go` | Create | Ollama HTTP client with usage tracking. |
| `internal/environment/environment.go` | Create | `Environment` interface. |
| `internal/environment/docker.go` | Create | `DockerREPL` lifecycle and `docker exec` runner. |
| `internal/environment/execscript.go` | Create | Python exec-script template builder. |
| `internal/prompt/prompts.go` | Create | `QueryMetadata`, system/user prompt builders. |
| `internal/server/server.go` | Create | gRPC server registration and lifecycle. |
| `cmd/rlm/main.go` | Create | CLI entry point (secondary). |
| `container/Dockerfile` | Create | `python:3.11-slim` with `dill` and `grpcio` baked in. |
| `go.mod` | Modify | Add `google.golang.org/grpc`, `google.golang.org/protobuf`, `golang.org/x/sync`. |
| `buf.gen.yaml`, `buf.yaml` | Create | Buf generation config. |

## Interfaces / Contracts

```go
type BaseLM interface {
    Completion(ctx context.Context, prompt any) (string, error)
    GetUsageSummary() ModelUsageSummary
    GetLastUsage() ModelUsageSummary
}

type Environment interface {
    ExecuteCode(ctx context.Context, code string) (REPLResult, error)
    LoadContext(ctx context.Context, payload any) error
    Cleanup(ctx context.Context) error
}

type REPLResult struct {
    Stdout        string
    Stderr        string
    Locals        map[string]string
    FinalAnswer   string
    ExecutionTime time.Duration
    LLMCalls      []RLMChatCompletion
}

type RLM struct {
    Backend               string
    BackendKwargs         map[string]any
    MaxDepth              int
    MaxIterations         int
    MaxBudget             *float64
    MaxTimeout            *float64
    MaxTokens             *int
    MaxErrors             *int
    MaxConcurrentSubcalls int
}

func (r *RLM) Completion(ctx context.Context, prompt any, rootPrompt string) (*RLMChatCompletion, error)

type LMHandler struct{ /* unexported */ }

func NewLMHandler(defaultClient BaseLM, opts ...LMHandlerOption) *LMHandler
func (h *LMHandler) Start() (host string, port int, err error)
func (h *LMHandler) Stop() error
func (h *LMHandler) GetUsageSummary() UsageSummary

type OllamaClient struct{ /* unexported */ }

func NewOllamaClient(model string, baseURL string, httpClient *http.Client) (*OllamaClient, error)
```

### Key protobuf snippets

```protobuf
enum PromptType { PROMPT_RAW = 0; PROMPT_JSON = 1; }

message CompleteRequest {
  string    prompt      = 1; // raw text or JSON-encoded value
  PromptType prompt_type = 2;
  string    model       = 3;
  int32     depth       = 4;
}

message CompleteResponse {
  string            content        = 1;
  string            root_model     = 2;
  ModelUsageSummary usage          = 3;
  google.protobuf.Duration execution_time = 4;
  string            error          = 5;
}
```

`BatchedRequest` uses `repeated CompleteRequest items` so each slot keeps its own model/depth and ordering is trivially preserved by index.

## Concurrency Model

- `RLM.Completion` runs the iteration loop synchronously; only subcalls are concurrent.
- `LMHandler.CompleteBatched` uses `errgroup.WithContext(ctx)` and `SetLimit(batchConcurrency)`; results are written into a pre-allocated slice by index.
- `RLMService.SubcallBatched` uses `errgroup.SetLimit(r.MaxConcurrentSubcalls)`; each slot spawns a child `RLM` with its own sibling container.
- `LMHandler` client registry is protected by `sync.RWMutex`.
- `OllamaClient` usage counters are protected by `sync.Mutex`.
- `DockerREPL` pending-call accumulation is protected by `sync.Mutex`.
- All long-running operations accept `context.Context`; `docker exec` uses `exec.CommandContext` so cancellation/timeout propagates.

## Error Propagation

- Internal errors are wrapped with `fmt.Errorf("...: %w", err)`.
- Limit errors are typed (`ErrTimeoutExceeded`, `ErrBudgetExceeded`, `ErrTokenLimitExceeded`, `ErrErrorThresholdExceeded`) and expose `PartialAnswer()`.
- Batched LM failures are captured per-item in `CompleteResponse.error`; the RPC itself succeeds.
- Container-side errors return `"Error: ..."` strings to the model instead of aborting the loop.
- gRPC serialization failures return `status.Error(codes.Internal, "...")`.

## Testing Strategy

| Layer | What to Test | Approach |
|-------|-------------|----------|
| Unit | `QueryMetadata`, prompt builders | Table-driven pure-function tests. |
| Unit | Code-block parsing / iteration formatting | Table-driven with raw LLM outputs. |
| Unit | `OllamaClient` | `httptest.Server` mocks for `/api/generate` and `/api/chat`; table-driven usage/duration cases. |
| Unit | `DockerREPL` exec-script template | Golden/string assertions. |
| Integration | `LMService` / `RLMService` | `bufconn` gRPC with fake `BaseLM` and fake subcall handler. |
| Integration | `RLM` orchestrator | Mock `Environment` and `BaseLM`; table-driven limit tests. |
| E2E | Docker container lifecycle, state/context | Real Docker; skipped via `testing.Short()` and `docker version` probe. |
| Race | All above | `go test -race ./...`. |

## Migration / Rollout

No migration required. This is a greenfield change. Rollback is `git rm` of the new packages and generated code.

## Work-Unit Commit Plan

1. `feat(proto): add rlm/v1 gRPC services and generated code`
2. `feat(types): add core RLM types and limit-error definitions`
3. `feat(client): add BaseLM interface and Ollama client with tests`
4. `feat(prompt): add QueryMetadata and system/user prompt builders`
5. `feat(handler): add LMHandler gRPC server with depth routing and batches`
6. `feat(repl): add Docker REPL lifecycle, exec script, and state contract`
7. `feat(rlm): add RLM orchestrator with limits and sibling-container recursion`
8. `feat(cli): add cmd/rlm entry point`

Each commit includes tests for the behavior it introduces.

## Open Questions

- None — all locked decisions are captured above.
