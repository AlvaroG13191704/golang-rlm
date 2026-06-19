# RLM Go Core — Specification

## Overview

Port the Python RLM inference runtime to Go. A single `Completion()` call spawns a Docker-based Python REPL, runs an iterative LLM loop, and returns the final answer produced by the model inside the REPL. The architecture is **hybrid gRPC**: container → host uses gRPC (`LMService`, `RLMService`); host → container uses `docker exec` with JSON stdout.

This change is greenfield. All requirements below are **ADDED**.

---

## Capability 1: gRPC Services

### Purpose
Define the protobuf contracts between the Go host and the Python REPL container.

### Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| GS-1 | `LMService` MUST expose `Complete`, `CompleteBatched`, and `GetUsage` RPCs. | Happy path: container calls `Complete` and receives content + usage. |
| GS-2 | `RLMService` MUST expose `Subcall` and `SubcallBatched` RPCs for recursive child RLM calls. | Happy path: container calls `Subcall` and host returns child RLM result. |
| GS-3 | Request messages MUST carry `prompt` (string or JSON-encoded Struct), optional `model`, and `depth`. | Edge case: empty model string routes by depth. |
| GS-4 | Response messages MUST carry `content`, `root_model`, `usage`, `execution_time`, and optional `error`. | Error state: LM failure returns `error` string and empty content. |
| GS-5 | `CompleteBatched` / `SubcallBatched` MUST preserve prompt order in responses. | Happy path: 4 prompts in → 4 responses in same order. |
| GS-6 | Shared types (`UsageSummary`, `ModelUsageSummary`) MUST be defined in a common `types.proto`. | — |

### Scenarios

#### Scenario: Single LM completion over gRPC
- GIVEN the host is serving `LMService`
- WHEN the container calls `Complete` with prompt "Summarize" and depth=1
- THEN the response contains the LM output, model name, and usage counters

#### Scenario: Batched completion preserves order
- GIVEN the host is serving `LMService`
- WHEN the container calls `CompleteBatched` with prompts ["A", "B", "C"]
- THEN the response contains responses ["Answer A", "Answer B", "Answer C"] in the same order

### Interfaces / Contracts

```protobuf
service LMService {
  rpc Complete(CompleteRequest) returns (CompleteResponse);
  rpc CompleteBatched(BatchedRequest) returns (BatchedResponse);
  rpc GetUsage(GetUsageRequest) returns (GetUsageResponse);
}

service RLMService {
  rpc Subcall(SubcallRequest) returns (SubcallResponse);
  rpc SubcallBatched(BatchedSubcallRequest) returns (BatchedSubcallResponse);
}
```

### Error Handling
- gRPC errors are surfaced as `error` strings in responses; the container exec script returns them to the model as `"Error: {message}"`.
- Serialization failures MUST return gRPC `INTERNAL` status with a lowercase message.

### Test Requirements
- Unit tests for generated message construction (table-driven).
- `bufconn` integration tests for `LMService` and `RLMService` with mock backends.
- Verify batched responses preserve order when one prompt fails.

---

## Capability 2: RLM Orchestrator

### Purpose
Run the iterative prompt → code → execute → check loop, enforce limits, and spawn child RLMs for recursive calls.

### Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| RO-1 | `RLM.Completion()` MUST spawn a fresh LMHandler and Docker REPL per call. | Happy path: single call returns final answer. |
| RO-2 | The loop MUST terminate when `REPLResult.final_answer` is non-empty. | Happy path: model sets `answer["ready"]=True`. |
| RO-3 | The loop MUST terminate after `max_iterations` and return a default answer. | Edge case: model never submits final answer. |
| RO-4 | The loop MUST enforce `max_timeout`, `max_budget`, `max_tokens`, and `max_errors`. | Error state: limit exceeded raises typed error with best partial answer. |
| RO-5 | Consecutive errors MUST reset to zero after one successful iteration. | Edge case: error-success-error pattern does not prematurely stop. |
| RO-6 | At `depth >= max_depth` the orchestrator MUST fall back to a plain LM completion. | Edge case: depth=1 with max_depth=1 skips REPL entirely. |
| RO-7 | `rlm_query` MUST spawn a child RLM in a sibling Docker container sharing the workspace. | Happy path: parent receives child final answer. |
| RO-8 | Child RLMs MUST inherit remaining budget, timeout, and token limits from the parent. | Edge case: exhausted budget returns error string immediately. |

### Scenarios

#### Scenario: Successful single-turn completion
- GIVEN a configured RLM with max_iterations=5
- WHEN the model emits a `repl` block that sets `answer["ready"]=True`
- THEN `Completion()` returns the final answer with one iteration recorded

#### Scenario: Iteration budget exhausted
- GIVEN a configured RLM with max_iterations=2
- WHEN the model emits code that does not set `answer["ready"]=True`
- THEN `Completion()` calls the default-answer prompt and returns after 2 iterations

#### Scenario: Timeout mid-loop
- GIVEN a configured RLM with max_timeout=0.1s
- WHEN the second iteration exceeds the timeout
- THEN `Completion()` returns a timeout error containing the best partial answer

### Interfaces / Contracts

```go
type RLM struct {
    Backend              string
    BackendKwargs        map[string]any
    MaxDepth             int
    MaxIterations        int
    MaxBudget            *float64
    MaxTimeout           *float64
    MaxTokens            *int
    MaxErrors            *int
    MaxConcurrentSubcalls int
}

func (r *RLM) Completion(ctx context.Context, prompt any, rootPrompt string) (*RLMChatCompletion, error)
```

### Error Handling
- Limit errors are typed (`TimeoutExceeded`, `BudgetExceeded`, `TokenLimitExceeded`, `ErrorThresholdExceeded`) and carry the best partial answer.
- Errors from the LM or REPL are surfaced as strings inside `REPLResult.stderr`; no auto-retry.

### Test Requirements
- Mock `Environment` and `BaseLM` interfaces for unit tests.
- Table-driven tests for limit enforcement.
- Race-free test with `go test -race`.

---

## Capability 3: LM Handler

### Purpose
Serve LM clients over gRPC, route requests by model name and depth, run batched fan-out, and aggregate usage.

### Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| LH-1 | The handler MUST register a default client and an optional depth-1 client. | — |
| LH-2 | `GetClient(model, depth)` MUST return the model-named client if provided, else depth=0 → default, depth=1 → other backend, otherwise default. | Happy path: depth=1 routes to other backend. |
| LH-3 | `CompleteBatched` MUST fan out prompts concurrently with a bounded worker pool. | Happy path: 8 prompts with concurrency 4 completes in 2 waves. |
| LH-4 | The handler MUST aggregate per-model usage across all registered clients. | — |
| LH-5 | The gRPC server lifecycle MUST be start/stoppable and bind to an ephemeral port. | — |

### Scenarios

#### Scenario: Depth-based routing
- GIVEN a handler with default model "llama3.1" and other backend "qwen2.5"
- WHEN `Complete` is called with depth=1 and no model override
- THEN the request routes to the "qwen2.5" client

#### Scenario: Batched partial failure
- GIVEN a batched request with 3 prompts where the second prompt fails
- WHEN `CompleteBatched` finishes
- THEN the first and third responses succeed, and the second response carries an `error` field

### Interfaces / Contracts

```go
type LMHandler struct { /* unexported fields */ }

func NewLMHandler(defaultClient BaseLM, opts ...LMHandlerOption) *LMHandler
func (h *LMHandler) Start() (host string, port int, err error)
func (h *LMHandler) Stop() error
func (h *LMHandler) GetUsageSummary() UsageSummary
```

### Error Handling
- LM errors are captured per prompt in batched mode; the overall RPC succeeds.
- Handler startup failures (e.g., port bind) return wrapped errors.

### Test Requirements
- `bufconn` gRPC tests with fake `BaseLM`.
- Verify routing matrix for model + depth combinations.
- Verify usage aggregation after mixed success/failure batches.

---

## Capability 4: Docker REPL

### Purpose
Manage a Docker container running Python, execute code via `docker exec`, persist state via dill, and load context from the shared workspace.

### Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| DR-1 | `DockerREPL` MUST create a container from a configurable image, mount a shared workspace volume, and install `dill` and `grpcio`. | — |
| DR-2 | `ExecuteCode(code)` MUST run a Python exec script via `docker exec` and parse JSON from stdout. | Happy path: code prints and sets `answer["ready"]=True`. |
| DR-3 | The exec script MUST load prior state from `/workspace/state.dill`, execute code, and save state back. | Happy path: variable set in turn 1 is visible in turn 2. |
| DR-4 | `LoadContext(payload)` MUST write `context.txt` (string) or `context.json` (object/list) to the workspace and expose it as `context` inside the REPL. | Happy path: string context accessible as `context`. |
| DR-5 | The REPL namespace MUST expose `llm_query`, `llm_query_batched`, `rlm_query`, `rlm_query_batched`, `SHOW_VARS`, and `answer`. | — |
| DR-6 | Recursive calls MUST use depth-prefixed state files: `state_d{depth}_{uuid}.dill`. | Happy path: sibling container does not clobber parent state. |
| DR-7 | `Cleanup()` MUST stop and remove the container and delete the temp workspace. | — |

### Scenarios

#### Scenario: State persistence across turns
- GIVEN a fresh Docker REPL
- WHEN turn 1 sets `x = 42` and turn 2 prints `x`
- THEN turn 2 stdout contains `42`

#### Scenario: Context string loaded
- GIVEN a context payload "The magic number is 7"
- WHEN `LoadContext` runs and the model prints `context[:20]`
- THEN stdout contains "The magic number is"

#### Scenario: Final answer signaling
- GIVEN code that sets `answer["content"]="done"` and `answer["ready"]=True`
- WHEN `ExecuteCode` returns
- THEN `REPLResult.final_answer` equals "done"

### Interfaces / Contracts

```go
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
```

### Error Handling
- Non-zero `docker exec` exit code or unparseable JSON returns a wrapped error and raw stdout/stderr for debugging.
- Missing gRPC handler address returns `"Error: no LM handler configured"` to the model.

### Test Requirements
- Unit tests for exec script template generation.
- Integration tests requiring Docker are skipped with `testing.Short()`.
- Mock tests for `ExecuteCode` using a fake command runner.

---

## Capability 5: LM Client

### Purpose
Abstract LM providers behind a `BaseLM` interface; ship an Ollama HTTP client with usage tracking.

### Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| LC-1 | `BaseLM` MUST support `Completion(ctx, prompt) (string, error)` and `GetUsageSummary() ModelUsageSummary`. | — |
| LC-2 | `OllamaClient` MUST implement `BaseLM` using the Ollama HTTP API (`/api/generate` for string prompts, `/api/chat` for message lists). | — |
| LC-3 | `OllamaClient` MUST read the base URL from `OLLAMA_HOST` or default to `http://localhost:11434/api`. | Error state: malformed base URL returns error. |
| LC-4 | `OllamaClient` MUST call `/api/generate` with `stream: false` for deterministic usage capture. | Happy path: single string prompt returns response text. |
| LC-5 | `OllamaClient` MUST call `/api/chat` with `stream: false` when the prompt is a message list. | Happy path: message list returns assistant content. |
| LC-6 | `OllamaClient` MUST convert nanosecond duration fields (`total_duration`, `prompt_eval_duration`, `eval_duration`) to `time.Duration` and compute derived metrics (toks/sec, TTFT) guarding zero division. | — |
| LC-7 | `OllamaClient` MUST map Ollama usage fields (`prompt_eval_count`, `eval_count`) to `ModelUsageSummary` input/output tokens; cost MAY be `nil` since Ollama does not report cost. | — |

### Scenarios

#### Scenario: Ollama string prompt completion
- GIVEN an Ollama client pointing at a mocked `/api/generate` endpoint
- WHEN `Completion` is called with string prompt "Hello"
- THEN it returns the `response` field, records 1 call, and maps `prompt_eval_count`/`eval_count` to input/output tokens

#### Scenario: Ollama chat with message list
- GIVEN an Ollama client pointing at a mocked `/api/chat` endpoint
- WHEN `Completion` is called with a message list `[{role: "user", content: "Hi"}]`
- THEN it returns `response.message.content` and records usage

#### Scenario: Ollama non-2xx response
- GIVEN an Ollama client pointing at an endpoint that returns HTTP 404 with plain text body
- WHEN `Completion` is called
- THEN it returns a wrapped error containing the status code and body preview

### Interfaces / Contracts

```go
type BaseLM interface {
    Completion(ctx context.Context, prompt any) (string, error)
    GetUsageSummary() ModelUsageSummary
}

type OllamaClient struct { /* unexported fields */ }

func NewOllamaClient(model string, baseURL string, httpClient *http.Client) (*OllamaClient, error)
```

### Error Handling
- Ollama non-2xx responses wrap the plain text body with the status code.
- Invalid response JSON returns wrapped errors.
- Missing or empty `model` returns an error immediately.

### Test Requirements
- `httptest.Server` mocks for Ollama `/api/generate` and `/api/chat`.
- Table-driven tests for usage tracking.
- Tests for nanosecond duration conversion and zero-division guards in Ollama metrics.

---

## Capability 6: Prompt Engine

### Purpose
Build system prompts and per-turn user prompts, including query metadata and optional custom tools.

### Requirements

| ID | Requirement | Scenarios |
|---|---|---|
| PE-1 | `BuildSystemPrompt` MUST combine `RLM_SYSTEM_PROMPT`, optional `ORCHESTRATOR_ADDENDUM`, custom tools section, and query metadata. | — |
| PE-2 | `QueryMetadata` MUST compute `context_type` (str/dict/list) and total character length. | Edge case: empty list → length 0, type list. |
| PE-3 | `BuildUserPrompt` MUST emit "Turn {iter}/{max_iter}" and a first-turn safeguard. | Happy path: iteration 0 includes safeguard text. |
| PE-4 | `BuildUserPrompt` MUST append notes for multiple contexts or histories when present. | Edge case: 2 contexts → note mentions context_0 and context_1. |

### Scenarios

#### Scenario: System prompt with metadata
- GIVEN a string prompt of 100 characters
- WHEN `BuildSystemPrompt` is called with orchestrator=true
- THEN the result contains the system role, metadata user message, and orchestrator addendum

#### Scenario: First-turn safeguard
- GIVEN max_iterations=5
- WHEN `BuildUserPrompt(rootPrompt, 0, 1, 0, 5)` is called
- THEN the content includes the safeguard and "Turn 1/5"

### Interfaces / Contracts

```go
type QueryMetadata struct {
    ContextLengths     []int
    ContextTotalLength int
    ContextType        string
}

func NewQueryMetadata(prompt any) (QueryMetadata, error)
func BuildSystemPrompt(systemPrompt string, meta QueryMetadata, customTools map[string]any, rootPrompt string, orchestrator bool) ([]Message, error)
func BuildUserPrompt(rootPrompt string, iteration, contextCount, historyCount, maxIterations int) Message
```

### Error Handling
- Unsupported prompt types return a clear error.
- Missing custom tools section renders as an empty string.

### Test Requirements
- Golden-file or string-contains tests for prompt output.
- Table-driven tests for `QueryMetadata` edge cases.

---

## Traceability Matrix

| Requirement | Capability | Test Layer | Priority |
|---|---|---|---|
| GS-1..6 | gRPC Services | Unit + bufconn integration | Must |
| RO-1..8 | RLM Orchestrator | Unit with mocks | Must |
| LH-1..5 | LM Handler | bufconn integration | Must |
| DR-1..7 | Docker REPL | Integration (Docker) + unit mocks | Must |
| LC-1..7 | LM Client | `httptest` unit | Must |
| PE-1..4 | Prompt Engine | Table-driven unit | Must |

## Assumptions and Dependencies

- Go 1.26.4+, `google.golang.org/grpc`, `google.golang.org/protobuf`, Docker Engine.
- Container image: `python:3.11-slim` with `dill` and `grpcio` installed at runtime or baked into the Dockerfile.
- Ollama client requires a running Ollama server at `http://localhost:11434` (or `OLLAMA_HOST`) for live tests.
- The host can reach `host.docker.internal` from containers (Docker Desktop / Linux with `--add-host`).
- `max_depth` defaults to 2 for v1 (root + one recursive level).
- Demo scope is a single `Completion()` call; persistent multi-turn REPLs are out of scope.
- Errors are surfaced as strings in REPL output; no automatic retry is implemented.

## Non-Functional Requirements

- `go build ./...` and `go test -race ./...` MUST pass.
- Every exported function and gRPC method MUST have at least one test.
- Docker-dependent tests MUST respect `testing.Short()`.
- Public library API lives in `pkg/rlm/`; CLI in `cmd/rlm/` is secondary.