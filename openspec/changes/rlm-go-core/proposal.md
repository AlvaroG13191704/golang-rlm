# Proposal: RLM Go Core

## Intent

Port the RLM core from Python to Go. The LLM generates code in a REPL, calling sub-LLMs (`llm_query`) and sub-RLMs (`rlm_query`). Go targets production concurrency, type safety, and a conference demo.

## Scope

### In Scope

- gRPC services (`LMService`, `RLMService`) with protobuf schemas
- RLM orchestrator (iteration, budget/timeout/error/token limits, recursion)
- LMHandler gRPC server with model/depth routing and batched fan-out
- Docker REPL: hybrid gRPC (container→host) + `docker exec` (host→container)
- BaseLM interface + Ollama client; prompt engine
- Python exec script with gRPC stubs; sibling-container recursion (depth-prefixed state)
- CLI (`cmd/rlm`); TDD for all components

### Out of Scope

- Additional LM clients, persistent REPL, compaction, custom tools, streaming

## Capabilities

### New Capabilities

- `grpc-services`: `LMService` + `RLMService` proto definitions (Complete, CompleteBatched, Subcall, SubcallBatched, GetUsage)
- `rlm-orchestrator`: Iteration loop, limit enforcement, child RLM spawning for `rlm_query`
- `lm-handler`: gRPC server, model/depth client routing, batched fan-out, usage tracking
- `docker-repl`: Container lifecycle, hybrid comms, dill state persistence, context loading
- `lm-client`: BaseLM interface + Ollama HTTP client
- `prompt-engine`: System prompt templates, query metadata, per-turn formatting

### Modified Capabilities

None — greenfield.

## Approach

**Hybrid gRPC**: container→host via gRPC for `llm_query`/`rlm_query`; host→container via `docker exec` with JSON stdout. Recursion spawns sibling containers sharing workspace volume; state files depth-prefixed (`state_d{depth}_{uuid}.dill`). Concurrency via `errgroup.WithContext` and `sync.RWMutex`.

## Affected Areas

All new: `proto/rlm/v1/` (gRPC defs), `internal/core/` (orchestrator, types), `internal/server/` (gRPC services), `internal/client/` (BaseLM + Ollama), `internal/environment/` (Docker REPL), `internal/prompt/` (templates), `cmd/rlm/` (CLI), `container/` (Dockerfile + exec script).

## Risks

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Container startup latency (1–3s) | Med | Acceptable for demo |
| Prompt type flexibility in protobuf | Med | JSON string + type discriminator |
| gRPC port discovery from container | Low | Exec script param + `host.docker.internal` |
| Sibling container resource pressure | Low | `errgroup.SetLimit` |

## Rollback Plan

Greenfield — delete artifacts and generated code. Python reference untouched.

## Dependencies

Go 1.26.4, `grpc`, `protobuf`, `buf`, Docker Engine, Python 3.11+ image (`dill` + `grpcio`).

## Success Criteria

- [ ] `go build ./...` and `go test -race ./...` pass
- [ ] RLM loop terminates on `answer["ready"]=True` or max iterations
- [ ] Budget/timeout/error/token limits enforced
- [ ] `llm_query` routes via gRPC; `rlm_query` spawns sibling container
- [ ] Docker lifecycle + dill state persistence work reliably
- [ ] CLI runs a completion against a real LM provider

## Proposal Question Round

1. **Demo scope**: Single `Completion()` call or multi-turn conversation?
2. **Recursion depth**: `max_depth=2` sufficient for demo?
3. **Error surfacing**: Auto-retry LM errors or surface as error strings?
4. **Priority**: Library API (`pkg/rlm/`) or CLI (`cmd/rlm/`) first?
