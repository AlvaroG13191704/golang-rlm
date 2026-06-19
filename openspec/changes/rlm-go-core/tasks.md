# Tasks: RLM Go Core

## Review Workload Forecast

| Field | Value |
|---|---|
| Estimated changed lines | 2500–3500 |
| 400-line budget risk | High |
| Chained PRs recommended | Yes |
| Suggested split | PR 1 proto/types → PR 2 client/prompt → PR 3 handler → PR 4 repl → PR 5 orchestrator → PR 6 api/cli |
| Delivery strategy | ask-on-risk |
| Chain strategy | stacked-to-main |

Decision needed before apply: No
Chained PRs recommended: Yes
Chain strategy: stacked-to-main
400-line budget risk: High

### Suggested Work Units

| Unit | Goal | Likely PR | Notes |
|---|---|---|---|
| 1 | gRPC proto + core types | PR 1 | Foundation; generated code committed |
| 2 | BaseLM + Ollama + prompt engine | PR 2 | Independent after PR 1 |
| 3 | LMHandler gRPC server | PR 3 | Depends on PR 2 |
| 4 | Docker REPL + exec script | PR 4 | Depends on PR 1 |
| 5 | RLM orchestrator + iteration | PR 5 | Depends on PR 3, PR 4 |
| 6 | Public API + CLI + integration | PR 6 | Depends on PR 5 |

## Phase 1: Foundation

- [x] 1.1 Update `go.mod` with gRPC, protobuf, and `x/sync` dependencies.
- [x] 1.2 Create `buf.yaml` and `buf.gen.yaml`.
- [x] 1.3 RED: Write table-driven tests for proto message construction.
- [x] 1.4 Create `proto/rlm/v1/types.proto`, `lm_service.proto`, and `rlm_service.proto`.
- [x] 1.5 GREEN: Generate `gen/rlm/v1/*.go` and make message tests pass.
- [x] 1.6 Create `internal/rlm/types.go` with `REPLResult`, `RLMChatCompletion`, and limit errors.

## Phase 2: LM Client & Prompt Engine

- [ ] 2.1 RED: Write `httptest` table for Ollama generate/chat/errors.
- [ ] 2.2 Create `internal/client/base.go` and `internal/client/ollama.go`.
- [ ] 2.3 GREEN: Make Ollama tests pass, including usage/duration/zero-division guards.
- [ ] 2.4 RED: Write `QueryMetadata` and prompt-builder tests.
- [ ] 2.5 Create `internal/prompt/prompts.go`.
- [ ] 2.6 GREEN: Make prompt tests pass.

## Phase 3: LM Handler gRPC Server

- [ ] 3.1 RED: Write `bufconn` tests for `LMService` routing and batched order.
- [ ] 3.2 Create `internal/rlm/handler.go` with client registry and depth routing.
- [ ] 3.3 GREEN: Make handler tests pass.
- [ ] 3.4 RED: Write `bufconn` tests for `RLMService` subcall.
- [ ] 3.5 Create `internal/server/server.go` wiring LM/RLM services.
- [ ] 3.6 GREEN: Make subcall tests pass.

## Phase 4: Docker REPL

- [ ] 4.1 RED: Write exec-script template golden tests.
- [ ] 4.2 Create `container/Dockerfile` and `internal/environment/execscript.go`.
- [ ] 4.3 GREEN: Make template tests pass.
- [ ] 4.4 Create `internal/environment/docker.go` with lifecycle and `docker exec` runner.
- [ ] 4.5 RED: Write `DockerREPL` mock command-runner tests.
- [ ] 4.6 GREEN: Make DockerREPL tests pass.

## Phase 5: RLM Orchestrator

- [ ] 5.1 RED: Write RLM limit-enforcement table tests with mock `Environment` and `BaseLM`.
- [ ] 5.2 Create `internal/rlm/orchestrator.go` and `internal/rlm/iteration.go`.
- [ ] 5.3 GREEN: Make orchestrator tests pass.
- [ ] 5.4 RED: Write sibling-container recursion tests.
- [ ] 5.5 GREEN: Implement `rlm_query` sibling-container spawn.

## Phase 6: Public API, CLI & Verification

- [ ] 6.1 Create `pkg/rlm/rlm.go` public wrapper.
- [ ] 6.2 Create `cmd/rlm/main.go` CLI.
- [ ] 6.3 RED: Write integration test for full `Completion` flow with mocks.
- [ ] 6.4 GREEN: Make integration test pass.
- [ ] 6.5 Run `go test -race ./...` and fix races.
- [ ] 6.6 Run Docker E2E tests, skipped via `testing.Short()`.
