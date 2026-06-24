// Package environment implements the Docker-based Python REPL used by the RLM
// orchestrator.
package environment

import (
	"bytes"
	"context"
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"rlm-golang/internal/rlm"
	"rlm-golang/internal/types"
)

// commandRunner abstracts os/exec so DockerREPL can be unit-tested without a
// real Docker daemon.
type commandRunner interface {
	Run(ctx context.Context, name string, arg ...string) (stdout []byte, stderr []byte, err error)
}

type realCommandRunner struct{}

func (r *realCommandRunner) Run(ctx context.Context, name string, arg ...string) ([]byte, []byte, error) {
	cmd := exec.CommandContext(ctx, name, arg...)
	var out, err bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &err
	runErr := cmd.Run()
	return out.Bytes(), err.Bytes(), runErr
}

// DockerREPLConfig configures a Docker-based Python REPL.
type DockerREPLConfig struct {
	// Image is the Docker image used to run the container. Defaults to
	// "rlm-sandbox".
	Image string
	// LMHandlerHost is the host address the container uses to reach the gRPC
	// LMService.  It is passed through to the Python exec script.
	LMHandlerHost string
	// LMHandlerPort is the host port the container uses to reach the gRPC
	// LMService.
	LMHandlerPort int
	// Depth is the recursion depth forwarded to LMService requests.
	Depth int
	// MaxDepth is the maximum recursion depth. At or past this cap, rlm_query
	// inside the container falls back to llm_query.
	MaxDepth int
	// Workspace is an optional host directory mounted at /workspace.  When
	// empty a temporary directory is created. Sibling containers created by
	// recursive Subcall invocations share the same workspace directory but use
	// distinct depth-prefixed state files so they do not clobber each other.
	Workspace string
	// WorkspaceID is an optional identifier embedded in the dill state file
	// name. When empty, a random identifier is generated.
	WorkspaceID string
}

// DockerREPL manages a Docker container that executes Python code on behalf of
// the RLM orchestrator.
type DockerREPL struct {
	image       string
	workspace   string
	containerID string
	lmHost      string
	lmPort      int
	depth       int
	maxDepth    int
	workspaceID string
	runner      commandRunner
}

// NewDockerREPL creates a workspace, starts the Docker container, and returns a
// ready REPL.  On startup failure the workspace is cleaned up.
func NewDockerREPL(cfg DockerREPLConfig) (*DockerREPL, error) {
	repl := NewDockerREPLWithRunner(cfg, &realCommandRunner{})
	if err := repl.Setup(context.Background()); err != nil {
		_ = repl.Cleanup(context.Background())
		return nil, err
	}
	return repl, nil
}

// NewDockerREPLWithRunner returns an unconfigured REPL using the supplied
// command runner.  Callers must invoke Setup before use; this constructor is
// intended for tests.
func NewDockerREPLWithRunner(cfg DockerREPLConfig, runner commandRunner) *DockerREPL {
	image := cfg.Image
	if image == "" {
		image = "rlm-sandbox"
	}
	workspaceID := cfg.WorkspaceID
	if workspaceID == "" {
		workspaceID = generateWorkspaceID()
	}
	return &DockerREPL{
		image:       image,
		workspace:   cfg.Workspace,
		lmHost:      cfg.LMHandlerHost,
		lmPort:      cfg.LMHandlerPort,
		depth:       cfg.Depth,
		maxDepth:    cfg.MaxDepth,
		workspaceID: workspaceID,
		runner:      runner,
	}
}

func generateWorkspaceID() string {
	b := make([]byte, 4)
	if _, err := rand.Read(b); err != nil {
		return fmt.Sprintf("%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(b)
}

// Workspace returns the host directory mounted into the container.
func (r *DockerREPL) Workspace() string {
	return r.workspace
}

// ContainerID returns the running container identifier.  It is empty before
// Setup succeeds or after Cleanup.
func (r *DockerREPL) ContainerID() string {
	return r.containerID
}

// Setup creates the workspace (if necessary) and starts the Docker container.
func (r *DockerREPL) Setup(ctx context.Context) error {
	if r.workspace == "" {
		ws, err := os.MkdirTemp("", "rlm-docker-repl-")
		if err != nil {
			slog.Error("DockerREPL failed to create workspace", "error", err)
			return fmt.Errorf("create workspace: %w", err)
		}
		r.workspace = ws
	}

	slog.Debug("DockerREPL starting container", "image", r.image, "workspace", r.workspace, "depth", r.depth)
	stdout, stderr, err := r.runner.Run(
		ctx,
		"docker",
		"run",
		"-d",
		"--rm",
		"-v", r.workspace+":/workspace",
		"--add-host", "host.docker.internal:host-gateway",
		r.image,
		"tail", "-f", "/dev/null",
	)
	if err != nil {
		slog.Error("DockerREPL container start failed", "image", r.image, "error", err, "stderr", strings.TrimSpace(string(stderr)))
		return fmt.Errorf("docker run: %w: %s", err, strings.TrimSpace(string(stderr)))
	}

	r.containerID = strings.TrimSpace(string(stdout))
	slog.Debug("DockerREPL container started", "container_id", r.containerID, "workspace", r.workspace, "depth", r.depth)
	return nil
}

// LoadContext writes the context payload to the workspace and exposes it as
// `context` inside the REPL.  String payloads are written as context.txt;
// objects and lists are written as context.json.
func (r *DockerREPL) LoadContext(ctx context.Context, payload any) error {
	if r.workspace == "" {
		slog.Error("DockerREPL LoadContext called without workspace")
		return errors.New("workspace not initialized")
	}

	switch v := payload.(type) {
	case string:
		slog.Debug("DockerREPL loading string context", "container_id", r.containerID, "path", filepath.Join(r.workspace, "context.txt"), "size", len(v))
		if err := os.WriteFile(filepath.Join(r.workspace, "context.txt"), []byte(v), 0o644); err != nil {
			slog.Error("DockerREPL failed to write context.txt", "container_id", r.containerID, "error", err)
			return fmt.Errorf("write context.txt: %w", err)
		}
		
		// If payload looks like a CSV, also write context.csv
		if strings.Contains(v, "\n") {
			firstLine := strings.SplitN(v, "\n", 2)[0]
			if strings.Contains(firstLine, ",") || strings.Contains(firstLine, ";") || strings.Contains(firstLine, "\t") {
				slog.Debug("DockerREPL loading CSV context", "container_id", r.containerID, "path", filepath.Join(r.workspace, "context.csv"), "size", len(v))
				if err := os.WriteFile(filepath.Join(r.workspace, "context.csv"), []byte(v), 0o644); err != nil {
					slog.Error("DockerREPL failed to write context.csv", "container_id", r.containerID, "error", err)
					return fmt.Errorf("write context.csv: %w", err)
				}
			}
		}
	default:
		data, err := json.Marshal(v)
		if err != nil {
			slog.Error("DockerREPL failed to marshal context", "container_id", r.containerID, "error", err)
			return fmt.Errorf("marshal context: %w", err)
		}
		slog.Debug("DockerREPL loading JSON context", "container_id", r.containerID, "path", filepath.Join(r.workspace, "context.json"), "size", len(data))
		if err := os.WriteFile(filepath.Join(r.workspace, "context.json"), data, 0o644); err != nil {
			slog.Error("DockerREPL failed to write context.json", "container_id", r.containerID, "error", err)
			return fmt.Errorf("write context.json: %w", err)
		}
	}
	return nil
}

// ExecuteCode runs the supplied Python code inside the container and parses
// the JSON result printed by the exec script.
func (r *DockerREPL) ExecuteCode(ctx context.Context, code string) (rlm.REPLResult, error) {
	start := time.Now()
	result := rlm.REPLResult{}

	if r.containerID == "" {
		slog.Error("DockerREPL ExecuteCode called with no running container")
		return result, errors.New("container not running")
	}

	slog.Debug("DockerREPL executing code", "container_id", r.containerID, "code_len", len(code), "depth", r.depth)
	slog.Debug("DockerREPL executing code", "container_id", r.containerID, "code", truncateString(code, 200))

	script := BuildExecScript(ExecConfig{
		Code:          code,
		LMHandlerPort: r.lmPort,
		Depth:         r.depth,
		MaxDepth:      r.maxDepth,
		WorkspaceID:   r.workspaceID,
	})

	stdout, stderr, err := r.runner.Run(
		ctx,
		"docker",
		"exec",
		r.containerID,
		"python", "-c", script,
	)
	result.ExecutionTime = time.Since(start)

	if err != nil {
		slog.Error("DockerREPL docker exec failed", "container_id", r.containerID, "error", err, "stderr", strings.TrimSpace(string(stderr)), "execution_time", result.ExecutionTime)
		result.Stdout = string(stdout)
		result.Stderr = string(stderr)
		return result, fmt.Errorf("docker exec: %w", err)
	}

	slog.Debug("DockerREPL docker exec completed", "container_id", r.containerID, "stdout_len", len(stdout), "stderr_len", len(stderr), "execution_time", result.ExecutionTime)
	return r.parseExecOutput(stdout, stderr, result.ExecutionTime)
}

func (r *DockerREPL) parseExecOutput(stdout, stderr []byte, elapsed time.Duration) (rlm.REPLResult, error) {
	lines := strings.Split(strings.ReplaceAll(string(stdout), "\r\n", "\n"), "\n")
	var last string
	for i := len(lines) - 1; i >= 0; i-- {
		if strings.TrimSpace(lines[i]) != "" {
			last = lines[i]
			break
		}
	}

	var raw struct {
		Stdout      string            `json:"stdout"`
		Stderr      string            `json:"stderr"`
		Locals      map[string]string `json:"locals"`
		FinalAnswer string            `json:"final_answer"`
		LLMCalls    []rawLLMCall      `json:"llm_calls"`
	}

	if err := json.Unmarshal([]byte(last), &raw); err != nil {
		return rlm.REPLResult{
			Stdout:        string(stdout),
			Stderr:        string(stderr),
			ExecutionTime: elapsed,
		}, fmt.Errorf("parse exec output: %w", err)
	}

	calls := make([]types.RLMChatCompletion, len(raw.LLMCalls))
	for i, c := range raw.LLMCalls {
		calls[i] = types.RLMChatCompletion{
			Prompt:        c.Prompt,
			Response:      c.Response,
			RootModel:     c.RootModel,
			Error:         c.Error,
			UsageSummary:  c.Usage.toUsageSummary(c.RootModel),
			ExecutionTime: time.Duration(c.ExecutionTime * float64(time.Second)),
		}
	}

	return rlm.REPLResult{
		Stdout:        raw.Stdout,
		Stderr:        raw.Stderr,
		Locals:        raw.Locals,
		FinalAnswer:   raw.FinalAnswer,
		ExecutionTime: elapsed,
		LLMCalls:      calls,
	}, nil
}

type rawLLMCall struct {
	Prompt        string   `json:"prompt"`
	Response      string   `json:"response"`
	RootModel     string   `json:"root_model"`
	Error         string   `json:"error"`
	Usage         rawUsage `json:"usage"`
	ExecutionTime float64  `json:"execution_time"`
}

type rawUsage struct {
	TotalCalls        int `json:"total_calls"`
	TotalInputTokens  int `json:"total_input_tokens"`
	TotalOutputTokens int `json:"total_output_tokens"`
}

func (u rawUsage) toUsageSummary(model string) types.UsageSummary {
	if u.TotalCalls == 0 && u.TotalInputTokens == 0 && u.TotalOutputTokens == 0 {
		return types.UsageSummary{ModelUsageSummaries: map[string]types.ModelUsageSummary{}}
	}
	return types.UsageSummary{
		ModelUsageSummaries: map[string]types.ModelUsageSummary{
			model: {
				TotalCalls:        u.TotalCalls,
				TotalInputTokens:  u.TotalInputTokens,
				TotalOutputTokens: u.TotalOutputTokens,
			},
		},
	}
}

func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// Cleanup stops and removes the container and deletes the temporary workspace.
func (r *DockerREPL) Cleanup(ctx context.Context) error {
	var errs []error

	if r.containerID != "" {
		slog.Debug("DockerREPL stopping container", "container_id", r.containerID)
		_, _, _ = r.runner.Run(ctx, "docker", "stop", r.containerID)
		r.containerID = ""
	}

	if r.workspace != "" {
		slog.Debug("DockerREPL removing workspace", "workspace", r.workspace)
		if err := os.RemoveAll(r.workspace); err != nil {
			slog.Error("DockerREPL failed to remove workspace", "workspace", r.workspace, "error", err)
			errs = append(errs, fmt.Errorf("remove workspace: %w", err))
		}
		r.workspace = ""
	}

	if len(errs) > 0 {
		return errs[0]
	}
	return nil
}
