package environment

import (
	"context"
	"errors"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"rlm-golang/internal/rlm"
)

// fakeRunner records every command and returns configured responses.
type fakeRunner struct {
	calls     [][]string
	stdout    []byte
	stderr    []byte
	runErr    error
	callIndex int
	responses []runnerResponse
}

type runnerResponse struct {
	stdout []byte
	stderr []byte
	err    error
}

func (f *fakeRunner) Run(ctx context.Context, name string, arg ...string) ([]byte, []byte, error) {
	cmd := append([]string{name}, arg...)
	f.calls = append(f.calls, cmd)

	if len(f.responses) > 0 {
		idx := f.callIndex
		if idx >= len(f.responses) {
			idx = len(f.responses) - 1
		}
		f.callIndex++
		resp := f.responses[idx]
		return resp.stdout, resp.stderr, resp.err
	}
	return f.stdout, f.stderr, f.runErr
}

func newTestREPL(t *testing.T, runner *fakeRunner) *DockerREPL {
	t.Helper()
	workspace := t.TempDir()
	return NewDockerREPLWithRunner(DockerREPLConfig{
		Image:         "rlm-sandbox",
		LMHandlerHost: "127.0.0.1",
		LMHandlerPort: 50051,
		Depth:         1,
		Workspace:     workspace,
	}, runner)
}

func TestDockerREPLSetupRunsContainer(t *testing.T) {
	runner := &fakeRunner{
		responses: []runnerResponse{{
			stdout: []byte("container123\n"),
		}},
	}
	repl := newTestREPL(t, runner)

	if err := repl.Setup(context.Background()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	if len(runner.calls) != 1 {
		t.Fatalf("expected 1 command, got %d", len(runner.calls))
	}
	call := runner.calls[0]
	if call[0] != "docker" || call[1] != "run" || call[2] != "-d" || call[3] != "--rm" {
		t.Errorf("unexpected docker run invocation: %v", call)
	}

	var foundVolume, foundHost bool
	for i, arg := range call {
		if arg == "-v" && i+1 < len(call) && strings.HasPrefix(call[i+1], repl.Workspace()) && strings.HasSuffix(call[i+1], ":/workspace") {
			foundVolume = true
		}
		if arg == "--add-host" && i+1 < len(call) && call[i+1] == "host.docker.internal:host-gateway" {
			foundHost = true
		}
	}
	if !foundVolume {
		t.Errorf("missing workspace volume mount, got %v", call)
	}
	if !foundHost {
		t.Errorf("missing host-gateway add-host, got %v", call)
	}

	if repl.ContainerID() != "container123" {
		t.Errorf("containerID = %q, want %q", repl.ContainerID(), "container123")
	}
}

func TestDockerREPLSetupReturnsError(t *testing.T) {
	runner := &fakeRunner{responses: []runnerResponse{{err: errors.New("docker run failed")}}}
	repl := newTestREPL(t, runner)

	if err := repl.Setup(context.Background()); err == nil {
		t.Fatalf("expected error from Setup")
	}
}

func TestDockerREPLLoadContextWritesString(t *testing.T) {
	runner := &fakeRunner{}
	repl := newTestREPL(t, runner)

	if err := repl.LoadContext(context.Background(), "The magic number is 7"); err != nil {
		t.Fatalf("LoadContext: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(repl.Workspace(), "context.txt"))
	if err != nil {
		t.Fatalf("reading context.txt: %v", err)
	}
	if string(data) != "The magic number is 7" {
		t.Errorf("context.txt = %q, want %q", string(data), "The magic number is 7")
	}
}

func TestDockerREPLLoadContextWritesCSV(t *testing.T) {
	runner := &fakeRunner{}
	repl := newTestREPL(t, runner)

	csvData := "name,age,city\nJohn,30,New York\nJane,25,Los Angeles"
	if err := repl.LoadContext(context.Background(), csvData); err != nil {
		t.Fatalf("LoadContext: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(repl.Workspace(), "context.csv"))
	if err != nil {
		t.Fatalf("reading context.csv: %v", err)
	}
	if string(data) != csvData {
		t.Errorf("context.csv = %q, want %q", string(data), csvData)
	}

	// Make sure context.txt is also written
	txtData, err := os.ReadFile(filepath.Join(repl.Workspace(), "context.txt"))
	if err != nil {
		t.Fatalf("reading context.txt: %v", err)
	}
	if string(txtData) != csvData {
		t.Errorf("context.txt = %q, want %q", string(txtData), csvData)
	}
}

func TestDockerREPLLoadContextWritesJSON(t *testing.T) {
	runner := &fakeRunner{}
	repl := newTestREPL(t, runner)

	payload := map[string]any{"key": "value", "items": []any{1, 2}}
	if err := repl.LoadContext(context.Background(), payload); err != nil {
		t.Fatalf("LoadContext: %v", err)
	}

	data, err := os.ReadFile(filepath.Join(repl.Workspace(), "context.json"))
	if err != nil {
		t.Fatalf("reading context.json: %v", err)
	}
	if !strings.Contains(string(data), `"key"`) || !strings.Contains(string(data), `"value"`) {
		t.Errorf("context.json missing expected fields: %s", data)
	}
}

func TestDockerREPLExecuteCodeParsesResult(t *testing.T) {
	output := `{"stdout":"42\n","stderr":"","locals":{"x":"42"},"final_answer":"42","llm_calls":[]}`
	runner := &fakeRunner{
		responses: []runnerResponse{
			{stdout: []byte("container123\n")},
			{stdout: []byte(output)},
		},
	}
	repl := newTestREPL(t, runner)
	if err := repl.Setup(context.Background()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	result, err := repl.ExecuteCode(context.Background(), "print(42)\nanswer['content']='42'\nanswer['ready']=True")
	if err != nil {
		t.Fatalf("ExecuteCode: %v", err)
	}

	if result.Stdout != "42\n" {
		t.Errorf("Stdout = %q, want %q", result.Stdout, "42\n")
	}
	if result.FinalAnswer != "42" {
		t.Errorf("FinalAnswer = %q, want %q", result.FinalAnswer, "42")
	}
	if result.Locals["x"] != "42" {
		t.Errorf("Locals[x] = %q, want %q", result.Locals["x"], "42")
	}

	call := runner.calls[len(runner.calls)-1]
	if call[0] != "docker" || call[1] != "exec" {
		t.Errorf("expected docker exec, got %v", call)
	}
	if !slices.Contains(call, "python") || !slices.Contains(call, "-c") {
		t.Errorf("expected python -c in exec args, got %v", call)
	}
}

func TestDockerREPLExecuteCodeReturnsErrorOnNonZeroExit(t *testing.T) {
	runner := &fakeRunner{
		responses: []runnerResponse{
			{stdout: []byte("container123\n")},
			{stdout: []byte("partial output"), stderr: []byte("boom"), err: errors.New("exit status 1")},
		},
	}
	repl := newTestREPL(t, runner)
	if err := repl.Setup(context.Background()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	result, err := repl.ExecuteCode(context.Background(), "raise Exception('boom')")
	if err == nil {
		t.Fatalf("expected error from ExecuteCode")
	}
	if result.Stdout != "partial output" {
		t.Errorf("Stdout = %q, want %q", result.Stdout, "partial output")
	}
	if !strings.Contains(err.Error(), "docker exec") {
		t.Errorf("error should mention docker exec, got %v", err)
	}
}

func TestDockerREPLExecuteCodeReturnsErrorOnInvalidJSON(t *testing.T) {
	runner := &fakeRunner{
		responses: []runnerResponse{
			{stdout: []byte("container123\n")},
			{stdout: []byte("not json")},
		},
	}
	repl := newTestREPL(t, runner)
	if err := repl.Setup(context.Background()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	_, err := repl.ExecuteCode(context.Background(), "print('hello')")
	if err == nil {
		t.Fatalf("expected error from invalid JSON")
	}
	if !strings.Contains(err.Error(), "parse exec output") {
		t.Errorf("error should mention parsing, got %v", err)
	}
}

func TestDockerREPLCleanupStopsContainer(t *testing.T) {
	runner := &fakeRunner{
		responses: []runnerResponse{
			{stdout: []byte("container123\n")},
			{}, // docker stop
		},
	}
	repl := newTestREPL(t, runner)
	if err := repl.Setup(context.Background()); err != nil {
		t.Fatalf("Setup: %v", err)
	}

	if err := repl.Cleanup(context.Background()); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}

	stop := runner.calls[len(runner.calls)-1]
	if stop[0] != "docker" || stop[1] != "stop" || stop[2] != "container123" {
		t.Errorf("expected docker stop container123, got %v", stop)
	}

	if _, err := os.Stat(repl.Workspace()); !os.IsNotExist(err) {
		t.Errorf("workspace %q was not removed", repl.Workspace())
	}
}

func TestDockerREPLCleanupIsIdempotent(t *testing.T) {
	runner := &fakeRunner{
		responses: []runnerResponse{
			{stdout: []byte("container123\n")},
			{}, // docker stop
		},
	}
	repl := newTestREPL(t, runner)
	if err := repl.Setup(context.Background()); err != nil {
		t.Fatalf("Setup: %v", err)
	}
	if err := repl.Cleanup(context.Background()); err != nil {
		t.Fatalf("first Cleanup: %v", err)
	}
	if err := repl.Cleanup(context.Background()); err != nil {
		t.Fatalf("second Cleanup: %v", err)
	}
}

func TestDockerREPLImplementsEnvironment(t *testing.T) {
	var _ rlm.Environment = (*DockerREPL)(nil)
}
