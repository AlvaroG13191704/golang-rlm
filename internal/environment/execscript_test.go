package environment

import (
	"encoding/base64"
	"strings"
	"testing"
)

func TestBuildExecScriptContainsRequiredImports(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "print('hi')", LMHandlerPort: 50051, Depth: 1, MaxDepth: 2, WorkspaceID: "abc"})

	for _, want := range []string{
		"import sys, io, json, base64, traceback, os, dill, grpc",
		"from rlm.v1 import lm_service_pb2 as lm_pb2",
		"from rlm.v1 import lm_service_pb2_grpc as lm_grpc",
		"from rlm.v1 import rlm_service_pb2 as rlm_pb2",
		"from rlm.v1 import rlm_service_pb2_grpc as rlm_grpc",
	} {
		if !strings.Contains(script, want) {
			t.Errorf("script missing %q", want)
		}
	}
}

func TestBuildExecScriptEmbedsCode(t *testing.T) {
	code := "x = 1 + 1\nprint(x)"
	script := BuildExecScript(ExecConfig{Code: code, LMHandlerPort: 50051, Depth: 1, MaxDepth: 2})

	want := base64.StdEncoding.EncodeToString([]byte(code))
	if !strings.Contains(script, want) {
		t.Errorf("script does not contain base64-encoded code")
	}
	if !strings.Contains(script, "base64.b64decode") {
		t.Errorf("script missing base64 decode step")
	}
}

func TestBuildExecScriptExposesNamespace(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "pass", LMHandlerPort: 50051, Depth: 2, MaxDepth: 3})

	for _, fn := range []string{
		"def llm_query",
		"def llm_query_batched",
		"def rlm_query",
		"def rlm_query_batched",
		"def SHOW_VARS",
	} {
		if !strings.Contains(script, fn) {
			t.Errorf("script missing %q", fn)
		}
	}
	if !strings.Contains(script, "\"content\": \"\"") {
		t.Errorf("script missing answer content scaffold")
	}
	if !strings.Contains(script, "\"ready\": False") {
		t.Errorf("script missing answer ready scaffold")
	}
}

func TestBuildExecScriptRlmQueryUsesGrpc(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "pass", LMHandlerPort: 50051, Depth: 1, MaxDepth: 2})

	if !strings.Contains(script, "_rlm_stub.Subcall") {
		t.Errorf("rlm_query does not call RLMService.Subcall")
	}
	if !strings.Contains(script, "_rlm_stub.SubcallBatched") {
		t.Errorf("rlm_query_batched does not call RLMService.SubcallBatched")
	}
	if strings.Contains(script, "Error: rlm_query not implemented") {
		t.Errorf("rlm_query still contains stub message")
	}
}

func TestBuildExecScriptRlmQueryFallbackAtMaxDepth(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "pass", LMHandlerPort: 50051, Depth: 2, MaxDepth: 2})

	if !strings.Contains(script, "if DEPTH >= MAX_DEPTH:") {
		t.Errorf("script missing max-depth guard")
	}
	if !strings.Contains(script, "return llm_query(prompt, model)") {
		t.Errorf("script does not fall back to llm_query")
	}
}

func TestBuildExecScriptUsesHostGateway(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "pass", LMHandlerPort: 8080, Depth: 3, MaxDepth: 4})

	if !strings.Contains(script, "host.docker.internal") {
		t.Errorf("script does not reference host.docker.internal")
	}
	if !strings.Contains(script, "PORT = 8080") {
		t.Errorf("script does not configure the LM handler port")
	}
}

func TestBuildExecScriptLoadsAndSavesState(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "pass", LMHandlerPort: 50051, Depth: 1, MaxDepth: 2, WorkspaceID: "abc"})

	if !strings.Contains(script, `WORKSPACE_ID = "abc"`) {
		t.Errorf("script did not embed workspace id")
	}
	if !strings.Contains(script, "state_d{DEPTH}_{WORKSPACE_ID}.dill") {
		t.Errorf("script missing depth-prefixed state file path")
	}
	if !strings.Contains(script, "def load_state()") {
		t.Errorf("script missing load_state function")
	}
	if !strings.Contains(script, "def save_state") {
		t.Errorf("script missing save_state function")
	}
}

func TestBuildExecScriptPrintsJsonResult(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "pass", LMHandlerPort: 50051, Depth: 1, MaxDepth: 2})

	if !strings.Contains(script, "print(json.dumps({") {
		t.Errorf("script missing final JSON output")
	}
	for _, key := range []string{"\"stdout\"", "\"stderr\"", "\"locals\"", "\"final_answer\"", "\"llm_calls\""} {
		if !strings.Contains(script, key) {
			t.Errorf("script missing result key %q", key)
		}
	}
}

func TestBuildExecScriptLoadsContextFiles(t *testing.T) {
	script := BuildExecScript(ExecConfig{Code: "pass", LMHandlerPort: 50051, Depth: 1, MaxDepth: 2})

	if !strings.Contains(script, "/workspace/context.txt") {
		t.Errorf("script missing context.txt load")
	}
	if !strings.Contains(script, "/workspace/context.json") {
		t.Errorf("script missing context.json load")
	}
}
