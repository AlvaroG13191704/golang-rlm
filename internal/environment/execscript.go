// Package environment implements the Docker-based Python REPL used by the RLM
// orchestrator.
package environment

import (
	"encoding/base64"
	"strconv"
	"strings"
)

// ExecConfig parametrizes the Python exec script generated for each
// ExecuteCode call.
type ExecConfig struct {
	// Code is the Python source supplied by the model.
	Code string
	// LMHandlerPort is the host gRPC LMService/RLMService port reachable from
	// the container via host.docker.internal.
	LMHandlerPort int
	// Depth is the current recursion depth; forwarded to gRPC requests.
	Depth int
	// MaxDepth is the maximum recursion depth. At or past this cap, rlm_query
	// falls back to llm_query.
	MaxDepth int
	// WorkspaceID distinguishes state files for sibling containers sharing the
	// same workspace directory.
	WorkspaceID string
}

// BuildExecScript returns the Python source run inside the container via
// `docker exec ... python -c`. It loads prior state, executes the supplied
// code, and prints a JSON result to stdout.
func BuildExecScript(cfg ExecConfig) string {
	codeB64 := base64.StdEncoding.EncodeToString([]byte(cfg.Code))
	workspaceID := cfg.WorkspaceID
	if workspaceID == "" {
		workspaceID = "default"
	}
	script := strings.ReplaceAll(execScriptTemplate, "__CODE_B64__", codeB64)
	script = strings.ReplaceAll(script, "__LM_PORT__", strconv.Itoa(cfg.LMHandlerPort))
	script = strings.ReplaceAll(script, "__DEPTH__", strconv.Itoa(cfg.Depth))
	script = strings.ReplaceAll(script, "__MAX_DEPTH__", strconv.Itoa(cfg.MaxDepth))
	script = strings.ReplaceAll(script, "__WORKSPACE_ID__", workspaceID)
	return script
}

const execScriptTemplate = `import sys, io, json, base64, traceback, os, dill, grpc, logging
from rlm.v1 import lm_service_pb2 as lm_pb2
from rlm.v1 import lm_service_pb2_grpc as lm_grpc
from rlm.v1 import rlm_service_pb2 as rlm_pb2
from rlm.v1 import rlm_service_pb2_grpc as rlm_grpc
from rlm.v1 import types_pb2

HOST = "host.docker.internal"
PORT = __LM_PORT__
DEPTH = __DEPTH__
MAX_DEPTH = __MAX_DEPTH__
WORKSPACE_ID = "__WORKSPACE_ID__"
STATE = f"/workspace/state_d{DEPTH}_{WORKSPACE_ID}.dill"

logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("rlm.exec")

logger.info("exec_script started depth=%s max_depth=%s port=%s workspace_id=%s", DEPTH, MAX_DEPTH, PORT, WORKSPACE_ID)

_channel = grpc.insecure_channel(f"{HOST}:{PORT}")
_lm_stub = lm_grpc.LMServiceStub(_channel)
_rlm_stub = rlm_grpc.RLMServiceStub(_channel)
_llm_calls = []


def _usage_to_dict(u):
    if u is None:
        return None
    return {
        "total_calls": u.total_calls,
        "total_input_tokens": u.total_input_tokens,
        "total_output_tokens": u.total_output_tokens,
    }

def _duration_to_seconds(d):
    if d is None:
        return 0.0
    return d.seconds + d.nanos / 1e9

def _encode_prompt(prompt):
    if isinstance(prompt, str):
        return prompt, types_pb2.PromptType.PROMPT_RAW
    return json.dumps(prompt), types_pb2.PromptType.PROMPT_JSON

def _record_call(prompt, response, root_model, error, usage, execution_time):
    _llm_calls.append({
        "prompt": prompt,
        "response": response,
        "root_model": root_model,
        "error": error,
        "usage": _usage_to_dict(usage),
        "execution_time": _duration_to_seconds(execution_time),
    })

def _prompt_summary(prompt):
    if isinstance(prompt, str):
        return f"str_len={len(prompt)}"
    raw = json.dumps(prompt)
    return f"json_len={len(raw)}"

def _decode_text(data):
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")

def llm_query(prompt, model=None):
    logger.info("llm_query called model=%s %s", model or "", _prompt_summary(prompt))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("llm_query prompt: %s", str(prompt)[:200] + "..." if len(str(prompt)) > 200 else prompt)
    try:
        p, pt = _encode_prompt(prompt)
        req = lm_pb2.CompleteRequest(prompt=p, prompt_type=pt, model=model or "", depth=DEPTH)
        resp = _lm_stub.Complete(req, timeout=300)
        _record_call(prompt, resp.content, resp.root_model, resp.error, resp.usage, resp.execution_time)
        if resp.error:
            logger.error("llm_query error: %s", resp.error)
            return f"Error: {resp.error}"
        logger.info("llm_query completed model=%s output_len=%s", resp.root_model, len(resp.content))
        return resp.content
    except Exception as e:
        logger.exception("llm_query failed")
        return f"Error: {e}"

def llm_query_batched(prompts, model=None):
    logger.info("llm_query_batched called count=%s model=%s", len(prompts), model or "")
    try:
        items = []
        for p in prompts:
            raw, pt = _encode_prompt(p)
            items.append(lm_pb2.CompleteRequest(prompt=raw, prompt_type=pt, model=model or "", depth=DEPTH))
        resp = _lm_stub.CompleteBatched(lm_pb2.BatchedRequest(items=items), timeout=300)
        results = []
        for i, r in enumerate(resp.responses):
            _record_call(prompts[i], r.content, r.root_model, r.error, r.usage, r.execution_time)
            results.append(r.content if not r.error else f"Error: {r.error}")
        logger.info("llm_query_batched completed count=%s", len(results))
        return results
    except Exception as e:
        logger.exception("llm_query_batched failed")
        return [f"Error: {e}"] * len(prompts)

def rlm_query(prompt, model=None):
    logger.info("rlm_query called depth=%s max_depth=%s model=%s %s", DEPTH, MAX_DEPTH, model or "", _prompt_summary(prompt))
    if logger.isEnabledFor(logging.DEBUG):
        logger.debug("rlm_query prompt: %s", str(prompt)[:200] + "..." if len(str(prompt)) > 200 else prompt)
    if DEPTH >= MAX_DEPTH:
        logger.info("rlm_query falling back to llm_query at max depth")
        return llm_query(prompt, model)
    try:
        p, pt = _encode_prompt(prompt)
        req = rlm_pb2.SubcallRequest(prompt=p, prompt_type=pt, model=model or "", depth=DEPTH)
        resp = _rlm_stub.Subcall(req, timeout=300)
        _record_call(prompt, resp.content, resp.root_model, resp.error, resp.usage, resp.execution_time)
        if resp.error:
            logger.error("rlm_query error: %s", resp.error)
            return f"Error: {resp.error}"
        logger.info("rlm_query completed model=%s output_len=%s", resp.root_model, len(resp.content))
        return resp.content
    except Exception as e:
        logger.exception("rlm_query failed")
        return f"Error: {e}"

def rlm_query_batched(prompts, model=None):
    logger.info("rlm_query_batched called count=%s depth=%s model=%s", len(prompts), DEPTH, model or "")
    if DEPTH >= MAX_DEPTH:
        logger.info("rlm_query_batched falling back to llm_query_batched at max depth")
        return llm_query_batched(prompts, model)
    try:
        items = []
        for p in prompts:
            raw, pt = _encode_prompt(p)
            items.append(rlm_pb2.SubcallRequest(prompt=raw, prompt_type=pt, model=model or "", depth=DEPTH))
        resp = _rlm_stub.SubcallBatched(rlm_pb2.BatchedSubcallRequest(items=items), timeout=300)
        results = []
        for i, r in enumerate(resp.responses):
            _record_call(prompts[i], r.content, r.root_model, r.error, r.usage, r.execution_time)
            results.append(r.content if not r.error else f"Error: {r.error}")
        logger.info("rlm_query_batched completed count=%s", len(results))
        return results
    except Exception as e:
        logger.exception("rlm_query_batched failed")
        return [f"Error: {e}"] * len(prompts)

def load_state():
    if os.path.exists(STATE):
        logger.info("loading state from %s", STATE)
        try:
            with open(STATE, "rb") as f:
                return dill.load(f)
        except Exception:
            logger.exception("failed to load state from %s", STATE)
    return {}

def save_state(s):
    clean = {k: v for k, v in s.items() if not k.startswith("_")}
    for k in list(clean.keys()):
        try:
            dill.dumps(clean[k])
        except Exception:
            logger.debug("dropping non-serializable state key %s", k)
            del clean[k]
    with open(STATE, "wb") as f:
        dill.dump(clean, f)
    logger.info("saved state to %s keys=%s", STATE, len(clean))

_locals = load_state()

if "answer" not in _locals or not isinstance(_locals.get("answer"), dict):
    _locals["answer"] = {"content": "", "ready": False}

if os.path.exists("/workspace/context.txt"):
    logger.info("loading string context from /workspace/context.txt")
    with open("/workspace/context.txt", "rb") as f:
        _locals["context"] = _decode_text(f.read())
    logger.info("loaded string context length=%s preview=%r", len(_locals["context"]), _locals["context"][:100])
elif os.path.exists("/workspace/context.json"):
    logger.info("loading JSON context from /workspace/context.json")
    with open("/workspace/context.json", "r") as f:
        _locals["context"] = json.load(f)

if os.path.exists("/workspace/context.csv"):
    logger.info("loading CSV context from /workspace/context.csv")
    try:
        import pandas as pd
        _locals["df"] = pd.read_csv("/workspace/context.csv")
        logger.info("loaded CSV context as DataFrame 'df' with shape %s", _locals["df"].shape)
    except Exception as e:
        logger.warning("failed to load CSV as DataFrame: %s", e)

def SHOW_VARS():
    available = {k: type(v).__name__ for k, v in _locals.items() if not k.startswith("_") and k != "answer"}
    if not available:
        return "No variables created yet. Use repl blocks to create variables."
    return f"Available variables: {available}"

_globals = {
    "__builtins__": __builtins__,
    "__name__": "__main__",
    "llm_query": llm_query,
    "llm_query_batched": llm_query_batched,
    "rlm_query": rlm_query,
    "rlm_query_batched": rlm_query_batched,
    "SHOW_VARS": SHOW_VARS,
}

code = base64.b64decode("__CODE_B64__").decode()
logger.info("executing user code length=%s", len(code))
if logger.isEnabledFor(logging.DEBUG):
    logger.debug("user code:\n%s", code)

stdout_buf, stderr_buf = io.StringIO(), io.StringIO()
old_stdout, old_stderr = sys.stdout, sys.stderr

try:
    sys.stdout, sys.stderr = stdout_buf, stderr_buf
    combined = {**_globals, **_locals}
    exec(code, combined, combined)
    for k, v in combined.items():
        if k not in _globals and not k.startswith("_"):
            _locals[k] = v
except Exception:
    logger.exception("user code raised an exception")
    traceback.print_exc(file=stderr_buf)
finally:
    sys.stdout, sys.stderr = old_stdout, old_stderr

save_state(_locals)
_ans = _locals.get("answer") if isinstance(_locals.get("answer"), dict) else None
_final = None
if _ans is not None and _ans.get("ready"):
    _final = str(_ans.get("content", ""))
    logger.info("final answer submitted length=%s", len(_final))
else:
    logger.info("no final answer submitted")

print(json.dumps({
    "stdout": stdout_buf.getvalue(),
    "stderr": stderr_buf.getvalue(),
    "locals": {k: repr(v) for k, v in _locals.items() if not k.startswith("_")},
    "final_answer": _final,
    "llm_calls": _llm_calls,
}, ensure_ascii=False))
`
