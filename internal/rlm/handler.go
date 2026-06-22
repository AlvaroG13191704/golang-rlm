package rlm

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/durationpb"

	"golang.org/x/sync/errgroup"

	rlmv1 "rlm-golang/gen/rlm/v1"
	"rlm-golang/internal/client"
	"rlm-golang/internal/types"
)

const defaultBatchConcurrency = 16
const defaultHost = "127.0.0.1"

// LMHandlerOption configures an LMHandler.
type LMHandlerOption func(*LMHandler)

// WithDefaultModel sets the model name used for the default client.
func WithDefaultModel(name string) LMHandlerOption {
	return func(h *LMHandler) {
		h.defaultClient.name = name
	}
}

// WithOtherBackend registers a depth-1 backend client with its model name.
func WithOtherBackend(c client.BaseLM, name string) LMHandlerOption {
	return func(h *LMHandler) {
		h.otherBackend = &clientEntry{name: name, client: c}
	}
}

// WithBatchConcurrency sets the maximum number of concurrent prompts in a
// batched request.
func WithBatchConcurrency(n int) LMHandlerOption {
	return func(h *LMHandler) {
		h.batchConcurrency = n
	}
}

// WithHost sets the host address the gRPC server binds to.
func WithHost(host string) LMHandlerOption {
	return func(h *LMHandler) {
		h.host = host
	}
}

type clientEntry struct {
	name   string
	client client.BaseLM
}

// LMHandler serves LM completions over gRPC, routes requests by model name or
// depth, and aggregates usage across all registered clients.
type LMHandler struct {
	rlmv1.UnimplementedLMServiceServer
	defaultClient    clientEntry
	otherBackend     *clientEntry
	registry         map[string]clientEntry
	mu               sync.RWMutex
	batchConcurrency int
	host             string
	server           *grpc.Server
	listener         net.Listener
}

// NewLMHandler creates a handler with the given default client. Use options to
// set the default model name, register an other backend, and tune concurrency.
func NewLMHandler(defaultClient client.BaseLM, opts ...LMHandlerOption) *LMHandler {
	h := &LMHandler{
		defaultClient:    clientEntry{client: defaultClient},
		registry:         make(map[string]clientEntry),
		batchConcurrency: defaultBatchConcurrency,
		host:             defaultHost,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// RegisterClient registers a client for a specific model name.
func (h *LMHandler) RegisterClient(modelName string, c client.BaseLM) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.registry[modelName] = clientEntry{name: modelName, client: c}
}

// GetClient returns the client and its model name for the given model override
// and depth. Model name takes precedence; otherwise depth=1 routes to the other
// backend if present, and everything else falls back to the default client.
func (h *LMHandler) GetClient(model string, depth int32) (string, client.BaseLM) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	if model != "" {
		if entry, ok := h.registry[model]; ok {
			return entry.name, entry.client
		}
	}

	if depth == 1 && h.otherBackend != nil {
		return h.otherBackend.name, h.otherBackend.client
	}

	return h.defaultClient.name, h.defaultClient.client
}

// Complete implements LMService.Complete.
func (h *LMHandler) Complete(ctx context.Context, req *rlmv1.CompleteRequest) (*rlmv1.CompleteResponse, error) {
	return h.completeOne(ctx, req), nil
}

// CompleteBatched implements LMService.CompleteBatched.
func (h *LMHandler) CompleteBatched(ctx context.Context, req *rlmv1.BatchedRequest) (*rlmv1.BatchedResponse, error) {
	items := req.GetItems()
	responses := make([]*rlmv1.CompleteResponse, len(items))

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(h.batchConcurrency)

	for i := range items {
		g.Go(func() error {
			responses[i] = h.completeOne(ctx, items[i])
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &rlmv1.BatchedResponse{Responses: responses}, nil
}

// GetUsage implements LMService.GetUsage.
func (h *LMHandler) GetUsage(ctx context.Context, req *rlmv1.GetUsageRequest) (*rlmv1.GetUsageResponse, error) {
	usage := h.GetUsageSummary()
	return &rlmv1.GetUsageResponse{Usage: h.toProtoUsageSummary(usage)}, nil
}

// GetUsageSummary aggregates per-model usage across the default client, the
// other backend, and all explicitly registered clients.
func (h *LMHandler) GetUsageSummary() types.UsageSummary {
	h.mu.RLock()
	defer h.mu.RUnlock()

	merged := make(map[string]types.ModelUsageSummary)
	h.mergeUsage(merged, h.defaultClient)
	if h.otherBackend != nil {
		h.mergeUsage(merged, *h.otherBackend)
	}
	for _, entry := range h.registry {
		h.mergeUsage(merged, entry)
	}
	return types.UsageSummary{ModelUsageSummaries: merged}
}

func (h *LMHandler) mergeUsage(dst map[string]types.ModelUsageSummary, entry clientEntry) {
	if entry.name == "" || entry.client == nil {
		return
	}
	summary := entry.client.GetUsageSummary()
	if existing, ok := dst[entry.name]; ok {
		dst[entry.name] = client.AddModelUsage(existing, summary)
	} else {
		dst[entry.name] = summary
	}
}

// Start starts the internal gRPC server on an ephemeral port and returns the
// host and port.
func (h *LMHandler) Start() (string, int, error) {
	if h.server != nil {
		return h.host, h.listener.Addr().(*net.TCPAddr).Port, nil
	}

	lis, err := net.Listen("tcp", net.JoinHostPort(h.host, "0"))
	if err != nil {
		return "", 0, fmt.Errorf("lmhandler listen: %w", err)
	}

	h.listener = lis
	h.server = grpc.NewServer()
	rlmv1.RegisterLMServiceServer(h.server, h)

	go func() {
		if err := h.server.Serve(lis); err != nil {
			// Server stopped intentionally via Stop; ignore.
		}
	}()

	return h.host, lis.Addr().(*net.TCPAddr).Port, nil
}

// Stop stops the internal gRPC server.
func (h *LMHandler) Stop() error {
	if h.server == nil {
		return nil
	}
	h.server.Stop()
	h.server = nil
	h.listener = nil
	return nil
}

func (h *LMHandler) completeOne(ctx context.Context, req *rlmv1.CompleteRequest) *rlmv1.CompleteResponse {
	prompt, err := h.promptFromRequest(req)
	if err != nil {
		return &rlmv1.CompleteResponse{Error: err.Error()}
	}

	name, c := h.GetClient(req.GetModel(), req.GetDepth())
	start := time.Now()
	content, err := c.Completion(ctx, prompt)
	elapsed := time.Since(start)

	if err != nil {
		return &rlmv1.CompleteResponse{
			RootModel: name,
			Error:     err.Error(),
		}
	}

	lastUsage := c.GetLastUsage()
	return &rlmv1.CompleteResponse{
		Content:       content,
		RootModel:     name,
		Usage:         h.toProtoModelUsage(lastUsage),
		ExecutionTime: durationpb.New(elapsed),
	}
}

func (h *LMHandler) promptFromRequest(req *rlmv1.CompleteRequest) (any, error) {
	if req.GetPromptType() == rlmv1.PromptType_PROMPT_JSON {
		var v any
		if err := json.Unmarshal([]byte(req.GetPrompt()), &v); err != nil {
			return nil, fmt.Errorf("unmarshal json prompt: %w", err)
		}
		return v, nil
	}
	return req.GetPrompt(), nil
}

func (h *LMHandler) toProtoModelUsage(u types.ModelUsageSummary) *rlmv1.ModelUsageSummary {
	resp := &rlmv1.ModelUsageSummary{
		TotalCalls:        int32(u.TotalCalls),
		TotalInputTokens:  int32(u.TotalInputTokens),
		TotalOutputTokens: int32(u.TotalOutputTokens),
	}
	if u.TotalCost != nil {
		resp.TotalCost = u.TotalCost
	}
	return resp
}

func (h *LMHandler) toProtoUsageSummary(u types.UsageSummary) *rlmv1.UsageSummary {
	models := make(map[string]*rlmv1.ModelUsageSummary, len(u.ModelUsageSummaries))
	for name, summary := range u.ModelUsageSummaries {
		models[name] = h.toProtoModelUsage(summary)
	}
	return &rlmv1.UsageSummary{ModelUsageSummaries: models}
}
