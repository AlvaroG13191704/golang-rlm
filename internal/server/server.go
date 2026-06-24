// Package server wires the gRPC services used between the container and the
// host. It hosts LMService (plain completions) and RLMService (recursive child
// RLM calls) on a single gRPC server.
package server

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net"

	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
	"google.golang.org/protobuf/types/known/durationpb"

	"golang.org/x/sync/errgroup"

	rlmv1 "rlm-golang/gen/rlm/v1"
	"rlm-golang/internal/types"
)

const defaultSubcallConcurrency = 16

// SubcallHandler implements the recursive RLM call. The orchestrator will
// provide a real implementation; tests substitute a fake.
type SubcallHandler interface {
	Subcall(ctx context.Context, prompt any, model string, depth int32) (types.RLMChatCompletion, error)
}

// RLMHandlerOption configures an RLMHandler.
type RLMHandlerOption func(*RLMHandler)

// WithSubcallConcurrency sets the maximum number of concurrent subcalls in a
// batched request.
func WithSubcallConcurrency(n int) RLMHandlerOption {
	return func(h *RLMHandler) {
		h.concurrency = n
	}
}

// RLMHandler serves recursive child RLM calls over gRPC.
type RLMHandler struct {
	rlmv1.UnimplementedRLMServiceServer
	handler     SubcallHandler
	concurrency int
}

// NewRLMHandler creates an RLMService implementation backed by the given
// subcall handler.
func NewRLMHandler(handler SubcallHandler, opts ...RLMHandlerOption) *RLMHandler {
	h := &RLMHandler{
		handler:     handler,
		concurrency: defaultSubcallConcurrency,
	}
	for _, opt := range opts {
		opt(h)
	}
	return h
}

// Subcall implements RLMService.Subcall.
func (h *RLMHandler) Subcall(ctx context.Context, req *rlmv1.SubcallRequest) (*rlmv1.SubcallResponse, error) {
	return h.subcallOne(ctx, req), nil
}

// SubcallBatched implements RLMService.SubcallBatched.
func (h *RLMHandler) SubcallBatched(ctx context.Context, req *rlmv1.BatchedSubcallRequest) (*rlmv1.BatchedSubcallResponse, error) {
	items := req.GetItems()
	responses := make([]*rlmv1.SubcallResponse, len(items))

	g, ctx := errgroup.WithContext(ctx)
	g.SetLimit(h.concurrency)

	for i := range items {
		g.Go(func() error {
			responses[i] = h.subcallOne(ctx, items[i])
			return nil
		})
	}

	if err := g.Wait(); err != nil {
		return nil, status.Error(codes.Internal, err.Error())
	}

	return &rlmv1.BatchedSubcallResponse{Responses: responses}, nil
}

func (h *RLMHandler) subcallOne(ctx context.Context, req *rlmv1.SubcallRequest) *rlmv1.SubcallResponse {
	prompt, err := promptFromRequest(req)
	if err != nil {
		return &rlmv1.SubcallResponse{Error: err.Error()}
	}

	completion, err := h.handler.Subcall(ctx, prompt, req.GetModel(), req.GetDepth())
	if err != nil {
		return &rlmv1.SubcallResponse{
			RootModel: completion.RootModel,
			Error:     err.Error(),
		}
	}

	return &rlmv1.SubcallResponse{
		Content:       completion.Response,
		RootModel:     completion.RootModel,
		Usage:         toProtoModelUsage(completion.UsageSummary.ModelUsageSummaries[completion.RootModel]),
		ExecutionTime: durationpb.New(completion.ExecutionTime),
	}
}

// Server hosts gRPC services for the RLM runtime.
type Server struct {
	lm  rlmv1.LMServiceServer
	rlm rlmv1.RLMServiceServer

	server   *grpc.Server
	listener net.Listener
}

// NewServer creates a server that will register the supplied service
// implementations. Either argument may be nil if that service is not needed.
func NewServer(lm rlmv1.LMServiceServer, rlm rlmv1.RLMServiceServer) *Server {
	return &Server{lm: lm, rlm: rlm}
}

// RegisterAndStart creates a gRPC server, registers the configured services,
// and starts serving on the provided listener.
func (s *Server) RegisterAndStart(lis net.Listener) error {
	s.server = grpc.NewServer(grpc.UnaryInterceptor(unaryLoggingInterceptor))
	if s.lm != nil {
		rlmv1.RegisterLMServiceServer(s.server, s.lm)
	}
	if s.rlm != nil {
		rlmv1.RegisterRLMServiceServer(s.server, s.rlm)
	}
	s.listener = lis

	srv := s.server
	go func() {
		if err := srv.Serve(lis); err != nil {
			// Server stopped intentionally via Stop; ignore.
		}
	}()
	return nil
}

func unaryLoggingInterceptor(ctx context.Context, req any, info *grpc.UnaryServerInfo, handler grpc.UnaryHandler) (any, error) {
	model, promptLen, batchSize := requestSummary(req)
	attrs := []any{
		slog.String("method", info.FullMethod),
		slog.String("model", model),
		slog.Int("prompt_len", promptLen),
	}
	if batchSize > 0 {
		attrs = append(attrs, slog.Int("batch_size", batchSize))
	}
	slog.Debug("gRPC request started", attrs...)

	resp, err := handler(ctx, req)

	if err != nil {
		slog.Error("gRPC request failed", "method", info.FullMethod, "error", err)
		return resp, err
	}

	resolvedModel, respErr, inputTokens, outputTokens := responseSummary(resp)
	logModel := model
	if resolvedModel != "" {
		logModel = resolvedModel
	}

	callType := "unknown"
	switch info.FullMethod {
	case "/rlm.v1.LMService/Complete", "/rlm.v1.LMService/CompleteBatched":
		callType = "llm_query"
	case "/rlm.v1.RLMService/Subcall", "/rlm.v1.RLMService/SubcallBatched":
		callType = "rlm_query"
	}

	logAttrs := []any{
		slog.String("call_type", callType),
		slog.String("method", info.FullMethod),
		slog.String("model", logModel),
	}
	if respErr != "" {
		logAttrs = append(logAttrs, slog.String("response_error", respErr))
	}
	if inputTokens >= 0 {
		logAttrs = append(logAttrs, slog.Int("input_tokens", inputTokens))
	}
	if outputTokens >= 0 {
		logAttrs = append(logAttrs, slog.Int("output_tokens", outputTokens))
	}
	slog.Info("gRPC request completed", logAttrs...)

	return resp, nil
}

func requestSummary(req any) (model string, promptLen int, batchSize int) {
	switch r := req.(type) {
	case *rlmv1.CompleteRequest:
		return r.GetModel(), len(r.GetPrompt()), 0
	case *rlmv1.SubcallRequest:
		return r.GetModel(), len(r.GetPrompt()), 0
	case *rlmv1.BatchedRequest:
		items := r.GetItems()
		maxLen := 0
		for _, item := range items {
			if l := len(item.GetPrompt()); l > maxLen {
				maxLen = l
			}
		}
		return "", maxLen, len(items)
	case *rlmv1.BatchedSubcallRequest:
		items := r.GetItems()
		maxLen := 0
		for _, item := range items {
			if l := len(item.GetPrompt()); l > maxLen {
				maxLen = l
			}
		}
		return "", maxLen, len(items)
	default:
		return "", 0, 0
	}
}

func responseSummary(resp any) (model string, respErr string, inputTokens, outputTokens int) {
	inputTokens = -1
	outputTokens = -1
	switch r := resp.(type) {
	case *rlmv1.CompleteResponse:
		if r.GetUsage() != nil {
			inputTokens = int(r.GetUsage().GetTotalInputTokens())
			outputTokens = int(r.GetUsage().GetTotalOutputTokens())
		}
		return r.GetRootModel(), r.GetError(), inputTokens, outputTokens
	case *rlmv1.SubcallResponse:
		if r.GetUsage() != nil {
			inputTokens = int(r.GetUsage().GetTotalInputTokens())
			outputTokens = int(r.GetUsage().GetTotalOutputTokens())
		}
		return r.GetRootModel(), r.GetError(), inputTokens, outputTokens
	case *rlmv1.BatchedResponse:
		var inToks, outToks int
		hasUsage := false
		resolvedModel := ""
		for _, item := range r.GetResponses() {
			if resolvedModel == "" && item.GetRootModel() != "" {
				resolvedModel = item.GetRootModel()
			}
			if item.GetUsage() != nil {
				hasUsage = true
				inToks += int(item.GetUsage().GetTotalInputTokens())
				outToks += int(item.GetUsage().GetTotalOutputTokens())
			}
		}
		if hasUsage {
			return resolvedModel, "", inToks, outToks
		}
		return resolvedModel, "", -1, -1
	case *rlmv1.BatchedSubcallResponse:
		var inToks, outToks int
		hasUsage := false
		resolvedModel := ""
		for _, item := range r.GetResponses() {
			if resolvedModel == "" && item.GetRootModel() != "" {
				resolvedModel = item.GetRootModel()
			}
			if item.GetUsage() != nil {
				hasUsage = true
				inToks += int(item.GetUsage().GetTotalInputTokens())
				outToks += int(item.GetUsage().GetTotalOutputTokens())
			}
		}
		if hasUsage {
			return resolvedModel, "", inToks, outToks
		}
		return resolvedModel, "", -1, -1
	default:
		return "", "", inputTokens, outputTokens
	}
}

// Stop stops the gRPC server.
func (s *Server) Stop() error {
	if s.server == nil {
		return nil
	}
	s.server.Stop()
	s.server = nil
	s.listener = nil
	return nil
}

func promptFromRequest(req interface {
	GetPrompt() string
	GetPromptType() rlmv1.PromptType
}) (any, error) {
	if req.GetPromptType() == rlmv1.PromptType_PROMPT_JSON {
		var v any
		if err := json.Unmarshal([]byte(req.GetPrompt()), &v); err != nil {
			return nil, fmt.Errorf("unmarshal json prompt: %w", err)
		}
		return v, nil
	}
	return req.GetPrompt(), nil
}

func toProtoModelUsage(u types.ModelUsageSummary) *rlmv1.ModelUsageSummary {
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
