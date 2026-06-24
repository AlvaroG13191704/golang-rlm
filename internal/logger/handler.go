package logger

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"strings"
	"sync"
)

// ColorHandler is a custom slog.Handler that formats logs with ANSI colors,
// bold messages, and clean indentation for multiline values like errors/stderr.
type ColorHandler struct {
	w      io.Writer
	opts   slog.HandlerOptions
	attrs  []slog.Attr
	groups []string
	mu     sync.Mutex
}

// NewColorHandler creates a new ColorHandler.
func NewColorHandler(w io.Writer, opts slog.HandlerOptions) *ColorHandler {
	return &ColorHandler{
		w:    w,
		opts: opts,
	}
}

// Enabled implements slog.Handler.
func (h *ColorHandler) Enabled(ctx context.Context, level slog.Level) bool {
	minLevel := slog.LevelInfo
	if h.opts.Level != nil {
		minLevel = h.opts.Level.Level()
	}
	return level >= minLevel
}

// Handle implements slog.Handler.
func (h *ColorHandler) Handle(ctx context.Context, r slog.Record) error {
	h.mu.Lock()
	defer h.mu.Unlock()

	timeStr := r.Time.Format("15:04:05.000")

	// ANSI colors
	const (
		reset     = "\033[0m"
		bold      = "\033[1m"
		gray      = "\033[90m"
		red       = "\033[31m"
		green     = "\033[32m"
		yellow    = "\033[33m"
		cyan      = "\033[36m"
		lightGray = "\033[37m"
	)

	var levelColor, levelStr string
	switch r.Level {
	case slog.LevelDebug:
		levelColor = cyan
		levelStr = "DEBUG"
	case slog.LevelInfo:
		levelColor = green
		levelStr = "INFO "
	case slog.LevelWarn:
		levelColor = yellow
		levelStr = "WARN "
	case slog.LevelError:
		levelColor = red
		levelStr = "ERROR"
	default:
		levelColor = bold
		levelStr = r.Level.String()
	}

	// Print [Time] LEVEL
	fmt.Fprintf(h.w, "%s[%s]%s %s%s%s ", gray, timeStr, reset, levelColor, levelStr, reset)

	// Print Message in bold
	fmt.Fprintf(h.w, "%s%s%s", bold, r.Message, reset)

	// Print WithAttrs
	for _, attr := range h.attrs {
		printAttr(h.w, attr, gray, lightGray, reset)
	}

	// Print current record attrs
	r.Attrs(func(attr slog.Attr) bool {
		printAttr(h.w, attr, gray, lightGray, reset)
		return true
	})

	fmt.Fprintln(h.w)
	return nil
}

func printAttr(w io.Writer, attr slog.Attr, keyColor, valColor, resetColor string) {
	if attr.Value.Kind() == slog.KindGroup {
		groupName := attr.Key
		for _, child := range attr.Value.Group() {
			child.Key = groupName + "." + child.Key
			printAttr(w, child, keyColor, valColor, resetColor)
		}
		return
	}

	valStr := attr.Value.String()

	if attr.Key == "prompt" || attr.Key == "payload" || attr.Key == "context" || attr.Key == "root_prompt" {
		if len(valStr) > 200 {
			valStr = valStr[:200] + "..."
		}
	}

	// If it's a traceback or has newlines, format on new indented lines
	if attr.Key == "stderr" || attr.Key == "error" || attr.Key == "code" || strings.Contains(valStr, "\n") {
		lines := strings.Split(valStr, "\n")
		fmt.Fprintf(w, "\n  %s%s:%s", keyColor, attr.Key, resetColor)
		for _, line := range lines {
			if strings.TrimSpace(line) != "" {
				fmt.Fprintf(w, "\n    %s", line)
			}
		}
		return
	}

	// Truncate long inline values to keep layout clean
	if len(valStr) > 120 {
		valStr = valStr[:120] + "..."
	}

	fmt.Fprintf(w, " %s%s=%s%s%s", keyColor, attr.Key, resetColor, valStr, resetColor)
}

// WithAttrs implements slog.Handler.
func (h *ColorHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	newAttrs := make([]slog.Attr, len(h.attrs)+len(attrs))
	copy(newAttrs, h.attrs)
	copy(newAttrs[len(h.attrs):], attrs)
	return &ColorHandler{
		w:     h.w,
		opts:  h.opts,
		attrs: newAttrs,
	}
}

// WithGroup implements slog.Handler.
func (h *ColorHandler) WithGroup(name string) slog.Handler {
	// Simple pass-through
	return h
}
