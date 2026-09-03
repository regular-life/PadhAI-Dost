package middleware

import (
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/metrics"
)

type statusWriter struct {
	http.ResponseWriter
	statusCode int
	written    bool
}

// WriteHeader captures the HTTP status code on first invocation.
func (w *statusWriter) WriteHeader(code int) {
	if w.written {
		return
	}
	w.written = true
	w.statusCode = code
	w.ResponseWriter.WriteHeader(code)
}

// Write records that response body bytes have been written and forwards to the underlying writer.
func (w *statusWriter) Write(b []byte) (int, error) {
	if !w.written {
		w.written = true
	}
	return w.ResponseWriter.Write(b)
}

// Flush implements http.Flusher interface so streaming responses can flush immediately.
func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// LoggingMiddleware logs request path, method, status code, latency, and increments Prometheus metrics.
func LoggingMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		sw := &statusWriter{ResponseWriter: w, statusCode: http.StatusOK}
		next.ServeHTTP(sw, r)
		duration := time.Since(start)

		log.Printf("[HTTP] %s %s → %d (%s)", r.Method, r.URL.Path, sw.statusCode, duration)
		metrics.RequestCount.WithLabelValues(r.Method, r.URL.Path, fmt.Sprintf("%d", sw.statusCode)).Inc()
		metrics.LatencyHistogram.WithLabelValues(r.Method, r.URL.Path).Observe(duration.Seconds())
	})
}
