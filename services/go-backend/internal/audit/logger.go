// Package audit provides structured JSON security and compliance audit logging for user operations.
package audit

import (
	"encoding/json"
	"log"
	"time"
)

// Entry represents an immutable audit log record for a user or system action.
type Entry struct {
	UserID    string        `json:"user_id"`
	DocID     string        `json:"doc_id"`
	QueryHash string        `json:"query_hash"`
	Action    string        `json:"action"`
	Timestamp time.Time     `json:"timestamp"`
	Latency   time.Duration `json:"latency_ms"`
	Status    string        `json:"status"`
	Details   string        `json:"details,omitempty"`
}

// Logger emits structured JSON audit log records to standard logging output.
type Logger struct{}

// NewLogger constructs a new audit Logger instance.
func NewLogger() *Logger {
	return &Logger{}
}

// Log serializes and prints an audit Entry with an accurate current timestamp.
func (l *Logger) Log(entry Entry) {
	entry.Timestamp = time.Now()
	data, err := json.Marshal(entry)
	if err != nil {
		log.Printf("[Audit] Failed to marshal entry: %v", err)
		return
	}
	log.Printf("[Audit] %s", string(data))
}

// LogQuery records an audit log entry for a document or general query deliberation.
func (l *Logger) LogQuery(userID, docID, queryHash string, latency time.Duration, status string) {
	l.Log(Entry{
		UserID:    userID,
		DocID:     docID,
		QueryHash: queryHash,
		Action:    "query",
		Latency:   latency,
		Status:    status,
	})
}

// LogIngest records an audit log entry for a document ingestion and chunking operation.
func (l *Logger) LogIngest(userID, docID string, latency time.Duration, status string) {
	l.Log(Entry{
		UserID:  userID,
		DocID:   docID,
		Action:  "ingest",
		Latency: latency,
		Status:  status,
	})
}
