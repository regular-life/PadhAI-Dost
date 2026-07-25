package handlers

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestJSONResponse(t *testing.T) {
	w := httptest.NewRecorder()
	payload := map[string]string{"status": "ok", "message": "hello"}

	jsonResponse(w, payload)

	res := w.Result()
	if res.StatusCode != http.StatusOK {
		t.Errorf("expected status 200, got %d", res.StatusCode)
	}

	contentType := res.Header.Get("Content-Type")
	if contentType != "application/json" {
		t.Errorf("expected Content-Type application/json, got %s", contentType)
	}

	var data map[string]string
	if err := json.NewDecoder(res.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode JSON response: %v", err)
	}

	if data["status"] != "ok" || data["message"] != "hello" {
		t.Errorf("unexpected body payload: %v", data)
	}
}

func TestJSONError(t *testing.T) {
	w := httptest.NewRecorder()

	jsonError(w, "invalid request parameter", http.StatusBadRequest)

	res := w.Result()
	if res.StatusCode != http.StatusBadRequest {
		t.Errorf("expected status 400, got %d", res.StatusCode)
	}

	var data map[string]string
	if err := json.NewDecoder(res.Body).Decode(&data); err != nil {
		t.Fatalf("failed to decode JSON error: %v", err)
	}

	if data["error"] != "invalid request parameter" {
		t.Errorf("expected error message 'invalid request parameter', got %q", data["error"])
	}
}
