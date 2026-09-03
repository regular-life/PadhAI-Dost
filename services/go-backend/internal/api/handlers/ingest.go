package handlers

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
)

// HandleIngest passes uploaded documents to the Python RAG service.
func (h *Handlers) HandleIngest(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	if err := r.ParseMultipartForm(50 << 20); err != nil {
		jsonError(w, "failed to parse form data", http.StatusBadRequest)
		return
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		jsonError(w, "file is required", http.StatusBadRequest)
		return
	}
	defer file.Close()

	docID := r.FormValue("doc_id")

	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", header.Filename)
	if err != nil {
		jsonError(w, "failed to create form file", http.StatusInternalServerError)
		return
	}
	if _, err := io.Copy(part, file); err != nil {
		jsonError(w, "failed to copy file", http.StatusInternalServerError)
		return
	}
	if docID != "" {
		_ = writer.WriteField("doc_id", docID)
	}
	writer.Close()

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", h.RAGServiceURL+"/ingest", &buf)
	if err != nil {
		log.Printf("[Ingest] Failed to create request: %v", err)
		jsonError(w, "failed to create request", http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", writer.FormDataContentType())
	if h.Tracer != nil {
		h.Tracer.InjectHTTPHeaders(r.Context(), httpReq)
	}

	resp, err := h.HTTPClient.Do(httpReq)
	if err != nil {
		log.Printf("[Ingest] Failed to contact RAG service: %v", err)
		jsonError(w, "RAG service unavailable", http.StatusServiceUnavailable)
		return
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Printf("[Ingest] Failed to read response body: %v", err)
		jsonError(w, "failed to read ingestion response", http.StatusInternalServerError)
		return
	}

	var parsedResp struct {
		DocID       string `json:"doc_id"`
		PreviewText string `json:"preview_text"`
	}
	if err := json.Unmarshal(body, &parsedResp); err == nil && parsedResp.DocID != "" {
		docID = parsedResp.DocID
	}

	if parsedResp.PreviewText != "" && docID != "" {
		summary, err := h.IngestAgent.SummarizeDocument(r.Context(), parsedResp.PreviewText)
		if err == nil && summary != "" {
			if err := h.Cache.Set(r.Context(), "doc_summary:"+docID, summary); err != nil {
				log.Printf("[Ingest] Failed to cache summary for doc %s: %v", docID, err)
			} else {
				log.Printf("[Ingest] Generated and cached summary for doc %s", docID)
			}
		} else {
			log.Printf("[Ingest] Summarization failed for doc %s: %v", docID, err)
		}
	}

	h.Audit.LogIngest(userID, docID, time.Since(start), "success")

	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(resp.StatusCode)
	_, _ = w.Write(body)
}

// HandleHealth checks Go control plane health and dependency states.
func (h *Handlers) HandleHealth(w http.ResponseWriter, r *http.Request) {
	status := map[string]interface{}{
		"status":  "healthy",
		"service": "go-backend",
		"version": "2.0.0",
	}

	if err := h.Cache.Ping(r.Context()); err != nil {
		status["redis"] = "unhealthy"
	} else {
		status["redis"] = "healthy"
	}

	httpReq, err := http.NewRequestWithContext(r.Context(), "GET", h.RAGServiceURL+"/health", nil)
	var resp *http.Response
	if err == nil {
		if h.Tracer != nil {
			h.Tracer.InjectHTTPHeaders(r.Context(), httpReq)
		}
		resp, err = h.HTTPClient.Do(httpReq)
	}
	if err != nil || resp.StatusCode != 200 {
		status["rag_service"] = "unhealthy"
	} else {
		status["rag_service"] = "healthy"
		_ = resp.Body.Close()
	}

	jsonResponse(w, status)
}
