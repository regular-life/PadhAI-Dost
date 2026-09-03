package handlers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"mime/multipart"
	"net/http"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
)

// HandleIngest passes uploaded documents to the Python RAG service.
func prepareIngestMultipart(r *http.Request) (multipart.File, *bytes.Buffer, string, string, int, error) {
	if err := r.ParseMultipartForm(50 << 20); err != nil {
		return nil, nil, "", "", http.StatusBadRequest, fmt.Errorf("failed to parse form data: %w", err)
	}

	file, header, err := r.FormFile("file")
	if err != nil {
		return nil, nil, "", "", http.StatusBadRequest, fmt.Errorf("file is required: %w", err)
	}

	docID := r.FormValue("doc_id")
	var buf bytes.Buffer
	writer := multipart.NewWriter(&buf)
	part, err := writer.CreateFormFile("file", header.Filename)
	if err != nil {
		file.Close()
		return nil, nil, "", "", http.StatusInternalServerError, fmt.Errorf("failed to create form file: %w", err)
	}
	if _, err := io.Copy(part, file); err != nil {
		file.Close()
		return nil, nil, "", "", http.StatusInternalServerError, fmt.Errorf("failed to copy file: %w", err)
	}
	if docID != "" {
		_ = writer.WriteField("doc_id", docID)
	}
	writer.Close()

	return file, &buf, writer.FormDataContentType(), docID, http.StatusOK, nil
}

func (h *Handlers) summarizeAndCacheIngestDoc(ctx context.Context, previewText, docID string) {
	if previewText == "" || docID == "" || h.IngestAgent == nil {
		return
	}
	summary, err := h.IngestAgent.SummarizeDocument(ctx, previewText)
	if err != nil || summary == "" {
		log.Printf("[Ingest] Summarization failed for doc %s: %v", docID, err)
		return
	}
	if h.Cache != nil {
		if err := h.Cache.Set(ctx, "doc_summary:"+docID, summary); err != nil {
			log.Printf("[Ingest] Failed to cache summary for doc %s: %v", docID, err)
		} else {
			log.Printf("[Ingest] Generated and cached summary for doc %s", docID)
		}
	}
}

func (h *Handlers) HandleIngest(w http.ResponseWriter, r *http.Request) {
	start := time.Now()
	userID := middleware.GetUserID(r.Context())

	file, buf, contentType, docID, status, err := prepareIngestMultipart(r)
	if err != nil {
		jsonError(w, err.Error(), status)
		return
	}
	defer file.Close()

	httpReq, err := http.NewRequestWithContext(r.Context(), "POST", h.RAGServiceURL+"/ingest", buf)
	if err != nil {
		log.Printf("[Ingest] Failed to create request: %v", err)
		jsonError(w, "failed to create request", http.StatusInternalServerError)
		return
	}
	httpReq.Header.Set("Content-Type", contentType)
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

	h.summarizeAndCacheIngestDoc(r.Context(), parsedResp.PreviewText, docID)
	if h.Audit != nil {
		h.Audit.LogIngest(userID, docID, time.Since(start), "success")
	}

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
