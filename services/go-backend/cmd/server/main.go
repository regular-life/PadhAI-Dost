package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/regular-life/CouncilAI/go-backend/internal/agent"
	"github.com/regular-life/CouncilAI/go-backend/internal/api"
	"github.com/regular-life/CouncilAI/go-backend/internal/api/handlers"
	"github.com/regular-life/CouncilAI/go-backend/internal/audit"
	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
	"github.com/regular-life/CouncilAI/go-backend/internal/cache"
	"github.com/regular-life/CouncilAI/go-backend/internal/config"
	"github.com/regular-life/CouncilAI/go-backend/internal/council"
	"github.com/regular-life/CouncilAI/go-backend/internal/llm"
	"github.com/regular-life/CouncilAI/go-backend/internal/memory"
)

func main() {
	log.Println("Starting CouncilAI Go Backend v2.0...")

	cfg := config.Load()

	// ── Core infrastructure ─────────────────────────────────────────
	jwtManager := auth.NewJWTManager(cfg.JWTSecret, cfg.JWTExpiration)
	redisCache := cache.NewRedisCache(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB)
	semCache := cache.NewRedisSemanticCache(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB)
	if err := semCache.EnsureIndex(context.Background()); err != nil {
		log.Printf("[Warning] Failed to ensure RediSearch index: %v", err)
	}
	defer semCache.Close()
	auditLogger := audit.NewLogger()

	// ── Conversation memory ─────────────────────────────────────────
	convStore := memory.NewConversationStore(
		cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB,
		10,             // max turns per session
		24*time.Hour,   // TTL
	)
	defer convStore.Close()

	// ── LLM provider configuration ──────────────────────────────────
	keys := llm.ProviderKeys{
		Gemini:     cfg.GeminiAPIKey,
		OpenRouter: cfg.OpenRouterAPIKey,
		NVIDIANim:  cfg.NVIDIANimAPIKey,
	}
	urls := llm.ProviderURLs{
		OpenRouter: cfg.OpenRouterURL,
		NVIDIANim:  cfg.NVIDIANimURL,
		VLLM:       cfg.VLLMURL,
	}

	// ── Dynamic council creation ────────────────────────────────────
	var councilClients []llm.LLMClient
	for i, slot := range cfg.CouncilSlots {
		if slot.Model == "" {
			log.Printf("Council member %d: skipped (no model configured)", i+1)
			continue
		}
		client, err := llm.NewClientFromProvider(slot.Provider, slot.Model, keys, urls, cfg.LLMTimeout)
		if err != nil {
			log.Fatalf("Council member %d: %v", i+1, err)
		}
		councilClients = append(councilClients, client)
	}
	if len(councilClients) == 0 {
		log.Fatal("No council members configured. Set at least COUNCIL_1_PROVIDER and COUNCIL_1_MODEL.")
	}

	// ── Chairman ────────────────────────────────────────────────────
	chairmanClient, err := llm.NewClientFromProvider(
		cfg.ChairmanSlot.Provider, cfg.ChairmanSlot.Model, keys, urls, cfg.StageTimeout,
	)
	if err != nil {
		log.Fatalf("Chairman: %v", err)
	}

	// ── Router agent ────────────────────────────────────────────────
	routerClient, err := llm.NewClientFromProvider(
		cfg.RouterSlot.Provider, cfg.RouterSlot.Model, keys, urls, 15*time.Second,
	)
	if err != nil {
		log.Fatalf("Router agent: %v", err)
	}
	queryRouter := agent.NewRouter(routerClient)

	// ── Ingest agent ────────────────────────────────────────────────
	ingestClient, err := llm.NewClientFromProvider(
		cfg.IngestionSlot.Provider, cfg.IngestionSlot.Model, keys, urls, 30*time.Second,
	)
	if err != nil {
		log.Fatalf("Ingest agent: %v", err)
	}
	ingestAgent := agent.NewIngestAgent(ingestClient)

	// ── Wire everything together ────────────────────────────────────
	councilOrchestrator := council.NewOrchestrator(councilClients, chairmanClient, cfg.StageTimeout)

	h := handlers.NewHandlers(
		cfg.RAGServiceURL,
		councilOrchestrator,
		redisCache,
		semCache,
		auditLogger,
		queryRouter,
		ingestAgent,
		convStore,
		float32(cfg.SemanticCacheThreshold),
	)
	authHandler := handlers.NewAuthHandler(jwtManager)
	router := api.NewRouter(cfg, h, authHandler, jwtManager)

	// ── HTTP server ─────────────────────────────────────────────────
	server := &http.Server{
		Addr:         fmt.Sprintf(":%s", cfg.ServerPort),
		Handler:      router,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: cfg.RequestTimeout + 5*time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		sigChan := make(chan os.Signal, 1)
		signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
		<-sigChan

		log.Println("Shutting down gracefully...")
		ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
		defer cancel()

		if err := server.Shutdown(ctx); err != nil {
			log.Printf("Server shutdown error: %v", err)
		}
		redisCache.Close()
		convStore.Close()
		log.Println("Server stopped")
	}()

	// ── Startup log ─────────────────────────────────────────────────
	log.Printf("Server listening on :%s", cfg.ServerPort)
	log.Printf("RAG Service URL: %s", cfg.RAGServiceURL)
	log.Printf("Council size: %d members", len(councilClients))
	for i, slot := range cfg.CouncilSlots {
		if slot.Model != "" {
			log.Printf("  Member %d: %s/%s", i+1, slot.Provider, slot.Model)
		}
	}
	log.Printf("Chairman: %s/%s", cfg.ChairmanSlot.Provider, cfg.ChairmanSlot.Model)
	log.Printf("Router agent: using %s/%s", cfg.ChairmanSlot.Provider, cfg.ChairmanSlot.Model)
	if cfg.VLLMConfig.Enabled {
		log.Printf("vLLM local inference: auto-enabled (model: %s, quantization: %s)",
			cfg.VLLMConfig.ModelName, cfg.VLLMConfig.Quantization)
		log.Println("  [IMPORTANT] If using Docker Compose, make sure to start the service using the 'local-models' profile:")
		log.Println("              docker compose --profile local-models up --build")
	}

	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}
