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
	"github.com/regular-life/CouncilAI/go-backend/internal/telemetry"
)

func main() {
	log.Println("Starting CouncilAI Go Backend v2.0...")

	cfg := config.Load()

	// ── Core infrastructure ─────────────────────────────────────────
	jwtManager := auth.NewJWTManager(cfg.JWTSecret, cfg.JWTExpiration)

	// Shared Redis Circuit Breaker for unified failure detection across L1 and L2
	cbConfig := cache.Config{
		FailureThreshold: cfg.CircuitBreakerFailureThreshold,
		SuccessThreshold: cfg.CircuitBreakerSuccessThreshold,
		Timeout:          cfg.CircuitBreakerTimeout,
		HalfOpenMaxCalls: 1,
	}
	redisCB := cache.NewCircuitBreaker("shared-redis-breaker", cbConfig)

	redisCache := cache.NewRedisCache(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB)
	redisCache.SetCircuitBreaker(redisCB)

	semCache := cache.NewRedisSemanticCache(cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB)
	semCache.SetCircuitBreaker(redisCB)
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

	// ── User Repository (PostgreSQL with Memory Fallback) ─────────────
	var userRepo auth.UserRepository
	pgConfig := auth.PostgresConfig{
		URL:             cfg.DatabaseURL,
		Host:            cfg.DBHost,
		Port:            cfg.DBPort,
		User:            cfg.DBUser,
		Password:        cfg.DBPassword,
		Database:        cfg.DBName,
		SSLMode:         cfg.DBSSLMode,
		MaxConns:        int32(cfg.DBMaxOpenConns),
		MinConns:        int32(cfg.DBMaxIdleConns),
		MaxConnLifetime: cfg.DBConnMaxLifetime,
		MaxConnIdleTime: cfg.DBConnMaxIdleTime,
	}

	dbCtx, dbCancel := context.WithTimeout(context.Background(), 5*time.Second)
	pgRepo, err := auth.NewPostgresUserRepository(dbCtx, pgConfig)
	dbCancel()

	if err != nil {
		log.Printf("[Warning] PostgreSQL initialization failed: %v. Falling back to MemoryUserRepository", err)
		memRepo := auth.NewMemoryUserRepository()
		_ = memRepo.SeedDemoUser(context.Background(), "demo", "demo123")
		userRepo = memRepo
	} else {
		log.Println("Connected to PostgreSQL successfully")
		initCtx, initCancel := context.WithTimeout(context.Background(), 5*time.Second)
		if err := pgRepo.InitSchema(initCtx); err != nil {
			log.Fatalf("Failed to initialize PostgreSQL schema: %v", err)
		}
		if err := pgRepo.SeedDemoUser(initCtx, "demo", "demo123"); err != nil {
			log.Printf("[Warning] Failed to seed demo user: %v", err)
		}
		initCancel()
		userRepo = pgRepo
	}
	defer userRepo.Close()

	// ── OpenTelemetry Tracing ───────────────────────────────────────
	env := os.Getenv("ENVIRONMENT")
	if env == "" {
		if cfg.Debug {
			env = "development"
		} else {
			env = "production"
		}
	}
	tracer, err := telemetry.NewTracerProvider(telemetry.Config{
		ServiceName:    "councilai-go-backend",
		ServiceVersion: "2.0.0",
		Environment:    env,
		ExporterType:   telemetry.ExporterNoop,
	})
	if err != nil {
		log.Printf("[Warning] Failed to initialize OpenTelemetry tracer: %v", err)
	} else {
		log.Println("OpenTelemetry distributed tracing initialized successfully")
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := tracer.Shutdown(shutdownCtx); err != nil {
				log.Printf("[Telemetry] Shutdown error: %v", err)
			}
		}()
	}

	// ── Wire everything together ────────────────────────────────────
	councilOrchestrator := council.NewOrchestrator(councilClients, chairmanClient, cfg.StageTimeout)
	if tracer != nil {
		councilOrchestrator.SetTracer(tracer)
	}

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
	if tracer != nil {
		h.SetTracer(tracer)
	}
	authHandler := handlers.NewAuthHandler(jwtManager, userRepo)
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
		userRepo.Close()
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
