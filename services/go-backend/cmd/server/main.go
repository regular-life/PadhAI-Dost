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

func initInfrastructure(cfg *config.Config) (*cache.RedisCache, *cache.RedisSemanticCache, *audit.Logger, *memory.ConversationStore) {
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

	auditLogger := audit.NewLogger()
	convStore := memory.NewConversationStore(
		cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB,
		10,
		24*time.Hour,
	)

	return redisCache, semCache, auditLogger, convStore
}

func initCouncilPipeline(cfg *config.Config) ([]llm.LLMClient, llm.LLMClient, *agent.Router, *agent.IngestAgent) {
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

	chairmanClient, err := llm.NewClientFromProvider(
		cfg.ChairmanSlot.Provider, cfg.ChairmanSlot.Model, keys, urls, cfg.StageTimeout,
	)
	if err != nil {
		log.Fatalf("Chairman: %v", err)
	}

	routerClient, err := llm.NewClientFromProvider(
		cfg.RouterSlot.Provider, cfg.RouterSlot.Model, keys, urls, 15*time.Second,
	)
	if err != nil {
		log.Fatalf("Router agent: %v", err)
	}

	ingestClient, err := llm.NewClientFromProvider(
		cfg.IngestionSlot.Provider, cfg.IngestionSlot.Model, keys, urls, 30*time.Second,
	)
	if err != nil {
		log.Fatalf("Ingest agent: %v", err)
	}

	return councilClients, chairmanClient, agent.NewRouter(routerClient), agent.NewIngestAgent(ingestClient)
}

func initUserRepository(cfg *config.Config) auth.UserRepository {
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
		return memRepo
	}

	log.Println("Connected to PostgreSQL successfully")
	initCtx, initCancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer initCancel()
	if err := pgRepo.InitSchema(initCtx); err != nil {
		log.Fatalf("Failed to initialize PostgreSQL schema: %v", err)
	}
	if err := pgRepo.SeedDemoUser(initCtx, "demo", "demo123"); err != nil {
		log.Printf("[Warning] Failed to seed demo user: %v", err)
	}
	return pgRepo
}

func initTracer(cfg *config.Config) telemetry.ShutdownableTracerProvider {
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
		return nil
	}
	log.Println("OpenTelemetry distributed tracing initialized successfully")
	return tracer
}

func printStartupSummary(cfg *config.Config, councilClients []llm.LLMClient) {
	log.Printf("Server listening on :%s", cfg.ServerPort)
	log.Printf("RAG Service URL: %s", cfg.RAGServiceURL)
	log.Printf("Council size: %d members", len(councilClients))
	for i, slot := range cfg.CouncilSlots {
		if slot.Model != "" {
			log.Printf("  Member %d: %s/%s", i+1, slot.Provider, slot.Model)
		}
	}
	log.Printf("Chairman: %s/%s", cfg.ChairmanSlot.Provider, cfg.ChairmanSlot.Model)
	log.Printf("Router agent: using %s/%s", cfg.RouterSlot.Provider, cfg.RouterSlot.Model)
	if cfg.VLLMConfig.Enabled {
		log.Printf("vLLM local inference: auto-enabled (model: %s, quantization: %s)",
			cfg.VLLMConfig.ModelName, cfg.VLLMConfig.Quantization)
	}
}

func main() {
	log.Println("Starting CouncilAI Go Backend v2.0...")
	cfg := config.Load()

	jwtManager := auth.NewJWTManager(cfg.JWTSecret, cfg.JWTExpiration)
	redisCache, semCache, auditLogger, convStore := initInfrastructure(cfg)
	defer redisCache.Close()
	defer semCache.Close()
	defer convStore.Close()

	councilClients, chairmanClient, queryRouter, ingestAgent := initCouncilPipeline(cfg)
	userRepo := initUserRepository(cfg)
	defer userRepo.Close()

	tracer := initTracer(cfg)
	if tracer != nil {
		defer func() {
			shutdownCtx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
			defer cancel()
			if err := tracer.Shutdown(shutdownCtx); err != nil {
				log.Printf("[Telemetry] Shutdown error: %v", err)
			}
		}()
	}

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

	printStartupSummary(cfg, councilClients)
	if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatalf("Server failed: %v", err)
	}
}
