// Package api configures the HTTP routing topology and middleware chains for the CouncilAI gateway.
package api

import (
	"github.com/go-chi/chi/v5"
	chimiddleware "github.com/go-chi/chi/v5/middleware"
	"github.com/go-chi/cors"
	"github.com/prometheus/client_golang/prometheus/promhttp"

	"github.com/regular-life/CouncilAI/go-backend/internal/api/handlers"
	"github.com/regular-life/CouncilAI/go-backend/internal/api/middleware"
	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
	"github.com/regular-life/CouncilAI/go-backend/internal/config"
)

// NewRouter constructs and mounts the chi HTTP router with middlewares and versioned endpoints.
func NewRouter(cfg *config.Config, h *handlers.Handlers, authHandler *handlers.AuthHandler, jwtManager *auth.JWTManager) *chi.Mux {
	r := chi.NewRouter()

	r.Use(chimiddleware.RequestID)
	r.Use(chimiddleware.RealIP)
	r.Use(middleware.LoggingMiddleware)
	r.Use(chimiddleware.Recoverer)
	r.Use(chimiddleware.Timeout(cfg.RequestTimeout))
	r.Use(cors.Handler(cors.Options{
		AllowedOrigins:   []string{"*"},
		AllowedMethods:   []string{"GET", "POST", "PUT", "DELETE", "OPTIONS"},
		AllowedHeaders:   []string{"Accept", "Authorization", "Content-Type"},
		ExposedHeaders:   []string{"Link"},
		AllowCredentials: true,
		MaxAge:           300,
	}))

	// Public
	r.Get("/health", h.HandleHealth)
	r.Handle("/metrics", promhttp.Handler())
	r.Post("/api/v1/login", authHandler.HandleLogin)
	r.Post("/api/v1/register", authHandler.HandleRegister)

	// Protected (JWT + rate limit)
	rateLimiter := middleware.NewRateLimiter(
		cfg.RedisAddr, cfg.RedisPassword, cfg.RedisDB,
		cfg.RateLimitRPS, cfg.RateLimitBurst,
	)
	r.Group(func(r chi.Router) {
		r.Use(middleware.AuthMiddleware(jwtManager))
		r.Use(rateLimiter.Middleware())

		r.Post("/api/v1/query", h.HandleQuery)
		r.Post("/api/v1/ingest", h.HandleIngest)
		r.Post("/api/v1/explain", h.HandleExplain)
		r.Post("/api/v1/generate-questions", h.HandleGenerateQuestions)
		r.Post("/api/v1/conversation/clear", h.HandleClearConversation)
	})

	return r
}
