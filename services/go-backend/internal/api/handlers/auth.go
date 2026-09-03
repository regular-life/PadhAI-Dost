package handlers

import (
	"encoding/json"
	"errors"
	"net/http"
	"strings"

	"golang.org/x/crypto/bcrypt"

	"github.com/regular-life/CouncilAI/go-backend/internal/auth"
)

// LoginRequest defines credentials for authentication endpoints.
type LoginRequest struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

// AuthHandler coordinates JWT-based user registrations and logins.
type AuthHandler struct {
	jwtManager *auth.JWTManager
	repo       auth.UserRepository
}

// NewAuthHandler initializes AuthHandler with injected JWT manager and user repository.
func NewAuthHandler(jwtManager *auth.JWTManager, repo auth.UserRepository) *AuthHandler {
	return &AuthHandler{
		jwtManager: jwtManager,
		repo:       repo,
	}
}

// HandleLogin authenticates users and issues JWT authorization tokens.
func (h *AuthHandler) HandleLogin(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}

	req.Username = strings.TrimSpace(req.Username)
	if req.Username == "" || req.Password == "" {
		jsonError(w, "username and password are required", http.StatusBadRequest)
		return
	}

	user, err := h.repo.GetUserByUsername(r.Context(), req.Username)
	if err != nil {
		if errors.Is(err, auth.ErrUserNotFound) {
			jsonError(w, "invalid credentials", http.StatusUnauthorized)
			return
		}
		jsonError(w, "internal authentication error", http.StatusInternalServerError)
		return
	}

	if err := bcrypt.CompareHashAndPassword([]byte(user.PasswordHash), []byte(req.Password)); err != nil {
		jsonError(w, "invalid credentials", http.StatusUnauthorized)
		return
	}

	token, err := h.jwtManager.GenerateToken(user.Username)
	if err != nil {
		jsonError(w, "failed to generate token", http.StatusInternalServerError)
		return
	}

	jsonResponse(w, map[string]string{
		"token":   token,
		"user_id": user.Username,
	})
}

// HandleRegister creates a new user and issues a JWT token.
func (h *AuthHandler) HandleRegister(w http.ResponseWriter, r *http.Request) {
	var req LoginRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		jsonError(w, "invalid request body", http.StatusBadRequest)
		return
	}

	req.Username = strings.TrimSpace(req.Username)
	if req.Username == "" || req.Password == "" {
		jsonError(w, "username and password are required", http.StatusBadRequest)
		return
	}

	hash, err := bcrypt.GenerateFromPassword([]byte(req.Password), bcrypt.DefaultCost)
	if err != nil {
		jsonError(w, "failed to hash password", http.StatusInternalServerError)
		return
	}

	user, err := h.repo.CreateUser(r.Context(), req.Username, string(hash))
	if err != nil {
		if errors.Is(err, auth.ErrUserAlreadyExists) {
			jsonError(w, "user already exists", http.StatusConflict)
			return
		}
		jsonError(w, "failed to create user", http.StatusInternalServerError)
		return
	}

	token, err := h.jwtManager.GenerateToken(user.Username)
	if err != nil {
		jsonError(w, "failed to generate token", http.StatusInternalServerError)
		return
	}

	w.Header().Set("Content-Type", "application/json")
	w.Header().Set("X-Content-Type-Options", "nosniff")
	w.Header().Set("X-Frame-Options", "DENY")
	w.Header().Set("X-XSS-Protection", "1; mode=block")
	w.WriteHeader(http.StatusCreated)
	_ = json.NewEncoder(w).Encode(map[string]string{
		"token":   token,
		"user_id": user.Username,
		"message": "user created successfully",
	})
}
