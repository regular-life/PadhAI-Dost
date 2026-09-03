// Package config manages system configurations loaded from YAML and environment variables.
package config

import (
	"fmt"
	"os"
	"strconv"
	"time"

	"gopkg.in/yaml.v3"
)

// ModelSlot represents a model provider and name pair.
type ModelSlot struct {
	Provider string
	Model    string
}

// LocalModelConfig holds settings for local vLLM model execution.
type LocalModelConfig struct {
	Enabled              bool
	ModelName            string
	Quantization         string
	DType                string
	GPUMemoryUtilization float64
	MaxModelLen          int
	TensorParallelSize   int
	SwapSpaceGB          int
	MaxNumSeqs           int
	KVCacheDType         string
	CPUOffloadGB         int
}

// Config represents the unified system configuration.
type Config struct {
	ServerPort                     string
	Debug                          bool
	RAGServiceURL                  string
	RedisAddr                      string
	RedisPassword                  string
	RedisDB                        int
	CircuitBreakerFailureThreshold int
	CircuitBreakerSuccessThreshold int
	CircuitBreakerTimeout          time.Duration
	JWTSecret                      string
	JWTExpiration                  time.Duration
	RateLimitRPS                   int
	RateLimitBurst                 int

	DatabaseURL       string
	DBHost            string
	DBPort            string
	DBUser            string
	DBPassword        string
	DBName            string
	DBSSLMode         string
	DBMaxOpenConns    int
	DBMaxIdleConns    int
	DBConnMaxLifetime time.Duration
	DBConnMaxIdleTime time.Duration

	OpenRouterAPIKey string
	OpenRouterURL    string
	GeminiAPIKey     string
	NVIDIANimAPIKey  string
	NVIDIANimURL     string

	VLLMURL    string
	VLLMConfig LocalModelConfig

	CouncilSize   int
	CouncilSlots  []ModelSlot
	ChairmanSlot  ModelSlot
	RouterSlot    ModelSlot
	IngestionSlot ModelSlot

	StageTimeout           time.Duration
	LLMTimeout             time.Duration
	RequestTimeout         time.Duration
	SemanticCacheThreshold float64
}

// yamlSchema defines the structure of config.yaml.
type yamlSchema struct {
	Server struct {
		Port          string `yaml:"port"`
		Debug         bool   `yaml:"debug"`
		JWTSecret     string `yaml:"jwt_secret"`
		JWTExpiration string `yaml:"jwt_expiration"`
		RateLimit     struct {
			RPS   int `yaml:"rps"`
			Burst int `yaml:"burst"`
		} `yaml:"rate_limit"`
		SemanticCacheThreshold float64 `yaml:"semantic_cache_threshold"`
	} `yaml:"server"`

	RAG struct {
		ServiceURL string `yaml:"service_url"`
	} `yaml:"rag"`

	Redis struct {
		Addr             string `yaml:"addr"`
		Password         string `yaml:"password"`
		DB               int    `yaml:"db"`
		FailureThreshold int    `yaml:"circuit_breaker_failure_threshold"`
		SuccessThreshold int    `yaml:"circuit_breaker_success_threshold"`
		Timeout          string `yaml:"circuit_breaker_timeout"`
	} `yaml:"redis"`

	Database struct {
		URL             string `yaml:"url"`
		Host            string `yaml:"host"`
		Port            string `yaml:"port"`
		User            string `yaml:"user"`
		Password        string `yaml:"password"`
		DBName          string `yaml:"dbname"`
		SSLMode         string `yaml:"sslmode"`
		MaxOpenConns    int    `yaml:"max_open_conns"`
		MaxIdleConns    int    `yaml:"max_idle_conns"`
		ConnMaxLifetime string `yaml:"conn_max_lifetime"`
		ConnMaxIdleTime string `yaml:"conn_max_idle_time"`
	} `yaml:"database"`

	Providers struct {
		OpenRouterURL string `yaml:"openrouter_url"`
		NVIDIANimURL  string `yaml:"nvidia_nim_url"`
		VLLMURL       string `yaml:"vllm_url"`
	} `yaml:"providers"`

	Keys struct {
		Gemini     string `yaml:"gemini"`
		OpenRouter string `yaml:"openrouter"`
		NVIDIANim  string `yaml:"nvidia_nim"`
	} `yaml:"keys"`

	Council struct {
		Size  int `yaml:"size"`
		Slots []struct {
			Provider string `yaml:"provider"`
			Model    string `yaml:"model"`
		} `yaml:"slots"`
		Chairman struct {
			Provider string `yaml:"provider"`
			Model    string `yaml:"model"`
		} `yaml:"chairman"`
		Router struct {
			Provider string `yaml:"provider"`
			Model    string `yaml:"model"`
		} `yaml:"router"`
		Ingestion struct {
			Provider string `yaml:"provider"`
			Model    string `yaml:"model"`
		} `yaml:"ingestion"`
		Timeouts struct {
			Stage   string `yaml:"stage"`
			LLM     string `yaml:"llm"`
			Request string `yaml:"request"`
		} `yaml:"timeouts"`
	} `yaml:"council"`

	VLLM struct {
		Enabled              bool    `yaml:"enabled"`
		ModelName            string  `yaml:"model_name"`
		DType                string  `yaml:"dtype"`
		MaxModelLen          int     `yaml:"max_model_len"`
		GPUMemoryUtilization float64 `yaml:"gpu_memory_utilization"`
		Quantization         string  `yaml:"quantization"`
		TensorParallelSize   int     `yaml:"tensor_parallel_size"`
		SwapSpaceGB          int     `yaml:"swap_space_gb"`
		MaxNumSeqs           int     `yaml:"max_num_seqs"`
		KVCacheDType         string  `yaml:"kv_cache_dtype"`
		CPUOffloadGB         int     `yaml:"cpu_offload_gb"`
	} `yaml:"vllm"`
}

// Load reads config.yaml and overlays environment variables.
func Load() *Config {
	var y yamlSchema

	// Read config.yaml if available.
	paths := []string{"/app/config.yaml", "config.yaml", "../config.yaml", "../../config.yaml"}
	for _, p := range paths {
		f, err := os.Open(p)
		if err == nil {
			dec := yaml.NewDecoder(f)
			if err := dec.Decode(&y); err == nil {
				_ = f.Close()
				break
			}
			_ = f.Close()
		}
	}

	// 1. Core Server Configs.
	port := getEnv("SERVER_PORT", y.Server.Port)
	if port == "" {
		port = "8080"
	}
	debug := getEnvBool("DEBUG", y.Server.Debug)
	jwtSecret := getEnv("JWT_SECRET", y.Server.JWTSecret)
	if jwtSecret == "" {
		jwtSecret = "council-ai-secret-change-me"
	}
	jwtExp := 24 * time.Hour
	if expStr := getEnv("JWT_EXPIRATION", y.Server.JWTExpiration); expStr != "" {
		if d, err := time.ParseDuration(expStr); err == nil {
			jwtExp = d
		}
	}
	rps := getEnvInt("RATE_LIMIT_RPS", y.Server.RateLimit.RPS)
	if rps <= 0 {
		rps = 10
	}
	burst := getEnvInt("RATE_LIMIT_BURST", y.Server.RateLimit.Burst)
	if burst <= 0 {
		burst = 20
	}
	cacheThresh := getEnvFloat64("SEMANTIC_CACHE_THRESHOLD", y.Server.SemanticCacheThreshold)
	if cacheThresh <= 0 {
		cacheThresh = 0.85
	}

	// 2. Peripheral Services.
	ragURL := getEnv("RAG_SERVICE_URL", y.RAG.ServiceURL)
	if ragURL == "" {
		ragURL = "http://python-rag:8000"
	}
	redisAddr := getEnv("REDIS_ADDR", y.Redis.Addr)
	if redisAddr == "" {
		redisAddr = "redis:6379"
	}
	redisPwd := getEnv("REDIS_PASSWORD", y.Redis.Password)
	redisDB := getEnvInt("REDIS_DB", y.Redis.DB)

	cbFailureThresh := getEnvInt("CIRCUIT_BREAKER_FAILURE_THRESHOLD", y.Redis.FailureThreshold)
	if cbFailureThresh <= 0 {
		cbFailureThresh = 3
	}
	cbSuccessThresh := getEnvInt("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", y.Redis.SuccessThreshold)
	if cbSuccessThresh <= 0 {
		cbSuccessThresh = 2
	}
	cbTimeout := parseDuration(getEnv("CIRCUIT_BREAKER_TIMEOUT", y.Redis.Timeout), 10*time.Second)

	// Database Settings
	dbURL := getEnv("DATABASE_URL", y.Database.URL)
	dbHost := getEnv("DB_HOST", y.Database.Host)
	dbPort := getEnv("DB_PORT", y.Database.Port)
	if dbPort == "" {
		dbPort = "5432"
	}
	dbUser := getEnv("DB_USER", y.Database.User)
	if dbUser == "" {
		dbUser = "council_user"
	}
	dbPass := getEnv("DB_PASSWORD", y.Database.Password)
	if dbPass == "" {
		dbPass = "council_pass"
	}
	dbName := getEnv("DB_NAME", y.Database.DBName)
	if dbName == "" {
		dbName = "council_db"
	}
	dbSSLMode := getEnv("DB_SSLMODE", y.Database.SSLMode)
	if dbSSLMode == "" {
		dbSSLMode = "disable"
	}

	if dbURL == "" && (dbHost != "" || os.Getenv("DB_HOST") != "") {
		if dbHost == "" {
			dbHost = "localhost"
		}
		dbURL = fmt.Sprintf("postgres://%s:%s@%s:%s/%s?sslmode=%s", dbUser, dbPass, dbHost, dbPort, dbName, dbSSLMode)
	}

	maxOpenConns := getEnvInt("DB_MAX_OPEN_CONNS", y.Database.MaxOpenConns)
	if maxOpenConns <= 0 {
		maxOpenConns = 25
	}
	maxIdleConns := getEnvInt("DB_MAX_IDLE_CONNS", y.Database.MaxIdleConns)
	if maxIdleConns <= 0 {
		maxIdleConns = 5
	}
	connMaxLifetime := parseDuration(getEnv("DB_CONN_MAX_LIFETIME", y.Database.ConnMaxLifetime), 1*time.Hour)
	connMaxIdleTime := parseDuration(getEnv("DB_CONN_MAX_IDLE_TIME", y.Database.ConnMaxIdleTime), 30*time.Minute)

	// 3. Provider URLs & API Keys.
	orURL := getEnv("OPENROUTER_URL", y.Providers.OpenRouterURL)
	if orURL == "" {
		orURL = "https://openrouter.ai/api/v1/chat/completions"
	}
	nimURL := getEnv("NVIDIA_NIM_URL", y.Providers.NVIDIANimURL)
	if nimURL == "" {
		nimURL = "https://integrate.api.nvidia.com/v1/chat/completions"
	}
	vllmURL := getEnv("VLLM_URL", y.Providers.VLLMURL)
	if vllmURL == "" {
		vllmURL = "http://vllm-inference:8001/v1/chat/completions"
	}

	geminiKey := getEnv("GEMINI_API_KEY", y.Keys.Gemini)
	orKey := getEnv("OPENROUTER_API_KEY", y.Keys.OpenRouter)
	nimKey := getEnv("NVIDIA_NIM_API_KEY", y.Keys.NVIDIANim)

	// 4. Council Members and Core Roles.
	cSize := getEnvInt("COUNCIL_SIZE", y.Council.Size)
	if cSize <= 0 {
		cSize = 3
	}

	slots := make([]ModelSlot, cSize)
	for i := 0; i < cSize; i++ {
		n := i + 1
		var defProvider string
		var defModel string

		if i < len(y.Council.Slots) {
			defProvider = y.Council.Slots[i].Provider
			defModel = y.Council.Slots[i].Model
		} else {
			defProvider = "openrouter"
		}

		slots[i] = ModelSlot{
			Provider: getEnv(fmt.Sprintf("COUNCIL_%d_PROVIDER", n), defProvider),
			Model:    getEnv(fmt.Sprintf("COUNCIL_%d_MODEL", n), defModel),
		}
	}

	chairmanProvider := getEnv("CHAIRMAN_PROVIDER", y.Council.Chairman.Provider)
	if chairmanProvider == "" {
		chairmanProvider = "gemini"
	}
	chairmanModel := getEnv("CHAIRMAN_MODEL", y.Council.Chairman.Model)
	if chairmanModel == "" {
		chairmanModel = "gemini-3-flash-preview"
	}

	routerProvider := getEnv("ROUTER_PROVIDER", y.Council.Router.Provider)
	if routerProvider == "" {
		routerProvider = "gemini"
	}
	routerModel := getEnv("ROUTER_MODEL", y.Council.Router.Model)
	if routerModel == "" {
		routerModel = "gemini-3-flash-preview"
	}

	ingestionProvider := getEnv("INGESTION_PROVIDER", y.Council.Ingestion.Provider)
	if ingestionProvider == "" {
		ingestionProvider = "gemini"
	}
	ingestionModel := getEnv("INGESTION_MODEL", y.Council.Ingestion.Model)
	if ingestionModel == "" {
		ingestionModel = "gemini-3-flash-preview"
	}

	// 5. Execution Timeouts.
	stageTimeout := parseDuration(getEnv("STAGE_TIMEOUT", y.Council.Timeouts.Stage), 30*time.Second)
	llmTimeout := parseDuration(getEnv("LLM_TIMEOUT", y.Council.Timeouts.LLM), 120*time.Second)
	reqTimeout := parseDuration(getEnv("REQUEST_TIMEOUT", y.Council.Timeouts.Request), 120*time.Second)

	// 6. Local Inference (vLLM Engine Configs).
	vllmEnabled := getEnvBool("VLLM_ENABLED", y.VLLM.Enabled)
	if !vllmEnabled {
		// Auto-enable if any council member, chairman, or router uses the "local" provider.
		for _, slot := range slots {
			if slot.Provider == "local" {
				vllmEnabled = true
				break
			}
		}
		if chairmanProvider == "local" || routerProvider == "local" {
			vllmEnabled = true
		}
	}

	vllmModelName := getEnv("VLLM_MODEL_NAME", y.VLLM.ModelName)
	if vllmModelName == "" {
		vllmModelName = "microsoft/Phi-4-mini-instruct"
	}
	vllmDType := getEnv("VLLM_DTYPE", y.VLLM.DType)
	vllmMaxLen := getEnvInt("VLLM_MAX_MODEL_LEN", y.VLLM.MaxModelLen)
	if vllmMaxLen <= 0 {
		vllmMaxLen = 4096
	}
	vllmGPUUtil := getEnvFloat64("VLLM_GPU_MEMORY_UTIL", y.VLLM.GPUMemoryUtilization)
	if vllmGPUUtil <= 0 {
		vllmGPUUtil = 0.85
	}
	vllmQuant := getEnv("VLLM_QUANTIZATION", y.VLLM.Quantization)
	vllmTP := getEnvInt("VLLM_TENSOR_PARALLEL", y.VLLM.TensorParallelSize)
	if vllmTP <= 0 {
		vllmTP = 1
	}
	vllmSwap := getEnvInt("VLLM_SWAP_SPACE_GB", y.VLLM.SwapSpaceGB)
	vllmMaxSeqs := getEnvInt("VLLM_MAX_NUM_SEQS", y.VLLM.MaxNumSeqs)
	if vllmMaxSeqs <= 0 {
		vllmMaxSeqs = 16
	}
	vllmKVDType := getEnv("VLLM_KV_CACHE_DTYPE", y.VLLM.KVCacheDType)
	vllmCPUOffload := getEnvInt("VLLM_CPU_OFFLOAD_GB", y.VLLM.CPUOffloadGB)

	return &Config{
		ServerPort:                     port,
		Debug:                          debug,
		RAGServiceURL:                  ragURL,
		RedisAddr:                      redisAddr,
		RedisPassword:                  redisPwd,
		RedisDB:                        redisDB,
		CircuitBreakerFailureThreshold: cbFailureThresh,
		CircuitBreakerSuccessThreshold: cbSuccessThresh,
		CircuitBreakerTimeout:          cbTimeout,
		JWTSecret:                      jwtSecret,
		JWTExpiration:                  jwtExp,
		RateLimitRPS:                   rps,
		RateLimitBurst:                 burst,

		DatabaseURL:       dbURL,
		DBHost:            dbHost,
		DBPort:            dbPort,
		DBUser:            dbUser,
		DBPassword:        dbPass,
		DBName:            dbName,
		DBSSLMode:         dbSSLMode,
		DBMaxOpenConns:    maxOpenConns,
		DBMaxIdleConns:    maxIdleConns,
		DBConnMaxLifetime: connMaxLifetime,
		DBConnMaxIdleTime: connMaxIdleTime,

		OpenRouterAPIKey: orKey,
		OpenRouterURL:    orURL,
		GeminiAPIKey:     geminiKey,
		NVIDIANimAPIKey:  nimKey,
		NVIDIANimURL:     nimURL,

		VLLMURL: vllmURL,
		VLLMConfig: LocalModelConfig{
			Enabled:              vllmEnabled,
			ModelName:            vllmModelName,
			Quantization:         vllmQuant,
			DType:                vllmDType,
			GPUMemoryUtilization: vllmGPUUtil,
			MaxModelLen:          vllmMaxLen,
			TensorParallelSize:   vllmTP,
			SwapSpaceGB:          vllmSwap,
			MaxNumSeqs:           vllmMaxSeqs,
			KVCacheDType:         vllmKVDType,
			CPUOffloadGB:         vllmCPUOffload,
		},

		CouncilSize:   cSize,
		CouncilSlots:  slots,
		ChairmanSlot:  ModelSlot{Provider: chairmanProvider, Model: chairmanModel},
		RouterSlot:    ModelSlot{Provider: routerProvider, Model: routerModel},
		IngestionSlot: ModelSlot{Provider: ingestionProvider, Model: ingestionModel},

		StageTimeout:           stageTimeout,
		LLMTimeout:             llmTimeout,
		RequestTimeout:         reqTimeout,
		SemanticCacheThreshold: cacheThresh,
	}
}

// getEnv retrieves environment variables with fallbacks.
func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

// getEnvBool parses boolean environment variables.
func getEnvBool(key string, fallback bool) bool {
	if v := os.Getenv(key); v != "" {
		if b, err := strconv.ParseBool(v); err == nil {
			return b
		}
	}
	return fallback
}

// getEnvInt parses integer environment variables.
func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

// getEnvFloat64 parses float64 environment variables.
func getEnvFloat64(key string, fallback float64) float64 {
	if v := os.Getenv(key); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return fallback
}

// parseDuration helper to safely parse timeout durations.
func parseDuration(s string, fallback time.Duration) time.Duration {
	if s == "" {
		return fallback
	}
	if d, err := time.ParseDuration(s); err == nil {
		return d
	}
	return fallback
}
