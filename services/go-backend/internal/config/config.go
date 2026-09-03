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
type serverSettings struct {
	port        string
	debug       bool
	jwtSecret   string
	jwtExp      time.Duration
	rps         int
	burst       int
	cacheThresh float64
	ragURL      string
}

type redisSettings struct {
	addr            string
	password        string
	db              int
	cbFailureThresh int
	cbSuccessThresh int
	cbTimeout       time.Duration
}

type databaseSettings struct {
	url             string
	host            string
	port            string
	user            string
	password        string
	name            string
	sslMode         string
	maxOpenConns    int
	maxIdleConns    int
	connMaxLifetime time.Duration
	connMaxIdleTime time.Duration
}

type providerSettings struct {
	openRouterURL string
	openRouterKey string
	nvidiaNIMURL  string
	nvidiaNIMKey  string
	vllmURL       string
	geminiKey     string
}

type councilSettings struct {
	size         int
	slots        []ModelSlot
	chairman     ModelSlot
	router       ModelSlot
	ingestion    ModelSlot
	stageTimeout time.Duration
	llmTimeout   time.Duration
	reqTimeout   time.Duration
}

func loadYAML() yamlSchema {
	var y yamlSchema
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
	return y
}

func loadServerSettings(y *yamlSchema) serverSettings {
	port := getEnv("SERVER_PORT", y.Server.Port)
	if port == "" {
		port = "8080"
	}
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
	ragURL := getEnv("RAG_SERVICE_URL", y.RAG.ServiceURL)
	if ragURL == "" {
		ragURL = "http://python-rag:8000"
	}

	return serverSettings{
		port:        port,
		debug:       getEnvBool("DEBUG", y.Server.Debug),
		jwtSecret:   jwtSecret,
		jwtExp:      jwtExp,
		rps:         rps,
		burst:       burst,
		cacheThresh: cacheThresh,
		ragURL:      ragURL,
	}
}

func loadRedisSettings(y *yamlSchema) redisSettings {
	addr := getEnv("REDIS_ADDR", y.Redis.Addr)
	if addr == "" {
		addr = "redis:6379"
	}
	cbFailureThresh := getEnvInt("CIRCUIT_BREAKER_FAILURE_THRESHOLD", y.Redis.FailureThreshold)
	if cbFailureThresh <= 0 {
		cbFailureThresh = 3
	}
	cbSuccessThresh := getEnvInt("CIRCUIT_BREAKER_SUCCESS_THRESHOLD", y.Redis.SuccessThreshold)
	if cbSuccessThresh <= 0 {
		cbSuccessThresh = 2
	}
	return redisSettings{
		addr:            addr,
		password:        getEnv("REDIS_PASSWORD", y.Redis.Password),
		db:              getEnvInt("REDIS_DB", y.Redis.DB),
		cbFailureThresh: cbFailureThresh,
		cbSuccessThresh: cbSuccessThresh,
		cbTimeout:       parseDuration(getEnv("CIRCUIT_BREAKER_TIMEOUT", y.Redis.Timeout), 10*time.Second),
	}
}

func loadDatabaseSettings(y *yamlSchema) databaseSettings {
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

	return databaseSettings{
		url:             dbURL,
		host:            dbHost,
		port:            dbPort,
		user:            dbUser,
		password:        dbPass,
		name:            dbName,
		sslMode:         dbSSLMode,
		maxOpenConns:    maxOpenConns,
		maxIdleConns:    maxIdleConns,
		connMaxLifetime: parseDuration(getEnv("DB_CONN_MAX_LIFETIME", y.Database.ConnMaxLifetime), 1*time.Hour),
		connMaxIdleTime: parseDuration(getEnv("DB_CONN_MAX_IDLE_TIME", y.Database.ConnMaxIdleTime), 30*time.Minute),
	}
}

func loadProviderSettings(y *yamlSchema) providerSettings {
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

	return providerSettings{
		openRouterURL: orURL,
		openRouterKey: getEnv("OPENROUTER_API_KEY", y.Keys.OpenRouter),
		nvidiaNIMURL:  nimURL,
		nvidiaNIMKey:  getEnv("NVIDIA_NIM_API_KEY", y.Keys.NVIDIANim),
		vllmURL:       vllmURL,
		geminiKey:     getEnv("GEMINI_API_KEY", y.Keys.Gemini),
	}
}

func loadCouncilSettings(y *yamlSchema) councilSettings {
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

	return councilSettings{
		size:         cSize,
		slots:        slots,
		chairman:     ModelSlot{Provider: chairmanProvider, Model: chairmanModel},
		router:       ModelSlot{Provider: routerProvider, Model: routerModel},
		ingestion:    ModelSlot{Provider: ingestionProvider, Model: ingestionModel},
		stageTimeout: parseDuration(getEnv("STAGE_TIMEOUT", y.Council.Timeouts.Stage), 30*time.Second),
		llmTimeout:   parseDuration(getEnv("LLM_TIMEOUT", y.Council.Timeouts.LLM), 120*time.Second),
		reqTimeout:   parseDuration(getEnv("REQUEST_TIMEOUT", y.Council.Timeouts.Request), 120*time.Second),
	}
}

func loadVLLMSettings(y *yamlSchema, slots []ModelSlot, chairmanProvider, routerProvider string) LocalModelConfig {
	vllmEnabled := getEnvBool("VLLM_ENABLED", y.VLLM.Enabled)
	if !vllmEnabled {
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

	modelName := getEnv("VLLM_MODEL_NAME", y.VLLM.ModelName)
	if modelName == "" {
		modelName = "microsoft/Phi-4-mini-instruct"
	}
	maxLen := getEnvInt("VLLM_MAX_MODEL_LEN", y.VLLM.MaxModelLen)
	if maxLen <= 0 {
		maxLen = 4096
	}
	gpuUtil := getEnvFloat64("VLLM_GPU_MEMORY_UTIL", y.VLLM.GPUMemoryUtilization)
	if gpuUtil <= 0 {
		gpuUtil = 0.85
	}
	tp := getEnvInt("VLLM_TENSOR_PARALLEL", y.VLLM.TensorParallelSize)
	if tp <= 0 {
		tp = 1
	}
	maxSeqs := getEnvInt("VLLM_MAX_NUM_SEQS", y.VLLM.MaxNumSeqs)
	if maxSeqs <= 0 {
		maxSeqs = 16
	}

	return LocalModelConfig{
		Enabled:              vllmEnabled,
		ModelName:            modelName,
		Quantization:         getEnv("VLLM_QUANTIZATION", y.VLLM.Quantization),
		DType:                getEnv("VLLM_DTYPE", y.VLLM.DType),
		GPUMemoryUtilization: gpuUtil,
		MaxModelLen:          maxLen,
		TensorParallelSize:   tp,
		SwapSpaceGB:          getEnvInt("VLLM_SWAP_SPACE_GB", y.VLLM.SwapSpaceGB),
		MaxNumSeqs:           maxSeqs,
		KVCacheDType:         getEnv("VLLM_KV_CACHE_DTYPE", y.VLLM.KVCacheDType),
		CPUOffloadGB:         getEnvInt("VLLM_CPU_OFFLOAD_GB", y.VLLM.CPUOffloadGB),
	}
}

// Load reads and parses configuration from config.yaml and environment variables.
func Load() *Config {
	y := loadYAML()
	srv := loadServerSettings(&y)
	red := loadRedisSettings(&y)
	db := loadDatabaseSettings(&y)
	prov := loadProviderSettings(&y)
	csl := loadCouncilSettings(&y)
	vllm := loadVLLMSettings(&y, csl.slots, csl.chairman.Provider, csl.router.Provider)

	return &Config{
		ServerPort:                     srv.port,
		Debug:                          srv.debug,
		RAGServiceURL:                  srv.ragURL,
		RedisAddr:                      red.addr,
		RedisPassword:                  red.password,
		RedisDB:                        red.db,
		CircuitBreakerFailureThreshold: red.cbFailureThresh,
		CircuitBreakerSuccessThreshold: red.cbSuccessThresh,
		CircuitBreakerTimeout:          red.cbTimeout,
		JWTSecret:                      srv.jwtSecret,
		JWTExpiration:                  srv.jwtExp,
		RateLimitRPS:                   srv.rps,
		RateLimitBurst:                 srv.burst,

		DatabaseURL:       db.url,
		DBHost:            db.host,
		DBPort:            db.port,
		DBUser:            db.user,
		DBPassword:        db.password,
		DBName:            db.name,
		DBSSLMode:         db.sslMode,
		DBMaxOpenConns:    db.maxOpenConns,
		DBMaxIdleConns:    db.maxIdleConns,
		DBConnMaxLifetime: db.connMaxLifetime,
		DBConnMaxIdleTime: db.connMaxIdleTime,

		OpenRouterAPIKey: prov.openRouterKey,
		OpenRouterURL:    prov.openRouterURL,
		GeminiAPIKey:     prov.geminiKey,
		NVIDIANimAPIKey:  prov.nvidiaNIMKey,
		NVIDIANimURL:     prov.nvidiaNIMURL,

		VLLMURL:    prov.vllmURL,
		VLLMConfig: vllm,

		CouncilSize:   csl.size,
		CouncilSlots:  csl.slots,
		ChairmanSlot:  csl.chairman,
		RouterSlot:    csl.router,
		IngestionSlot: csl.ingestion,

		StageTimeout:           csl.stageTimeout,
		LLMTimeout:             csl.llmTimeout,
		RequestTimeout:         csl.reqTimeout,
		SemanticCacheThreshold: srv.cacheThresh,
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
