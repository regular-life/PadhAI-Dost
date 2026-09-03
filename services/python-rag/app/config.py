"""Configuration loader and settings models for the Python RAG service."""

import logging
import os
from functools import lru_cache
from pathlib import Path
from pydantic_settings import BaseSettings
import yaml

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """Configuration settings for the RAG service."""

    chroma_db_path: str = "/data/chroma_db"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False


@lru_cache()
def get_settings() -> Settings:
    """Loads configuration from config.yaml with environment variable overrides.

    Returns:
        Settings: Validated RAG application settings instance.
    """
    yaml_data = {}
    config_paths = ["/app/config.yaml", "config.yaml", "../../config.yaml", "../../../config.yaml"]
    
    for path_str in config_paths:
        path = Path(path_str)
        if path.exists():
            try:
                with open(path, "r") as f:
                    yaml_data = yaml.safe_load(f) or {}
                logger.info(f"Loaded configuration from {path}")
                break
            except Exception as e:
                logger.warning(f"Failed to read config file {path}: {e}")

    # Extract sections.
    rag_conf = yaml_data.get("rag", {})
    server_conf = yaml_data.get("server", {})

    chroma_db_path = os.getenv("CHROMA_DB_PATH", rag_conf.get("chroma_db_path", "/data/chroma_db"))
    embedding_model = os.getenv("EMBEDDING_MODEL", rag_conf.get("embedding_model", "BAAI/bge-small-en-v1.5"))
    reranker_model = os.getenv("RERANKER_MODEL", rag_conf.get("reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
    
    host = os.getenv("RAG_HOST", server_conf.get("host", "0.0.0.0"))
    port = int(os.getenv("RAG_PORT", server_conf.get("port", 8000)))
    debug = os.getenv("DEBUG", str(server_conf.get("debug", False))).lower() in ("true", "1", "t")

    return Settings(
        chroma_db_path=chroma_db_path,
        embedding_model=embedding_model,
        reranker_model=reranker_model,
        host=host,
        port=port,
        debug=debug,
    )
