"""FastAPI application entry point for the CouncilAI Python RAG service."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.routers import ingest, retrieve, retrieve_all, embed, search
from app.models import HealthResponse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles startup and shutdown lifecycle events."""
    logger.info("Starting Python RAG Service...")
    settings = get_settings()

    from app.embedding.transformer import TransformerEmbeddings
    from app.retrieval.reranker import Reranker

    # Preload embedding and reranking models at startup.
    logger.info(f"Preloading embedding model: {settings.embedding_model}")
    TransformerEmbeddings(settings.embedding_model)
    logger.info("Embedding model loaded successfully")

    logger.info(f"Preloading reranker model: {settings.reranker_model}")
    Reranker(settings.reranker_model)
    logger.info("Reranker model loaded successfully")

    yield
    logger.info("Shutting down Python RAG Service")


app = FastAPI(
    title="CouncilAI RAG Service",
    description="Document intelligence API with adaptive OCR, layout-aware chunking, and semantic retrieval.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest.router, tags=["Ingestion"])
app.include_router(retrieve.router, tags=["Retrieval"])
app.include_router(retrieve_all.router, tags=["Retrieval"])
app.include_router(embed.router, tags=["Embedding"])
app.include_router(search.router, tags=["Web Search"])


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Returns the service health status."""
    return HealthResponse()


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
    )
