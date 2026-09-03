"""
Pydantic models for the Python RAG service.
"""

from pydantic import BaseModel, Field
from typing import Optional
from enum import Enum


class ChunkType(str, Enum):
    """Enumeration of semantic chunk structural types."""

    PARAGRAPH = "paragraph"
    TABLE = "table"
    CAPTION = "caption"
    HEADING = "heading"
    LIST = "list"
    UNKNOWN = "unknown"


class DocumentMetadata(BaseModel):
    """Metadata describing document inspection and physical layout attributes."""

    has_text_layer: bool = False
    is_scanned: bool = False
    has_tables: bool = False
    is_multicolumn: bool = False
    page_count: int = 0
    file_type: str = ""
    file_name: str = ""


class Chunk(BaseModel):
    """A single semantic text chunk with structural classification and page index."""

    content: str
    chunk_type: ChunkType = ChunkType.PARAGRAPH
    page_number: int = 0
    chunk_index: int = 0
    metadata: dict = Field(default_factory=dict)


class OCRBlock(BaseModel):
    """A raw block extracted from an OCR engine prior to chunking."""

    content: str
    block_type: ChunkType = ChunkType.PARAGRAPH
    page_number: int = 0
    bbox: Optional[list[float]] = None
    confidence: float = 1.0


class OCRResult(BaseModel):
    """Aggregated OCR engine execution output containing extracted blocks and document metadata."""

    blocks: list[OCRBlock] = Field(default_factory=list)
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    ocr_method: str = ""


class IngestRequest(BaseModel):
    """Optional payload parameters for document ingestion."""

    doc_id: Optional[str] = None


class IngestResponse(BaseModel):
    """Response payload returned upon successful document ingestion and vectorization."""

    doc_id: str
    chunk_count: int
    metadata: DocumentMetadata
    preview_text: str = ""
    message: str = "Document ingested successfully"


class RetrieveRequest(BaseModel):
    """Request payload for semantic vector similarity retrieval."""

    question: str
    doc_id: Optional[str] = None
    top_k: int = 5
    rerank: bool = True


class RetrieveResponse(BaseModel):
    """Response payload containing ranked semantic chunks matching the query."""

    chunks: list[Chunk]
    question: str
    doc_id: Optional[str] = None


class RetrieveAllRequest(BaseModel):
    """Request payload to fetch all document chunks for a document ID."""

    doc_id: str


class RetrieveAllResponse(BaseModel):
    """Response payload containing all sequential document chunks."""

    chunks: list[Chunk]
    doc_id: str
    total_chunks: int


class EmbedRequest(BaseModel):
    """Request payload to compute a text vector embedding."""

    text: str


class EmbedResponse(BaseModel):
    """Response payload containing the float32 vector embedding."""

    embedding: list[float]


class HealthResponse(BaseModel):
    """Service healthcheck status payload."""

    status: str = "healthy"
    version: str = "1.0.0"
    service: str = "python-rag"
