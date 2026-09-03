"""ChromaDB vector store adapter for persisting and retrieving document embeddings."""

import hashlib
import logging
from typing import Optional

from langchain_community.vectorstores import Chroma

from app.models import Chunk, ChunkType
from app.embedding.transformer import TransformerEmbeddings
from app.config import get_settings

logger = logging.getLogger(__name__)


class ChromaStore:
    """Vector store manager for ChromaDB collection operations."""

    def __init__(self, persist_directory: Optional[str] = None):
        """Initializes ChromaStore with directory path and embedding model.

        Args:
            persist_directory: Optional filesystem path for ChromaDB storage.
        """
        settings = get_settings()
        self.persist_directory = persist_directory or settings.chroma_db_path
        self.embeddings = TransformerEmbeddings(settings.embedding_model)

    def ingest(self, chunks: list[Chunk], doc_id: str) -> int:
        """Stores document chunks in ChromaDB with metadata.

        Args:
            chunks: List of semantic text chunks to embed and store.
            doc_id: Unique document identifier.

        Returns:
            int: Number of chunks successfully persisted.
        """
        if not chunks:
            logger.warning("No chunks to ingest")
            return 0

        texts = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "doc_id": doc_id,
                "chunk_type": chunk.chunk_type.value,
                "page_number": chunk.page_number,
                "chunk_index": chunk.chunk_index,
                **{k: str(v) for k, v in chunk.metadata.items()},
            }
            for chunk in chunks
        ]

        ids = [
            f"{doc_id}_{chunk.chunk_index}_{hashlib.md5(chunk.content[:100].encode()).hexdigest()[:8]}"
            for chunk in chunks
        ]

        db = Chroma.from_texts(
            texts=texts,
            embedding=self.embeddings,
            metadatas=metadatas,
            ids=ids,
            persist_directory=self.persist_directory,
            collection_name=self._collection_name(doc_id),
        )
        db.persist()

        logger.info(f"Ingested {len(chunks)} chunks for doc_id={doc_id}")
        return len(chunks)

    def retrieve(
        self,
        query: str,
        doc_id: Optional[str] = None,
        top_k: int = 5,
        rerank: bool = True,
    ) -> list[Chunk]:
        """Query relevant document chunks with optional Cross-Encoder re-ranking."""
        collection_name = self._collection_name(doc_id) if doc_id else "default"

        try:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
        except Exception as e:
            logger.error(f"Failed to open ChromaDB collection '{collection_name}': {e}")
            return []

        # Over-fetch candidate chunks to feed the cross-encoder.
        fetch_k = top_k * 3
        results = db.similarity_search_with_relevance_scores(query, k=fetch_k)

        chunks = []
        for doc, score in results:
            meta = doc.metadata or {}
            chunk = Chunk(
                content=doc.page_content,
                chunk_type=ChunkType(meta.get("chunk_type", "paragraph")),
                page_number=int(meta.get("page_number", 0)),
                chunk_index=int(meta.get("chunk_index", 0)),
                metadata={**meta, "relevance_score": score},
            )
            chunks.append(chunk)

        # Cross-encoder re-ranking.
        if rerank and len(chunks) > top_k:
            try:
                from app.retrieval.reranker import Reranker

                reranker = Reranker()
                documents = [chunk.content for chunk in chunks]
                reranked = reranker.rerank(query=query, documents=documents, top_k=top_k)

                reranked_chunks = []
                for orig_idx, rerank_score in reranked:
                    chunk = chunks[orig_idx]
                    chunk.metadata["rerank_score"] = float(rerank_score)
                    reranked_chunks.append(chunk)

                logger.info(f"Re-ranked {len(chunks)} candidates to {len(reranked_chunks)} for doc_id={doc_id}")
                return reranked_chunks

            except Exception as e:
                logger.warning(f"Re-ranking failed, falling back to vector score: {e}")

        return chunks[:top_k]

    def delete_collection(self, doc_id: str) -> bool:
        """Purge a document collection from ChromaDB."""
        try:
            import chromadb

            client = chromadb.PersistentClient(path=self.persist_directory)
            collection_name = self._collection_name(doc_id)
            client.delete_collection(collection_name)
            logger.info(f"Deleted collection: {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    @staticmethod
    def _collection_name(doc_id: Optional[str]) -> str:
        """Safely format doc_id as a valid ChromaDB collection name."""
        if not doc_id:
            return "default"
        # ChromaDB requires 3-63 chars, alphanumeric + underscores.
        safe_name = "".join(c if c.isalnum() else "_" for c in doc_id)
        return f"doc_{safe_name}"[:63]

    def get_document_text(self, doc_id: str) -> str:
        """Retrieve all text content for a document collection."""
        collection_name = self._collection_name(doc_id)
        try:
            db = Chroma(
                persist_directory=self.persist_directory,
                embedding_function=self.embeddings,
                collection_name=collection_name,
            )
            results = db.get()
            if results and results.get("documents"):
                return "\n\n".join(results["documents"])
            return ""
        except Exception as e:
            logger.error(f"Failed to get document text for {doc_id}: {e}")
            return ""
