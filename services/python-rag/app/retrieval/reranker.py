"""Cross-encoder semantic reranker for fine-grained document relevance ranking."""

import logging
from sentence_transformers import CrossEncoder
from app.config import get_settings

logger = logging.getLogger(__name__)


class Reranker:
    """Lightweight Cross-Encoder re-ranker using ms-marco-MiniLM."""

    _instance = None  # Class-level singleton to avoid reloading model weights.

    def __init__(self, model_name: str | None = None):
        if model_name is None:
            settings = get_settings()
            model_name = settings.reranker_model
            
        if Reranker._instance is None:
            logger.info(f"Loading re-ranker model: {model_name}")
            Reranker._instance = CrossEncoder(model_name)
        self.model = Reranker._instance

    def rerank(self, query: str, documents: list[str], top_k: int = 5) -> list[tuple[int, float]]:
        """Re-rank candidate documents based on cross-attention scores.

        Args:
            query: The search query string.
            documents: Candidate document text strings.
            top_k: Number of sorted results to return.

        Returns:
            List of (original_index, score) pairs sorted by relevance.
        """
        if not documents:
            return []

        pairs = [(query, doc) for doc in documents]
        scores = self.model.predict(pairs)

        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        return indexed_scores[:top_k]
