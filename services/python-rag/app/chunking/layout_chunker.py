"""
Layout-Aware Chunking.

Replaces simple token-based chunking with intelligent, structure-preserving chunking:
- Paragraphs → individual chunks
- Tables → single chunk (never split a table)
- Figure + caption → grouped as one chunk
- Headings → attached to the following content

Metadata is preserved on each chunk (type, page number, source block).
"""

import logging
from app.models import OCRResult, OCRBlock, Chunk, ChunkType

logger = logging.getLogger(__name__)

# Maximum characters per chunk before splitting long paragraphs
MAX_CHUNK_CHARS = 1500
# Minimum characters for a chunk to be considered meaningful
MIN_CHUNK_CHARS = 50


def chunk_document(ocr_result: OCRResult) -> list[Chunk]:
    """
    Convert OCR output blocks into layout-aware chunks.

    Strategy:
    1. Tables are kept as single chunks — never split.
    2. Headings are merged with the paragraph that follows.
    3. Captions are merged with the preceding or following content.
    4. Long paragraphs are split at sentence boundaries.
    5. Short blocks are merged with neighbors.
    """
    blocks = ocr_result.blocks
    if not blocks:
        return []

    chunks: list[Chunk] = []
    chunk_index = 0
    i = 0

    while i < len(blocks):
        block = blocks[i]

        if block.block_type == ChunkType.TABLE:
            # Tables are always a single chunk
            chunks.append(
                Chunk(
                    content=block.content,
                    chunk_type=ChunkType.TABLE,
                    page_number=block.page_number,
                    chunk_index=chunk_index,
                    metadata={"ocr_method": ocr_result.ocr_method},
                )
            )
            chunk_index += 1
            i += 1

        elif block.block_type == ChunkType.HEADING:
            # Merge heading with next paragraph/content block
            heading_text = block.content
            page_num = block.page_number
            i += 1

            # Collect subsequent content until next heading or table
            body_parts: list[str] = []
            while i < len(blocks) and blocks[i].block_type not in (
                ChunkType.HEADING,
                ChunkType.TABLE,
            ):
                body_parts.append(blocks[i].content)
                page_num = blocks[i].page_number
                i += 1

            combined = heading_text + "\n\n" + "\n".join(body_parts) if body_parts else heading_text

            # Split if too long
            for sub_chunk in _split_long_text(combined, MAX_CHUNK_CHARS):
                chunks.append(
                    Chunk(
                        content=sub_chunk,
                        chunk_type=ChunkType.PARAGRAPH,
                        page_number=page_num,
                        chunk_index=chunk_index,
                        metadata={
                            "has_heading": True,
                            "ocr_method": ocr_result.ocr_method,
                        },
                    )
                )
                chunk_index += 1

        elif block.block_type == ChunkType.CAPTION:
            # Try to group caption with preceding or following content
            caption_text = block.content
            i += 1

            # If previous chunk exists and is a table or figure, merge
            if chunks and chunks[-1].chunk_type == ChunkType.TABLE:
                chunks[-1].content += "\n\n" + caption_text
                chunks[-1].metadata["has_caption"] = True
            else:
                # Stand-alone caption chunk
                chunks.append(
                    Chunk(
                        content=caption_text,
                        chunk_type=ChunkType.CAPTION,
                        page_number=block.page_number,
                        chunk_index=chunk_index,
                        metadata={"ocr_method": ocr_result.ocr_method},
                    )
                )
                chunk_index += 1

        else:
            # Regular paragraph or list block
            content = block.content

            if len(content) < MIN_CHUNK_CHARS and i + 1 < len(blocks):
                # Merge small blocks with the next one
                merged_parts = [content]
                while (
                    i + 1 < len(blocks)
                    and blocks[i + 1].block_type
                    not in (ChunkType.HEADING, ChunkType.TABLE)
                    and sum(len(p) for p in merged_parts) < MAX_CHUNK_CHARS
                ):
                    i += 1
                    merged_parts.append(blocks[i].content)

                content = "\n".join(merged_parts)

            # Split long paragraphs
            for sub_chunk in _split_long_text(content, MAX_CHUNK_CHARS):
                chunks.append(
                    Chunk(
                        content=sub_chunk,
                        chunk_type=block.block_type,
                        page_number=block.page_number,
                        chunk_index=chunk_index,
                        metadata={"ocr_method": ocr_result.ocr_method},
                    )
                )
                chunk_index += 1

            i += 1

    logger.info(f"Chunked document into {len(chunks)} chunks")
    return chunks


def _split_long_text(text: str, max_chars: int) -> list[str]:
    """
    Split text at sentence boundaries if it exceeds max_chars.
    Tries to keep chunks at natural break points.
    """
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    sentences = _split_sentences(text)
    current_chunk: list[str] = []
    current_length = 0

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_length + sentence_len > max_chars and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_len
        else:
            current_chunk.append(sentence)
            current_length += sentence_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Simple sentence splitter."""
    import re

    # Split on sentence-ending punctuation followed by space + capital letter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sentences if s.strip()]
