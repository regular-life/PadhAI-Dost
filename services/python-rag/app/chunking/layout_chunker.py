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


def _create_table_chunk(block: OCRBlock, ocr_method: str, chunk_index: int) -> Chunk:
    """Create a standalone table chunk without splitting."""
    return Chunk(
        content=block.content,
        chunk_type=ChunkType.TABLE,
        page_number=block.page_number,
        chunk_index=chunk_index,
        metadata={"ocr_method": ocr_method},
    )


def _process_heading(
    blocks: list[OCRBlock],
    i: int,
    ocr_method: str,
    chunk_index: int,
) -> tuple[list[Chunk], int, int]:
    """Merge heading with following text blocks and split if needed."""
    block = blocks[i]
    heading_text = block.content
    page_num = block.page_number
    i += 1

    body_parts: list[str] = []
    while i < len(blocks) and blocks[i].block_type not in (
        ChunkType.HEADING,
        ChunkType.TABLE,
    ):
        body_parts.append(blocks[i].content)
        page_num = blocks[i].page_number
        i += 1

    combined = heading_text + "\n\n" + "\n".join(body_parts) if body_parts else heading_text
    chunks: list[Chunk] = []
    for sub_chunk in _split_long_text(combined, MAX_CHUNK_CHARS):
        chunks.append(
            Chunk(
                content=sub_chunk,
                chunk_type=ChunkType.PARAGRAPH,
                page_number=page_num,
                chunk_index=chunk_index,
                metadata={"has_heading": True, "ocr_method": ocr_method},
            )
        )
        chunk_index += 1

    return chunks, i, chunk_index


def _process_caption(
    block: OCRBlock,
    chunks: list[Chunk],
    ocr_method: str,
    chunk_index: int,
) -> int:
    """Group caption with preceding table or append standalone caption chunk."""
    caption_text = block.content
    if chunks and chunks[-1].chunk_type == ChunkType.TABLE:
        chunks[-1].content += "\n\n" + caption_text
        chunks[-1].metadata["has_caption"] = True
        return chunk_index

    chunks.append(
        Chunk(
            content=caption_text,
            chunk_type=ChunkType.CAPTION,
            page_number=block.page_number,
            chunk_index=chunk_index,
            metadata={"ocr_method": ocr_method},
        )
    )
    return chunk_index + 1


def _process_paragraph(
    blocks: list[OCRBlock],
    i: int,
    ocr_method: str,
    chunk_index: int,
) -> tuple[list[Chunk], int, int]:
    """Merge short paragraph blocks or split long ones."""
    block = blocks[i]
    content = block.content

    if len(content) < MIN_CHUNK_CHARS and i + 1 < len(blocks):
        merged_parts = [content]
        while (
            i + 1 < len(blocks)
            and blocks[i + 1].block_type not in (ChunkType.HEADING, ChunkType.TABLE)
            and sum(len(p) for p in merged_parts) < MAX_CHUNK_CHARS
        ):
            i += 1
            merged_parts.append(blocks[i].content)
        content = "\n".join(merged_parts)

    chunks: list[Chunk] = []
    for sub_chunk in _split_long_text(content, MAX_CHUNK_CHARS):
        chunks.append(
            Chunk(
                content=sub_chunk,
                chunk_type=block.block_type,
                page_number=block.page_number,
                chunk_index=chunk_index,
                metadata={"ocr_method": ocr_method},
            )
        )
        chunk_index += 1

    return chunks, i + 1, chunk_index


def chunk_document(ocr_result: OCRResult) -> list[Chunk]:
    """
    Convert OCR output blocks into layout-aware chunks.

    Args:
        ocr_result: The parsed OCR output blocks.

    Returns:
        A list of layout-aware text and table chunks.
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
            chunks.append(_create_table_chunk(block, ocr_result.ocr_method, chunk_index))
            chunk_index += 1
            i += 1
        elif block.block_type == ChunkType.HEADING:
            new_chunks, i, chunk_index = _process_heading(blocks, i, ocr_result.ocr_method, chunk_index)
            chunks.extend(new_chunks)
        elif block.block_type == ChunkType.CAPTION:
            chunk_index = _process_caption(block, chunks, ocr_result.ocr_method, chunk_index)
            i += 1
        else:
            new_chunks, i, chunk_index = _process_paragraph(blocks, i, ocr_result.ocr_method, chunk_index)
            chunks.extend(new_chunks)

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
