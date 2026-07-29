#!/usr/bin/env python3
import sys
import time
from pathlib import Path

# Add the 'services/python-rag' directory to PYTHONPATH so we can import 'app'
repo_root = Path(__file__).parent.parent.absolute()
python_rag_dir = repo_root / "services/python-rag"
sys.path.insert(0, str(python_rag_dir))

from app.models import OCRResult, OCRBlock, ChunkType, DocumentMetadata
from app.chunking.layout_chunker import chunk_document

def generate_synthetic_page(page_number: int) -> list[OCRBlock]:
    blocks = []
    # 1 Heading
    blocks.append(OCRBlock(
        content=f"Heading for Page {page_number}",
        block_type=ChunkType.HEADING,
        page_number=page_number,
    ))
    
    # 5 Long Paragraphs (need splitting)
    long_para_text = ("This is a fairly long sentence that simulates typical text. " * 30)
    for _ in range(5):
        blocks.append(OCRBlock(
            content=long_para_text,
            block_type=ChunkType.PARAGRAPH,
            page_number=page_number,
        ))
        
    # 5 Short Paragraphs (need merging)
    short_para_text = "This is a short block. "
    for _ in range(5):
        blocks.append(OCRBlock(
            content=short_para_text,
            block_type=ChunkType.PARAGRAPH,
            page_number=page_number,
        ))
        
    # 1 Table
    blocks.append(OCRBlock(
        content="Col1 | Col2\nVal1 | Val2",
        block_type=ChunkType.TABLE,
        page_number=page_number,
    ))
    
    # 1 Caption
    blocks.append(OCRBlock(
        content=f"Figure {page_number}: A descriptive caption.",
        block_type=ChunkType.CAPTION,
        page_number=page_number,
    ))
    
    return blocks

def create_synthetic_document(pages: int) -> OCRResult:
    all_blocks = []
    for p in range(1, pages + 1):
        all_blocks.extend(generate_synthetic_page(p))
    return OCRResult(
        blocks=all_blocks,
        metadata=DocumentMetadata(page_count=pages),
        ocr_method="synthetic"
    )

def benchmark(pages: int, iterations: int = 10):
    doc = create_synthetic_document(pages)
    
    print(f"Benchmarking Layout Chunker with {pages} pages ({len(doc.blocks)} blocks)...")
    
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        chunks = chunk_document(doc)
        latencies.append(time.perf_counter() - start)
        
    avg_latency_ms = (sum(latencies) / len(latencies)) * 1000
    print(f"  Iterations: {iterations}")
    print(f"  Resulting Chunks: {len(chunks)}")
    print(f"  Average Latency: {avg_latency_ms:.2f} ms\n")

if __name__ == "__main__":
    import logging
    logging.getLogger("app.chunking.layout_chunker").setLevel(logging.WARNING)

    benchmark(10, iterations=50)
    benchmark(50, iterations=20)
    benchmark(200, iterations=10)
    benchmark(1000, iterations=5)
