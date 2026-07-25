import unittest
import sys
import os

# Ensure python-rag root is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models import OCRResult, OCRBlock, ChunkType
from app.chunking.layout_chunker import chunk_document

class TestLayoutChunker(unittest.TestCase):
    def test_chunk_document_paragraphs(self):
        ocr = OCRResult(
            blocks=[
                OCRBlock(content="Introduction", block_type=ChunkType.HEADING, page_number=1),
                OCRBlock(content="This is the introduction paragraph explaining Wasserstein GANs in detail.", block_type=ChunkType.PARAGRAPH, page_number=1),
                OCRBlock(content="| Metric | Value |\n|---|---|\n| L1 | 0.95 |", block_type=ChunkType.TABLE, page_number=1),
            ]
        )
        chunks = chunk_document(ocr)
        self.assertGreater(len(chunks), 0)

    def test_empty_document(self):
        ocr = OCRResult(blocks=[])
        chunks = chunk_document(ocr)
        self.assertEqual(len(chunks), 0)

if __name__ == "__main__":
    unittest.main()
