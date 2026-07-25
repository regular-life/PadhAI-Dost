import unittest
import sys
import os

# Ensure python-rag root is in import path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.models import OCRResult, OCRBlock, ChunkType, Chunk, DocumentMetadata

class TestRAGModels(unittest.TestCase):
    def test_chunk_model_instantiation(self):
        chunk = Chunk(
            content="Sample text content for vector embedding",
            chunk_type=ChunkType.PARAGRAPH,
            page_number=1,
            chunk_index=0
        )
        self.assertEqual(chunk.content, "Sample text content for vector embedding")
        self.assertEqual(chunk.chunk_type, ChunkType.PARAGRAPH)

    def test_ocr_result_instantiation(self):
        ocr = OCRResult(
            blocks=[
                OCRBlock(content="Header", block_type=ChunkType.HEADING, page_number=1)
            ],
            metadata=DocumentMetadata(file_name="test_doc.pdf", page_count=2),
            ocr_method="pdfplumber"
        )
        self.assertEqual(len(ocr.blocks), 1)
        self.assertEqual(ocr.metadata.file_name, "test_doc.pdf")
        self.assertEqual(ocr.ocr_method, "pdfplumber")

if __name__ == "__main__":
    unittest.main()
