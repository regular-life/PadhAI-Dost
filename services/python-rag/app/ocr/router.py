"""Adaptive OCR router selecting optimal extraction engines based on document characteristics."""

import io
import logging
from pathlib import Path

from PyPDF2 import PdfReader

from app.models import DocumentMetadata, OCRResult, OCRBlock, ChunkType
from app.ocr.interface import OCRBackend
from app.ocr.tesseract import TesseractOCR
from app.ocr.layout_aware import LayoutAwareOCR

logger = logging.getLogger(__name__)


class DirectTextExtractor(OCRBackend):
    """Direct text extractor for PDFs that already possess a clean embedded text layer."""

    def name(self) -> str:
        """Returns the unique identifier of the extractor.

        Returns:
            str: Identifier name 'direct_text'.
        """
        return "direct_text"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Extracts embedded text directly from PDF pages.

        Args:
            file_bytes: Raw binary bytes of the PDF file.
            filename: Name of the input file.

        Returns:
            OCRResult: Extracted blocks and document metadata.
        """
        blocks: list[OCRBlock] = []
        ext = Path(filename).suffix.lower()
        page_count = 0

        if ext != ".pdf":
            logger.warning("DirectTextExtractor only supports PDFs")
            return OCRResult(
                blocks=blocks,
                metadata=DocumentMetadata(file_name=filename, file_type=ext),
                ocr_method=self.name(),
            )

        try:
            reader = PdfReader(io.BytesIO(file_bytes))
            page_count = len(reader.pages)
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    paragraphs = text.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            blocks.append(
                                OCRBlock(
                                    content=para.strip(),
                                    block_type=ChunkType.PARAGRAPH,
                                    page_number=page_num,
                                    confidence=1.0,
                                )
                            )
        except Exception as e:
            logger.error(f"Direct text extraction failed: {e}")

        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(
                file_name=filename,
                file_type=ext,
                page_count=page_count,
            ),
            ocr_method=self.name(),
        )


def route_ocr(file_bytes: bytes, filename: str, metadata: DocumentMetadata) -> OCRResult:
    """Choose the most appropriate extraction backend based on document metadata."""
    backend: OCRBackend

    if metadata.has_text_layer and not metadata.is_scanned:
        if metadata.has_tables:
            logger.info(f"Routing {filename} to layout_aware (tables found)")
            backend = LayoutAwareOCR()
        else:
            logger.info(f"Routing {filename} to direct_text (text layer found)")
            backend = DirectTextExtractor()
    elif metadata.has_tables:
        logger.info(f"Routing {filename} to layout_aware (tables found)")
        backend = LayoutAwareOCR()
    else:
        logger.info(f"Routing {filename} to tesseract (scanned/image)")
        backend = TesseractOCR()

    result = backend.process(file_bytes, filename)
    result.ocr_method = backend.name()

    return result
