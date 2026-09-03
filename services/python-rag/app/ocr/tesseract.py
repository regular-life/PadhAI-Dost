"""Tesseract OCR backend for scanned PDF and raster image document processing."""

import io
import logging
from pathlib import Path

from PIL import Image
from PyPDF2 import PdfReader

from app.models import OCRBlock, OCRResult, DocumentMetadata, ChunkType
from app.ocr.interface import OCRBackend

logger = logging.getLogger(__name__)


class TesseractOCR(OCRBackend):
    """OCR backend using pytesseract for general-purpose image and scanned PDF text extraction."""

    def name(self) -> str:
        """Returns the unique identifier of the OCR backend.

        Returns:
            str: Identifier name 'tesseract'.
        """
        return "tesseract"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Extracts text from images or scanned PDFs using Tesseract OCR.

        Args:
            file_bytes: Raw binary bytes of the file.
            filename: Input filename with extension.

        Returns:
            OCRResult: Extracted OCR text blocks and document metadata.
        """
        import pytesseract

        ext = Path(filename).suffix.lower()
        blocks: list[OCRBlock] = []

        try:
            if ext in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
                blocks = self._process_image(file_bytes, pytesseract)
            elif ext == ".pdf":
                blocks = self._process_pdf(file_bytes, pytesseract)
            else:
                logger.warning(f"Tesseract: unsupported file type {ext}")
        except Exception as e:
            logger.error(f"Tesseract OCR failed: {e}")

        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(
                file_name=filename,
                file_type=ext,
                page_count=max(1, max((b.page_number for b in blocks), default=0)),
            ),
            ocr_method=self.name(),
        )

    def _process_image(self, file_bytes: bytes, pytesseract) -> list[OCRBlock]:
        """Extract blocks and confidence details from a single image using image_to_data."""
        img = Image.open(io.BytesIO(file_bytes))
        data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

        blocks: list[OCRBlock] = []
        current_block_num = -1
        current_text_parts: list[str] = []
        current_conf: list[float] = []

        for i in range(len(data["text"])):
            block_num = data["block_num"][i]
            text = data["text"][i].strip()
            conf = float(data["conf"][i])

            if block_num != current_block_num:
                if current_text_parts:
                    block_text = " ".join(current_text_parts)
                    if block_text.strip():
                        avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0
                        blocks.append(
                            OCRBlock(
                                content=block_text.strip(),
                                block_type=ChunkType.PARAGRAPH,
                                page_number=1,
                                confidence=max(0, avg_conf / 100.0),
                            )
                        )
                current_block_num = block_num
                current_text_parts = []
                current_conf = []

            if text and conf > 0:
                current_text_parts.append(text)
                current_conf.append(conf)

        if current_text_parts:
            block_text = " ".join(current_text_parts)
            if block_text.strip():
                avg_conf = sum(current_conf) / len(current_conf) if current_conf else 0
                blocks.append(
                    OCRBlock(
                        content=block_text.strip(),
                        block_type=ChunkType.PARAGRAPH,
                        page_number=1,
                        confidence=max(0, avg_conf / 100.0),
                    )
                )

        return blocks

    def _process_pdf(self, file_bytes: bytes, pytesseract) -> list[OCRBlock]:
        """Convert scanned PDF pages to images and process each page."""
        blocks: list[OCRBlock] = []

        try:
            from pdf2image import convert_from_bytes

            images = convert_from_bytes(file_bytes, dpi=300)

            for page_num, img in enumerate(images, start=1):
                text = pytesseract.image_to_string(img)
                if text.strip():
                    paragraphs = text.split("\n\n")
                    for para in paragraphs:
                        if para.strip():
                            blocks.append(
                                OCRBlock(
                                    content=para.strip(),
                                    block_type=ChunkType.PARAGRAPH,
                                    page_number=page_num,
                                    confidence=0.8,
                                )
                            )
        except ImportError:
            logger.warning("pdf2image not installed. Falling back to PyPDF2 text extraction.")
            reader = PdfReader(io.BytesIO(file_bytes))
            for page_num, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    blocks.append(
                        OCRBlock(
                            content=text.strip(),
                            block_type=ChunkType.PARAGRAPH,
                            page_number=page_num,
                            confidence=1.0,
                        )
                    )

        return blocks
