"""Layout-aware OCR implementation preserving table structures and reading order."""

import io
import logging
from pathlib import Path

from app.models import OCRBlock, OCRResult, DocumentMetadata, ChunkType
from app.ocr.interface import OCRBackend

logger = logging.getLogger(__name__)


class LayoutAwareOCR(OCRBackend):
    """Layout-aware OCR backend using pdfplumber for table and layout structure preservation."""

    def name(self) -> str:
        """Returns the unique identifier of the OCR backend.

        Returns:
            str: Identifier name 'layout_aware'.
        """
        return "layout_aware"

    def process(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Processes input file using layout-aware page extraction.

        Args:
            file_bytes: Raw binary bytes of the input document.
            filename: Original file name with extension.

        Returns:
            OCRResult: Extracted blocks, bounding boxes, and document metadata.
        """
        ext = Path(filename).suffix.lower()

        if ext != ".pdf":
            logger.warning(f"LayoutAwareOCR is optimized for PDFs, got {ext}. Falling back to basic extraction.")
            return self._fallback_extraction(file_bytes, filename, ext)

        return self._process_pdf(file_bytes, filename)

    def _extract_page_tables(self, page, page_num: int) -> tuple[list[OCRBlock], list]:
        """Extract table blocks and their bounding boxes from a PDF page."""
        blocks: list[OCRBlock] = []
        table_bboxes = []
        tables = page.extract_tables()
        if not tables:
            return blocks, table_bboxes

        for table_obj in page.find_tables():
            table_bboxes.append(table_obj.bbox)

        for table in tables:
            table_text = self._format_table(table)
            if table_text.strip():
                blocks.append(
                    OCRBlock(
                        content=table_text,
                        block_type=ChunkType.TABLE,
                        page_number=page_num,
                        confidence=0.9,
                    )
                )
        return blocks, table_bboxes

    def _extract_page_paragraphs(self, page, table_bboxes: list, page_num: int) -> list[OCRBlock]:
        """Extract non-table text blocks and classify them into paragraphs/headings."""
        blocks: list[OCRBlock] = []
        if table_bboxes:
            def _not_in_table(obj):
                if obj.get("object_type") != "char":
                    return True
                x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                for tx0, ty0, tx1, ty1 in table_bboxes:
                    if tx0 <= x0 <= tx1 and ty0 <= top <= ty1:
                        return False
                return True
            non_table_page = page.filter(_not_in_table)
            text = non_table_page.extract_text() or ""
        else:
            text = page.extract_text() or ""

        if text.strip():
            for para in self._split_paragraphs(text):
                if para.strip():
                    blocks.append(
                        OCRBlock(
                            content=para.strip(),
                            block_type=self._classify_block(para),
                            page_number=page_num,
                            confidence=0.95,
                        )
                    )
        return blocks

    def _process_pdf(self, file_bytes: bytes, filename: str) -> OCRResult:
        """Extract tables and surrounding paragraphs using pdfplumber."""
        import pdfplumber

        blocks: list[OCRBlock] = []
        page_count = 0

        try:
            with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
                page_count = len(pdf.pages)
                for page_num, page in enumerate(pdf.pages, start=1):
                    tbl_blocks, bboxes = self._extract_page_tables(page, page_num)
                    blocks.extend(tbl_blocks)
                    para_blocks = self._extract_page_paragraphs(page, bboxes, page_num)
                    blocks.extend(para_blocks)
        except Exception as e:
            logger.error(f"Layout-aware OCR failed: {e}")

        return OCRResult(
            blocks=blocks,
            metadata=DocumentMetadata(
                file_name=filename,
                file_type=".pdf",
                page_count=page_count,
                has_tables=any(b.block_type == ChunkType.TABLE for b in blocks),
            ),
            ocr_method=self.name(),
        )

    def _format_table(self, table: list[list]) -> str:
        """Format 2D table array into markdown-style string representation."""
        if not table:
            return ""

        rows = []
        for row in table:
            cells = [str(cell).strip() if cell else "" for cell in row]
            rows.append("| " + " | ".join(cells) + " |")

        if len(rows) > 1:
            separator = "| " + " | ".join(["---"] * len(table[0])) + " |"
            rows.insert(1, separator)

        return "\n".join(rows)

    def _split_paragraphs(self, text: str) -> list[str]:
        """Split page text into paragraphs using double newlines and capital-sentence cues."""
        paragraphs = text.split("\n\n")

        if len(paragraphs) <= 1:
            lines = text.split("\n")
            result: list[str] = []
            current: list[str] = []

            for line in lines:
                stripped = line.strip()
                if not stripped:
                    if current:
                        result.append(" ".join(current))
                        current = []
                    continue

                if (
                    current
                    and current[-1].rstrip().endswith(".")
                    and stripped[0].isupper()
                ):
                    result.append(" ".join(current))
                    current = [stripped]
                else:
                    current.append(stripped)

            if current:
                result.append(" ".join(current))

            return result

        return paragraphs

    def _classify_block(self, text: str) -> ChunkType:
        """Classify block type (heading, caption, list, paragraph) using heuristics."""
        stripped = text.strip()

        if len(stripped) < 100 and not stripped.endswith(".") and stripped[0].isupper():
            words = stripped.split()
            if len(words) <= 10:
                return ChunkType.HEADING

        caption_prefixes = ("figure", "fig.", "fig ", "table", "chart", "diagram")
        if stripped.lower().startswith(caption_prefixes):
            return ChunkType.CAPTION

        if stripped.startswith(("•", "-", "*", "1.", "2.", "3.")):
            return ChunkType.LIST

        return ChunkType.PARAGRAPH

    def _fallback_extraction(self, file_bytes: bytes, filename: str, ext: str) -> OCRResult:
        """Fallback for non-PDF image extractions."""
        from app.ocr.tesseract import TesseractOCR
        return TesseractOCR().process(file_bytes, filename)
