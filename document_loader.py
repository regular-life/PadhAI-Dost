from PyPDF2 import PdfReader
from PIL import Image
import pytesseract


def load_document(file):
    """Loads text from a PDF or PNG file."""
    if file.type == "application/pdf":
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    elif file.type == "image/png":
        img = Image.open(file)
        text = pytesseract.image_to_string(img)
        return text
    else:
        raise ValueError(
            "Unsupported file type. Please upload a PDF or PNG file.")
