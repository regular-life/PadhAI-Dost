# PadhAI Dost

PadhAI Dost ("Friend of Others" in Hindi) is an AI-powered chatbot that provides intelligent assistance for students to understand and master their course materials. It utilizes Google's Gemini API and Retrieval-Augmented Generation (RAG) to deliver personalized learning support.

## Project Architecture

### 1. **Core Components**
- **LLM:** Uses `Gemini-2.0-Flash` for natural language processing and response generation.
- **RAG Pipeline:**
  - **Chunking:** Documents are split into smaller, manageable text chunks.
  - **Embedding:** Uses `sentence-transformers/all-MiniLM-L6-v2` to generate vector representations.
  - **Vector Storage:** Stores embeddings in **ChromaDB** for efficient semantic search.
  - **Retrieval & Generation:** Retrieves relevant document chunks and generates responses using Gemini.
- **Document Processing:**
  - `PyPDF2` extracts text from PDFs.
  - `pytesseract` (OCR) extracts text from PNGs.
- **Frontend:** Built with `Streamlit` for an interactive web interface.

### 2. **Code Structure**
```
parai-dost/
│── app.py                # Streamlit-based web interface
│── document_loader.py    # Handles PDF/PNG text extraction
│── rag_pipeline.py       # Implements the RAG retrieval system
│── samjha_do.py          # Generates detailed document explanations
│── pucho.py              # Creates practice questions
│── requirements.txt      # Project dependencies
│── .env                  # Stores API keys
│── .chatbot_env/         # Virtual environment (ignored in version control)
```

## Installation & Setup

### 1. **Clone the Repository**
```sh
git clone https://github.com/regular-life/PadhAI-Dost
cd PadhAI-Dost
```

### 2. **Set Up a Virtual Environment**
```sh
python -m venv .chatbot_env
source .chatbot_env/bin/activate  # Windows: .chatbot_env\Scripts\activate
```

### 3. **Install Dependencies**
```sh
pip install -r requirements.txt
```

### 4. **Set Up API Keys**
- Create a `.env` file and add your API key:
  ```
  GEMINI_API_KEY=your_google_api_key
  ```

### 5. **Run the Application**
```sh
streamlit run app.py
```

## Key Features
- **AI-Powered Question Answering:** Ask document-related questions and get context-aware responses.
- **"Samjha Do" (Explain It To Me):** Provides explanations at different knowledge levels.
- **"Pucho" (Ask Me):** Generates practice questions based on document content.
- **Efficient RAG Pipeline:** Uses ChromaDB for optimized document retrieval.

## Future Enhancements
- Support for PDF, DOCX, PPTX, PNG file formats.
- Adaptive learning paths based on user progress.


---
PadhAI Dost brings AI-driven learning assistance, making studying more interactive and accessible.

