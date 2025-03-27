from transformers import AutoTokenizer, AutoModel
import torch
from langchain.embeddings.base import Embeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import GoogleGenerativeAI

class TransformerEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
    
    def embed_documents(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        embeddings = []
        for text in texts:
            # Tokenize input text
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs)
            # Mean pooling on token embeddings
            token_embeddings = outputs.last_hidden_state  # shape: (batch_size, seq_length, hidden_size)
            input_mask_expanded = inputs["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            embedding = (sum_embeddings / sum_mask).squeeze().tolist()
            embeddings.append(embedding)
        return embeddings
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

def create_rag_pipeline(text, api_key):
    # Instantiate the free transformer embeddings model
    embeddings_model = TransformerEmbeddings()

    texts = [text]

    # Create a Chroma vectorstore from texts using our transformer embeddings
    db = Chroma.from_texts(texts, embeddings_model, persist_directory="./chroma_db")
    db.persist()

    llm = GoogleGenerativeAI(model="gemini-2.0-flash", api_key=api_key)
    qa = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=db.as_retriever())
    return qa