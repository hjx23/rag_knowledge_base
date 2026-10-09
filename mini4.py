import chromadb
from mini1 import get_embedding
import requests
from model_config import API_KEY,API_URL,MODEL_EP_ID

def chunk_text(text: str, size: int = 70, overlap: int = 10) -> list:
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks