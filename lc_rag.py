from langchain_core.embeddings import Embeddings
from mini1 import get_embedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

class DoubaoEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [get_embedding(t) for t in texts]

    def embed_query(self, text):
        return get_embedding(text)

loader = TextLoader("./docs/test.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=10)
chunks = splitter.split_documents(docs)