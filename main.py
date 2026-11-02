import chromadb
from fastapi import FastAPI

from mini1 import get_embedding
from mini4 import chunk_text, get_answer

app = FastAPI()

# ============ 知识库的原始文档 ============
with open("./docs/test.txt", encoding= "utf-8") as f:
    DOCUMENT = f.read()

# ============ 建索引：只在服务启动时做一次 ============
client = chromadb.PersistentClient(path="chroma_db")
collection = client.get_or_create_collection(
    name="rag_docs",
    metadata={"hnsw:space": "cosine"},
)

if collection.count() == 0:
    chunks = chunk_text(DOCUMENT)
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(c) for c in chunks]
    collection.upsert(ids=ids, embeddings=embeddings, documents=chunks)
    print(f"[建索引] 首次启动，已把 {len(chunks)} 个片段向量化并入库")
else:
    print(f"[建索引] 索引已存在（{collection.count()} 个片段），跳过，不重复烧 token")


# ============ 接口 1：健康检查（验证服务活着）============
@app.get("/")
def read_root():
    return {"message": "hello"}


# ============ 接口 2：问答 ============
@app.get("/ask")
def ask(question: str):
    # 1. 问题转向量
    q_vec = get_embedding(question)

    # 2. 检索最相关的片段
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
    )
    sources = results["documents"][0]

    # 3. 拼成 context，交给对话模型生成答案
    context = "\n".join(sources)
    answer = get_answer(question, context)

    # 4. 打包成字典返回
    return {
        "question": question,
        "answer": answer,
        "sources": sources,
    }
