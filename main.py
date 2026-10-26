import chromadb
from fastapi import FastAPI

from mini1 import get_embedding
from mini4 import chunk_text, get_answer

app = FastAPI()

# ============ 知识库的原始文档 ============
DOCUMENT = (
    "龙卷风是一种强烈的旋转气流，常呈漏斗状云柱，风速极快，破坏力巨大。"
    "龙卷风的形成通常与强雷暴天气有关，需要不稳定的大气层结和强烈的风切变。"
    "当上升气流遇到水平旋转的空气时，会把它倾斜成垂直的旋转轴，进而形成龙卷风。"
    "龙卷风按破坏程度分为EF0到EF5六个等级，其中EF5级最强，风速可超过每小时三百公里。"
    "龙卷风多发于春夏之交，常伴随雷暴、冰雹等强对流天气。"
    "龙卷风来临前，气象部门通常会发布预警信息，公众应及时关注天气变化。"
    "遇到龙卷风时，应迅速躲进坚固的建筑物、地下室或远离窗户的内侧房间，切不可在空旷处停留，也不要躲在汽车或活动板房里。"
)

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
