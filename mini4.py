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


# ============ 新增能力 2：生成答案（对话模型） ============
def get_answer(question: str, context: str) -> str:
    """把检索到的片段和问题拼成 prompt，交给豆包对话模型生成答案。

    注意：这里用的是"对话模型"（MODEL_EP_ID），
        跟 mini1 的"向量模型"（EMBEDDING_EP_ID）是两个不同的模型。
        向量模型负责"找"，对话模型负责"答"。
    """
    prompt = (
        "你是一个知识库问答助手。请只根据下面提供的资料回答问题，"
        "如果资料里没有答案，就回答：资料中未找到相关信息。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}\n\n"
        "答案："
    )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    body = {
        "model": MODEL_EP_ID,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(API_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ============ 主流程：完整 RAG 闭环 ============
if __name__ == "__main__":
    # 1. 一篇长文档（模拟真实知识库里的内容）
    document = (
        "龙卷风是一种强烈的旋转气流，常呈漏斗状云柱，风速极快，破坏力巨大。"
        "龙卷风的形成通常与强雷暴天气有关，需要不稳定的大气层结和强烈的风切变。"
        "当上升气流遇到水平旋转的空气时，会把它倾斜成垂直的旋转轴，进而形成龙卷风。"
        "龙卷风按破坏程度分为EF0到EF5六个等级，其中EF5级最强，风速可超过每小时三百公里。"
        "龙卷风多发于春夏之交，常伴随雷暴、冰雹等强对流天气。"
        "龙卷风来临前，气象部门通常会发布预警信息，公众应及时关注天气变化。"
        "遇到龙卷风时，应迅速躲进坚固的建筑物、地下室或远离窗户的内侧房间，切不可在空旷处停留，也不要躲在汽车或活动板房里。"
    )

    # 2. 切分
    chunks = chunk_text(document)
    print(f"文档共 {len(document)} 字，切成了 {len(chunks)} 段：")
    for i, c in enumerate(chunks):
        print(f"  [{i}] {c}")

    # 3. 每段向量化 + 入库（用独立表 rag_docs，避免和 mini3 的 docs 表混在一起）
    client = chromadb.PersistentClient(path="chroma_db")
    collection = client.get_or_create_collection(
        name="rag_docs",
        metadata={"hnsw:space": "cosine"},
    )
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = [get_embedding(c) for c in chunks]
    collection.upsert(ids=ids, documents=chunks, embeddings=embeddings)
    print(f"\n已入库 {len(chunks)} 个片段")

    # 4. 提问 -> 检索
    question = "遇到龙卷风应该怎么做？"
    q_vec = get_embedding(question)
    results = collection.query(
        query_embeddings=[q_vec],
        n_results=3,
    )

    # 5. 把检索到的几段拼成一段"资料"
    context = "\n".join(results["documents"][0])
    print(f"\n问题：{question}")
    print("检索到的片段：")
    for i, doc in enumerate(results["documents"][0], start=1):
        print(f"  {i}. {doc}")

    # 6. 生成答案
    print("\n正在生成答案...")
    answer = get_answer(question, context)
    print(f"\n答案：{answer}")