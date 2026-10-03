import chromadb
from mini1 import get_embedding


# 1. 打开（或创建）本地向量库：数据存到 chroma_db 这个文件夹
client = chromadb.PersistentClient(path="chroma_db")

# 2. 建一个集合（collection），相当于数据库里的一张表
#    hnsw:space = cosine 表示"用余弦距离"来找最近的向量
collection = client.get_or_create_collection(
    name="docs",
    metadata={"hnsw:space": "cosine"},
)

# 3. 准备要入库的片段（3 个龙卷风相关 + 1 个无关干扰项）
segments = [
    "龙卷风是一种强烈的旋转气流，风速极快。",
    "龙卷风按破坏程度分为EF0到EF5六个等级，EF5级最强。",
    "龙卷风多发于春夏之交，常伴随雷暴天气。",
    "今晚吃什么，火锅还是饺子。",
]

# 4. 每个片段向量化，再连同原文一起入库
ids = [f"seg_{i}" for i in range(len(segments))]
embeddings = [get_embedding(s) for s in segments]

# upsert = 有就更新、没有就插入，重复跑不会报错
collection.upsert(
    ids=ids,
    documents=segments,
    embeddings=embeddings,
)
print(f"已入库 {len(segments)} 个片段")

# 5. 提问 → 向量化 → 检索最相关的 3 个片段
question = "龙卷风的风速是多少？"
q_vec = get_embedding(question)

results = collection.query(
    query_embeddings=[q_vec],
    n_results=3,
)

# 6. 打印结果
print(f"\n问题：{question}")
print("最相关的 3 个片段：")
for i, doc in enumerate(results["documents"][0], start=1):
    dist = results["distances"][0][i - 1]
    print(f"  {i}. {doc}  (距离 {dist:.4f})")
