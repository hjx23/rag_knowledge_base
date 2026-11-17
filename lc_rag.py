from langchain_core.embeddings import Embeddings
from mini1 import get_embedding
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from model_config import API_KEY,MODEL_EP_ID

class DoubaoEmbeddings(Embeddings):
    def embed_documents(self, texts):#texts是列表吗？如果是，列表里面是什么格式
        return [get_embedding(t) for t in texts]

    def embed_query(self, text):#text是字符串?
        return get_embedding(text)

loader = TextLoader("./docs/test.txt", encoding="utf-8")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=70, chunk_overlap=10)#splitter是个什么？对象吗？
chunks = splitter.split_documents(docs)#这里能够直接处理docs这个document对象吗？既然是给了70和10，为什么还能按分隔符切分呢？

# vectorstore = Chroma.from_documents(
#     documents=chunks,
#     embedding=DoubaoEmbeddings(),
#     persist_directory="./chroma_lc"
# )

vectorstore = Chroma(
    persist_directory="./chroma_lc",
    embedding_function=DoubaoEmbeddings(),
)#vectorstore是什么？
if vectorstore._collection.count() == 0:
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=DoubaoEmbeddings(),
        persist_directory="./chroma_lc",
    )#这样是不是就算换了文档，也依旧不会入库？

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

prompt = ChatPromptTemplate.from_messages([
    ("system","你是一个知识库问答助手。严格只根据下面的资料回答，禁止使用资料以外的任何知识。资料里没有就回答：资料中未找到相关信息。\n\n资料：\n{context}"),
    ("human","{question}"),
])#from_messages这是什么？

llm = ChatOpenAI(
    model=MODEL_EP_ID,
    base_url="https://ark.cn-beijing.volces.com/api/v3",
    api_key=API_KEY,
    temperature=0.1,
)#llm是什么啊？

def ask(question):
    docs = retriever.invoke(question)#invoke是什么啊？
    context = "\n".join(d.page_content for d in docs)#docs能进行循环？
    return llm.invoke(prompt.format(context=context, question=question)).content#为什么要.content  返回什么啊？

if __name__ == "__main__":
    print(ask("红楼梦是谁写的"))