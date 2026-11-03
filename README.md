# RAG 知识库问答

根据本地 txt 知识库，用 RAG 技术把文本切分成块、转成向量、存入 ChromaDB。提问时检索最相关的片段，交给大模型生成答案。

## 1. 准备资料

1. 在项目目录下新建一个 `docs` 文件夹；
2. 往里面放一个 `test.txt`，内容写你想让 AI 回答的资料（比如一段关于某本书、某个领域的知识）。

> 程序启动时会读取 `docs/test.txt`，把它切分、向量化后入库。没有这个文件，程序会报错。

## 2. 安装依赖

在项目目录下打开命令行，执行：

```
pip install -r requirements.txt
```

## 3. 配置密钥

1. 把 `model_config.example.py` 复制一份，重命名为 `model_config.py`；
2. 打开 `model_config.py`，改下面这几行（其余不用动）：

```python
API_KEY = "你的密钥"                    # 改成你的火山方舟 API Key
MODEL_EP_ID = "模型编号"                # 改成你的对话模型接入点 ID（ep-xxx）
EMBEDDING_EP_ID = "embedding模型名称"    # 改成你的向量模型名
```

> 密钥和模型 ID 在「火山引擎 - 火山方舟」控制台里申请。

## 4. 启动

在项目目录下执行：

```
uvicorn main:app --reload
```

启动成功后，浏览器打开：

```
http://127.0.0.1:8000/ask?question=你想问的问题
```

把「你想问的问题」换成要问的内容，例如 `http://127.0.0.1:8000/ask?question=红楼梦是谁写的`。
