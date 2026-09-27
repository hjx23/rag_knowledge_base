import requests
from model_config import API_KEY, EMBEDDING_URL, EMBEDDING_EP_ID


def get_embedding(text: str) -> list:
    """把一段文字变成一串坐标（向量）。"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    body = {
        "model": EMBEDDING_EP_ID,
        "input": [
            {"type": "text", "text": text}
        ]
    }
    resp = requests.post(EMBEDDING_URL, headers=headers, json=body, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["data"]["embedding"]


if __name__ == "__main__":
    text = "龙卷风是一种强烈的旋转气流，风速极快。"
    vec = get_embedding(text)

    print(f"原文：{text}")
    print(f"向量长度：{len(vec)} 个数字")
    print(f"前 10 个数字：{vec[:10]}")