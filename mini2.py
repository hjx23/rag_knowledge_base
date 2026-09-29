import math
from mini1 import get_embedding


def cosine(a: list, b: list) -> float:
    # 1. 点积：对应位置相乘，再全部加起来
    dot = sum(x * y for x, y in zip(a, b))

    # 2. 模长：每个数平方、求和、开根号
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))

    # 3. 余弦 = 点积 / (模长A * 模长B)
    return dot / (mag_a * mag_b)


if __name__ == "__main__":
    v1 = get_embedding("龙卷风")
    v2 = get_embedding("龙卷风跑多快")
    v3 = get_embedding("今晚吃什么")

    print(f"龙卷风 vs 龙卷风跑多快：{cosine(v1, v2):.4f}")
    print(f"龙卷风 vs 今晚吃什么：{cosine(v1, v3):.4f}")
