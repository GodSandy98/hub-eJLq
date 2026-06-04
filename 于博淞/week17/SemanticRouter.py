from typing import Any, Callable, List, Optional, Union

import faiss
import numpy as np


class SemanticRouter:
    """意图路由：把一个问题路由到最相似的预设类别（target）。

    本质是 SemanticCache 的"检索"部分换个返回值——返回 target 而非 response。

    设计：
    - add_route 把一组示例问题编码成向量加入 FAISS，
      并记录每个向量对应的 target（self.targets，下标与 FAISS id 对齐）
    - route 编码新问题，取最近的向量，距离在阈值内则返回其 target，否则 None
    """

    def __init__(
            self,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            distance_threshold: float = 0.1,
    ):
        self.embedding_method = embedding_method
        self.distance_threshold = distance_threshold
        self.index = None
        self.targets: List[str] = []  # 下标 i 对应 FAISS 中第 i 个向量的 target

    def add_route(self, questions: List[str], target: str):
        embedding = np.asarray(self.embedding_method(questions), dtype=np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1])

        self.index.add(embedding)
        self.targets.extend([target] * len(questions))

    def route(self, question: str) -> Optional[str]:
        if self.index is None or self.index.ntotal == 0:
            return None

        embedding = np.asarray(self.embedding_method(question), dtype=np.float32)
        dis, ind = self.index.search(embedding, k=1)

        if dis[0][0] > self.distance_threshold:
            return None
        return self.targets[ind[0][0]]

    def __call__(self, question: str) -> Optional[str]:
        return self.route(question)


if __name__ == "__main__":
    # 简单 mock：按字符集合做 one-hot，相同/相近文本向量更接近
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        vecs = []
        for t in text:
            v = np.zeros(128, dtype=np.float32)
            for ch in t.lower():
                v[ord(ch) % 128] += 1.0
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)

    router = SemanticRouter(embedding_method=get_embedding, distance_threshold=5.0)
    router.add_route(
        questions=["Hi, good morning", "Hi, good afternoon"],
        target="greeting",
    )
    router.add_route(
        questions=["如何退货", "怎么申请退款"],
        target="refund",
    )

    print("route1", router("Hi, good morning"))   # -> greeting
    print("route2", router.route("如何退货"))        # -> refund
    print("route3", router("完全不相关的句子 xyz"))   # -> None 或最近类别
