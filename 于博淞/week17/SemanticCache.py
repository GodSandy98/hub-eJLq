import os
from typing import Any, Callable, List, Union

import faiss
import numpy as np
import redis


class SemanticCache:
    """缓存『问题 -> 回答』，靠语义相似命中（而非精确匹配）。

    设计：
    - 问题向量存进 FAISS（IndexFlatL2），索引落盘 {name}.index
    - 问题->回答存进 Redis（key: {name}key:{prompt}）
    - 问题按插入顺序 rpush 进 Redis list（{name}list），
      list 的下标和 FAISS 向量 id 一一对应
    """

    def __init__(
            self,
            name: str,
            embedding_method: Callable[[Union[str, List[str]]], Any],
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
            distance_threshold: float = 0.1,
    ):
        self.name = name
        self.embedding_method = embedding_method
        self.ttl = ttl
        self.distance_threshold = distance_threshold
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )

        self.index_path = f"{self.name}.index"
        if os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        else:
            self.index = None

    @property
    def _list_key(self):
        return self.name + "list"

    def _prompt_key(self, prompt: str):
        return self.name + "key:" + prompt

    def store(self, prompt: Union[str, List[str]], response: Union[str, List[str]]):
        if isinstance(prompt, str):
            prompt = [prompt]
            response = [response]

        embedding = np.asarray(self.embedding_method(prompt), dtype=np.float32)
        if self.index is None:
            self.index = faiss.IndexFlatL2(embedding.shape[1])

        self.index.add(embedding)
        faiss.write_index(self.index, self.index_path)

        try:
            with self.redis.pipeline() as pipe:
                for q, a in zip(prompt, response):
                    pipe.setex(self._prompt_key(q), self.ttl, a)
                    # rpush 保证 list 顺序与 FAISS 插入顺序一致，下标才对得上
                    pipe.rpush(self._list_key, q)
                return pipe.execute()
        except Exception as e:
            print(f"Store error: {e}")
            return -1

    def call(self, prompt: str):
        """语义检索：命中返回回答列表，未命中返回 None。"""
        if self.index is None:
            return None

        embedding = np.asarray(self.embedding_method(prompt), dtype=np.float32)

        k = min(100, self.index.ntotal)
        dis, ind = self.index.search(embedding, k=k)

        # 距离都超过阈值 -> 没有语义相近的缓存
        if dis[0][0] > self.distance_threshold:
            return None

        prompts = self.redis.lrange(self._list_key, 0, -1)
        hit_prompts = [
            prompts[idx]
            for d, idx in zip(dis[0], ind[0])
            if d <= self.distance_threshold and idx != -1
        ]

        return self.redis.mget([self._prompt_key(q.decode()) for q in hit_prompts])

    def clear_cache(self):
        prompts = self.redis.lrange(self._list_key, 0, -1)
        if prompts:
            self.redis.delete(*[self._prompt_key(q.decode()) for q in prompts])
        self.redis.delete(self._list_key)
        if os.path.exists(self.index_path):
            os.unlink(self.index_path)
        self.index = None


if __name__ == "__main__":
    def get_embedding(text):
        if isinstance(text, str):
            text = [text]
        # 用文本长度造一点区分度，方便观察命中/未命中
        return np.array([np.ones(768) * len(t) for t in text], dtype=np.float32)

    cache = SemanticCache(
        name="semantic_cache",
        embedding_method=get_embedding,
        ttl=360,
        redis_url="localhost",
        distance_threshold=0.1,
    )
    cache.clear_cache()

    cache.store(prompt="hello world", response="answer-for-hello-world")
    print("hit  ", cache.call(prompt="hello world"))   # 应返回对应回答

    cache.store(prompt="another question here", response="answer-2")
    print("hit  ", cache.call(prompt="hello world"))   # 仍应返回第一条的回答
    print("hit2 ", cache.call(prompt="another question here"))
