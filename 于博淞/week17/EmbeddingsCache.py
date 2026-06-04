import hashlib
from typing import List, Union

import numpy as np
import redis


class EmbeddingsCache:
    """缓存『文本 -> embedding 向量』的精确匹配缓存。

    key   = f"{name}:{md5(text)}"
    value = embedding.tobytes()
    用文本哈希做 key，所以只有完全相同的文本才会命中（精确匹配，
    区别于 SemanticCache 的语义近似命中）。
    """

    # 存取统一用 float32，避免存 float64 取 float32 导致维度错乱
    DTYPE = np.float32

    def __init__(
            self,
            name: str,
            ttl: int = 3600 * 24,
            redis_url: str = "localhost",
            redis_port: int = 6379,
            redis_password: str = None,
    ):
        self.name = name
        self.ttl = ttl
        self.redis = redis.Redis(
            host=redis_url,
            port=redis_port,
            password=redis_password,
        )

    def _key(self, text: str) -> str:
        t_code = hashlib.md5(text.encode()).hexdigest()
        return f"{self.name}:{t_code}"

    def store(self, text: Union[str, List[str]], embedding: np.ndarray):
        """存入文本和对应向量。text 为单条 str 或 List[str]，
        embedding 为 (n, dim) 的 numpy 数组。"""
        if isinstance(text, str):
            text = [text]

        embedding = np.asarray(embedding, dtype=self.DTYPE)
        if embedding.ndim == 1:
            embedding = embedding[None, :]

        try:
            with self.redis.pipeline() as pipe:
                for t, vec in zip(text, embedding):
                    pipe.setex(self._key(t), self.ttl, vec.tobytes())
                return pipe.execute()
        except Exception as e:
            print(f"Store error: {e}")
            return -1

    def call(self, text: Union[str, List[str]]):
        """取回向量，未命中的位置返回 None，整体出错返回 None。"""
        if isinstance(text, str):
            text = [text]

        try:
            results = self.redis.mget([self._key(t) for t in text])
            embeddings = []
            for result in results:
                if result is None:
                    embeddings.append(None)
                else:
                    embeddings.append(np.frombuffer(result, dtype=self.DTYPE))
            return embeddings
        except Exception as e:
            print(f"Call error: {e}")
            return None

    def delete(self, text: Union[str, List[str]]):
        if isinstance(text, str):
            text = [text]

        try:
            return self.redis.delete(*[self._key(t) for t in text])
        except Exception as e:
            print(f"Delete error: {e}")
            return -1


if __name__ == "__main__":
    embed_cache = EmbeddingsCache(
        name="embedding_cache",
        ttl=360,
        redis_url="localhost",
    )

    def get_embedding(text):
        return np.random.rand(768)

    print("store ", embed_cache.store(text="hello world", embedding=get_embedding("hello world")))
    print("call  ", embed_cache.call(text="hello world"))
    print("delete", embed_cache.delete(text="hello world"))
    print("miss  ", embed_cache.call(text="hello world"))  # 已删除，应返回 [None]
