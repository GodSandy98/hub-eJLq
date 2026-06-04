import json
from typing import Any, Dict, List, Optional, Union

import Levenshtein  # pip install Levenshtein
import redis


class SemanticMessageHistory:
    """多轮对话历史。整个历史以 JSON 存在一个 Redis key 里。

    支持：
    - get_recent：按角色过滤 + 取最近 top_k 条
    - get_relevant：按内容相关度（子串匹配 + 编辑距离）取 top_k 条
    """

    def __init__(
            self,
            name: str,  # 会话名，类似 session id
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

    @property
    def _key(self):
        return f"semantic_history:{self.name}"

    def get_history(self) -> List[Dict[Any, Any]]:
        history = self.redis.get(self._key)
        if not history:
            return []
        return json.loads(history)

    def _save(self, history: List[Dict[Any, Any]]):
        self.redis.setex(self._key, self.ttl, json.dumps(history))

    def add_message(self, message: Union[Dict[Any, Any], List[Dict[Any, Any]]]):
        if isinstance(message, dict):
            message = [message]
        history = self.get_history()
        history.extend(message)
        self._save(history)

    def get_recent(self, role: Optional[str] = None, top_k: int = 10):
        history = self.get_history()
        if role:
            history = [m for m in history if m.get("role", "") == role]
        if top_k:
            history = history[-top_k:]
        return history

    def get_relevant(self, content: str, top_k: int = 10):
        history = self.get_history()
        selected = [m for m in history if content in m.get("content", "")]
        if not selected:
            return []
        selected.sort(
            key=lambda m: Levenshtein.ratio(m.get("content", ""), content),
            reverse=True,
        )
        if top_k:
            selected = selected[:top_k]
        return selected

    def delete_history(self, top_k: int = 10):
        """只保留最近 top_k 条。"""
        history = self.get_history()[-top_k:]
        self._save(history)

    def clear_history(self):
        return self.redis.delete(self._key)


if __name__ == "__main__":
    history = SemanticMessageHistory(
        name="my-session",
        redis_url="localhost",
    )
    history.clear_history()
    history.add_message([
        {"role": "user", "content": "hello, how are you?"},
        {"role": "llm", "content": "I'm doing fine, thanks."},
        {"role": "user", "content": "what is the weather going to be today?"},
        {"role": "llm", "content": "I don't know", "metadata": {"model": "gpt-4"}},
        {"role": "user", "content": "what is the weather going to be today?"},
    ])

    print("get_history     ", history.get_history())
    print("get_recent k=1  ", history.get_recent(top_k=1))
    print("get_recent user ", history.get_recent(role="user", top_k=1))
    print("relevant today  ", history.get_relevant("today", top_k=1))
    print("relevant thanks ", history.get_relevant("thanks", top_k=1))
