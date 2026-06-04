
## 第 0 步：地基 Prompt

```
我在学 redis-vl-python，想自己实现一套简化版的 LLM 缓存工具。
技术栈：Python + redis 库 + faiss + numpy。
每个类我会单独让你写，请保持风格一致：构造函数都接收 name/ttl/redis 连接参数；
每个文件末尾带 if __name__ == '__main__' 的测试代码。
先不要写，等我描述具体类。
```

---

## 第 1 步：EmbeddingsCache

**Prompt：**

```
写一个 EmbeddingsCache 类，作用是缓存『文本 → embedding 向量』。要求：
- 用 md5(文本) 当 Redis 的 key，向量用 tobytes() 存成 value，带 TTL
- 支持单条 str 或 List[str] 批量
- 方法：store(text, embedding)、call(text) 取回向量、delete(text)
- 用 redis pipeline 批量操作
- 末尾写个用 np.random.rand(768) 模拟 embedding 的测试
```

**撞到的坑：** 存的时候 `np.random.rand` 是 float64，取回时若写 `np.frombuffer(..., dtype=np.float32)` 就会出错。

**反馈 Prompt：**

```
我跑测试时取回的向量维度不对/报错，是不是 dtype 不一致？帮我统一一下。
```

---

## 第 2 步：SemanticMessageHistory

**第一轮（先要核心）：**

```
写 SemanticMessageHistory，存多轮对话。整个历史以 JSON 存在一个 Redis key 里。
先实现：add_message(messages) 追加、get_history() 取全部、clear_history() 清空。
```

**跑通后再加：**

```
现在加两个查询方法：
get_recent(role, top_k) 按角色取最近 N 条；
get_relevant(content, top_k) 找内容相关的，先用简单的子串匹配 + Levenshtein 编辑距离排序就行。
```

---

## 第 3 步：SemanticCache（首次引入向量检索，最易踩坑）

**Prompt：**

```
写 SemanticCache，缓存『问题→回答』，但靠语义相似命中而不是精确匹配。设计：
- 构造函数接收一个 embedding_method 回调（输入文本，输出 (n,dim) 的 numpy 向量）
- 用 FAISS 的 IndexFlatL2 存问题向量，索引落盘
- 问题→回答存 Redis；问题按顺序存进一个 Redis list，用 list 下标对齐 FAISS 的向量 id
- store(prompt, response) 和 call(prompt)：编码→FAISS 搜索→距离小于阈值才命中→用下标回查问题→去 Redis 取回答
- 有 distance_threshold 参数
```

**撞到的逻辑 bug：** 若 AI 用 `lpush`（左插），list 顺序和 FAISS add 顺序相反，下标对不上，命中错答案。

**反馈 Prompt：**

```
我存了两条数据，查第一条却返回了第二条的答案。
是不是 list 插入顺序和 FAISS 的 id 顺序反了？
```

---

## 第 4 步：SemanticRouter

**Prompt：**

```
写 SemanticRouter，做意图分类。构造函数接收 embedding_method。
- add_route(questions, target)：把一组示例问题编码成向量，都归到 target 标签下存起来
- route(question)：新问题编码→和所有已存向量算相似度→取最近的→返回对应 target；没有匹配（超阈值）返回 None
- 实现 __call__ 让 router(question) 等价于 router.route(question)
- 参考我之前的 SemanticCache 用 FAISS 的写法，保持风格一致
```
