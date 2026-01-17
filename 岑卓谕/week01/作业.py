import numpy as np
import jieba # 中文分词用途
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer # 词频统计
from sklearn.neighbors import KNeighborsClassifier # KNN
from openai import OpenAI

print("begin")

df=pd.read_csv("dataset.csv",sep="\t",header=None)
X=df[0].apply(lambda x:' '.join(jieba.lcut(x)))

word_vec=CountVectorizer()
word_vec.fit(X.values)
X=word_vec.transform(X.values)
y=df[1].values

knn=KNeighborsClassifier()
knn.fit(X,y)
print(knn)
print()
x_str="帮我播放一下郭德纲的小品"
x=' '.join(jieba.lcut(x_str))
print(x)
x=word_vec.transform([x])
y=knn.predict(x)
print(y)

word_set=set(df[1].values)
print(f"类别：{word_set}")

llm_client=OpenAI(
    api_key="sk-899b7455a4074832b92b7ac81b6aa6e2",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)
llm_completion=llm_client.chat.completions.create(
    model="qwen-flash",  # 模型的代号

    # 对话列表
    messages=[
        {"role": "system", "content": "你是一个AI助手."},  # 给大模型的命令，角色的定义
        {"role": "user", "content": f"""
        帮我进行文本分类，其中输入的文本是:{x_str}
        你需要从下边类别中选择一个类别：
        {word_set}
        注意你给我的回答内容只能是某一种类别
        """},  # 用户的提问

    ]
)
print(llm_completion.choices[0].message.content)
