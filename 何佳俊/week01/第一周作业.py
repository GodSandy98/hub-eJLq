import jieba
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

dataset = pd.read_csv("dataset.csv", sep="\t", header=None, nrows=10000)

# 提取 文本的特征 tfidf， dataset[0]
# 构建一个模型 knn， 学习 提取的特征和 标签 dataset[1] 的关系
# 预测，用户输入的一个文本，进行预测结果
input_sententce = dataset[0].apply(lambda x: " ".join(jieba.lcut(x))) # sklearn对中文处理

vector = CountVectorizer() # 对文本进行提取特征 默认是使用标点符号分词
vector.fit(input_sententce.values)
input_feature = vector.transform(input_sententce.values)

model = KNeighborsClassifier()
model.fit(input_feature, dataset[1].values)

#第一种，机器学习，使用jieba分词
def test_classify_using_ml(text: str) -> str:
    test_sentence = " ".join(jieba.lcut(text))
    test_feature = vector.transform([test_sentence])
    return model.predict(test_feature)[0]

#第二种，大模型
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-9f96f86d7xxxxx78d33859df2", # 账号绑定的

    # 大模型厂商的地址
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
def test_calssifu_using_llm(text: str) -> str:
    completion = client.chat.completions.create(
        # 模型列表：https://help.aliyun.com/zh/model-studio/getting-started/models
        # model="qwen-plus",  # 模型的代号
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "system", "content": "You are a helpful assistant."},  # 给大模型的命令，角色的定义
            {"role": "user", "content": f"帮我进行文本分类:{text}。输出的类别只能从如下分类选择：{dataset[1].value_counts()}"},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content

if __name__ == "__main__":
    print("机器学习：",test_classify_using_ml("帮我导航到天安门。"))
    print("大模型：",test_calssifu_using_llm("帮我导航到天安门。"))
