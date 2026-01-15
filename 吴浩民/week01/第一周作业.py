import jieba
import pandas as pd
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from openai import OpenAI

# 数据加载(读取csv文件)
data = pd.read_csv('dataset.csv', sep="\t", names=["text", "label"], nrows=None)

print(f"数据集大小: {len(data)}")
print("数据预览:")
print(data.head(10))

# 数据预处理(中文分词: 将连续中文文本切分成词语, 并用空格连接供CountVectorizer进行词频统计)
data['cut_text'] = data['text'].apply(lambda x: " ".join(jieba.lcut(x)))

# 特征提取(文本向量化, 将文本转换为数值特征向量, 创建词袋模型)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['cut_text'])
y = data['label']

# 模型训练(KNN)
knn_model = KNeighborsClassifier()
knn_model.fit(X, y)

# 大模型客户端
client = OpenAI(
    # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx",
    # https://bailian.console.aliyun.com/?tab=model#/api-key
    api_key="sk-92aa865c659e484a881281f153f521f2",

    # 大模型厂商的地址，阿里云
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


def text_classify_with_knn(text: str) -> str:
    """
    文本分类（KNN）
    """
    # 对文本进行相同预处理(分词)
    cut_text = " ".join(jieba.lcut(text))

    # 特征转换(使用相同的向量化器)
    text_vector = vectorizer.transform([cut_text])

    # 模型预测(KNN)
    prediction = knn_model.predict(text_vector)[0]

    return prediction


def text_classify_with_llm(text: str) -> str:
    """
    文本分类（LLM）
    """
    completion = client.chat.completions.create(
        model="qwen-flash",  # 模型的代号

        messages=[
            {"role": "user", "content": f"""帮我进行文本分类：{text}

            输出的类别只能从如下中进行选择， 除了类别之外下列的类别，请给出最合适的类别。
            FilmTele-Play            
            Video-Play               
            Music-Play              
            Radio-Listen           
            Alarm-Update        
            Travel-Query        
            HomeAppliance-Control  
            Weather-Query          
            Calendar-Query      
            TVProgram-Play      
            Audio-Play       
            Other             
            """},  # 用户的提问
        ]
    )
    return completion.choices[0].message.content


if __name__ == "__main__":
    # 测试文本
    test_texts = [
        "帮我播放周杰伦的歌曲"
    ]

    for i, text in enumerate(test_texts, 1):
        resultKNN = text_classify_with_knn(text)
        print("KNN模型测试结果:" + resultKNN)
        resultLLM = text_classify_with_llm(text)
        print("LLM测试结果:" + resultLLM)
