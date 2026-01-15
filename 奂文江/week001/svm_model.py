import pandas as pds
import jieba
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC

dataset_str = pds.read_csv("dataset.csv", sep="\t", header=None)

input_sentence = dataset_str[0].apply(lambda x: " ".join(jieba.lcut(x)))

countVectorizer = CountVectorizer()
countVectorizer.fit(input_sentence)
input_feature = countVectorizer.transform(input_sentence.values)

#支持向量机（SVM）
model = LinearSVC(
    C=1.0,  # 正则化参数
    penalty='l2',
    loss='squared_hinge',
    max_iter=1000
)

model.fit(input_feature, dataset_str[1].values)

text = "中国之声广播电台"

if __name__ == "__main__":
    text_sentence = " ".join(jieba.lcut(text))
    text_feature = countVectorizer.transform([text_sentence])
    result = model.predict(text_feature)[0]
    print(f"LinearSvc模型预测结果为：{result}")
