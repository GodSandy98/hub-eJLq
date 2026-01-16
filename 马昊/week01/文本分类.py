import pandas as pd
# 作中文分词
import jieba
# countvector可以将文本数据转换为特征矩阵
from sklearn.feature_extraction.text import CountVectorizer as Cv
# 导入knn，knn创建模型，训练模型，并预测结果
from sklearn.neighbors import KNeighborsClassifier as Knn
from openai import OpenAI


# filepath_or_buffer表示文件路径或缓冲区，可以省略，直接写位置。
# sep表示用什么作为分隔符，names自行命名列，header以某行作为列名，nrows指定读取的数据行数
# 如果想读取指定行的数据，可以加一个skiprows=x，表示跳过前面x行，nrows只指定数据行
data = pd.read_csv(filepath_or_buffer="dataset.csv",sep='\t',names=['text','label'],nrows=None)
# 提取标签列所有唯一值
unique_labels = data['label'].unique()
unique_labels_list = list(unique_labels)
unique_labels_list1 = '\t'.join(unique_labels_list)
print(f'所有标签类型为：{unique_labels_list1}')

# 对text列每一个元素应用括号内的匿名函数操作 'x'.join表示用x分隔开每个中文字符，并将他们加在同一列表中
input_sentence = data['text'].apply(lambda x:' '.join(jieba.lcut(x)))

# 创建词频向量器实例，后续用它来处理文本
vector = Cv()
# fit和transform可以合并成一个命令，代码更加简洁
# vector.fit(input_sentence.values)
# 构建词汇表并转换成特征矩阵
input_feature = vector.fit_transform(input_sentence.values)


# 创建了一个客户端实例，用于调用openai的api
client = OpenAI(

    api_key='2108bf46-84ae-45f7-8975-51272d72cdd6',
    # 厂商地址须使用厂商提供的地址
    base_url='https://ark.cn-beijing.volces.com/api/v3'
)

def textclassify_using_ml(text:str) -> str:
    '''
    文本分类：使用机器学习
    '''
    # 创建机器学习模型
    model = Knn()
    # 第一个参数是传入的文本，第二个参数是需要识别的标签
    model.fit(input_feature, data.label)
    # 将测试文本用jieba分词
    text_sentence = ' '.join(jieba.lcut(text))
    # 将分词好后的句子转换为特征矩阵
    text_feature = vector.transform([text_sentence])
    # 预测结果并赋值给result
    result = model.predict(text_feature)[0]
    # 返回预测结果
    return result


def textclassify_using_llm(text:str) -> str:
    '''
    文本分类：使用大语言模型
    '''
    completion = client.chat.completions.create(
        # 模型名称须准确填写
        model = 'doubao-seed-1-6-lite-251015',
        # 使用一个prompt使大模型完成指定任务
        messages = [
            {'role':'system','content':f'''帮我对这句话进行文本分类:{text}
            类别在以下类别中选择：
            Travel-Query 
            Music-Play 
            FilmTele-Play 
            Video-Play 
            Radio-Listen 
            HomeAppliance-Control 
            Weather-Query 
            Alarm-Update 
            Calendar-Query 
            TVProgram-Play 
            Audio-Play 
            Other，不要输出其他内容'''},
        ]
    )
    return  completion.choices[0].message.content


if __name__ == '__main__':
    # print(data.head(10))这行代码可以打印数据集的前十行数据
    # print('数据集的样本维度：',data.shape)这行代码可以打印数据集的样本维度
    # print(data['label'].value_counts())这行代码可以统计标签出现的次数
    print('机器学习',textclassify_using_ml('帮我播放热门的原神游戏视频'))
    print('大语言模型',textclassify_using_llm('帮我播放热门的原神游戏视频'))
