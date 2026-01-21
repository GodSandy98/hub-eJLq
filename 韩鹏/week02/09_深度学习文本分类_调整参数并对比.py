import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 读取数据集
dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
# 第一列输入数据列表
texts = dataset[0].tolist()
# 第二列标签向量列表
string_labels = dataset[1].tolist()
# 标签-索引字典：{'Alarm-Update': 8, 'Audio-Play': 9, 'Calendar-Query': 0, 'FilmTele-Play': 2, 'HomeAppliance-Control': 6, 'Music-Play': 4, 'Other': 1, 'Radio-Listen': 5, 'TVProgram-Play': 11, 'Travel-Query': 3, 'Video-Play': 10, 'Weather-Query': 7}
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
# 标签向量列表转标签索引列表
numerical_labels = [label_to_index[label] for label in string_labels]
# 输出维度
output_dim = len(label_to_index)
# 字符-索引字典
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
# 索引-字符字典
index_to_char = {i: char for char, i in char_to_index.items()}
# 字符-索引数量
vocab_size = len(char_to_index)
# 截取的长度
max_len = 40


# 自定义数据集（custom dataset）
class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        # 传入需要建模的文本
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        # 取长补短
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        # 词频编码
        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        # 获取数据集大小
        return len(self.texts)

    def __getitem__(self, idx):
        # 获取单个样本 x[1]会调用这个方法
        return self.bow_vectors[idx], self.labels[idx]

# 自定义神经网络模型
class SimpleClassifier(nn.Module):
    # 调整隐藏层为可自定义层数
    def __init__(self, input_dim, hidden_dims, output_dim): # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        # 层列表
        layers = []
        # 添加输入层
        prev_dim = input_dim
        # 循环添加隐藏层
        for i in range(len(hidden_dims)):
            layers.append(nn.Linear(prev_dim, hidden_dims[i])) # 添加全连接层
            layers.append(nn.ReLU()) # 添加ReLU激活函数层
            prev_dim = hidden_dims[i]
        # 添加输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        # 解包layers,并使用Sequential容器包装
        self.network = nn.Sequential(*layers)

    # 模型计算入口，定义了向前计算过程
    def forward(self, x):
        return self.network(x)

# torch dataset是读取单个样本
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
# 封装，dataloader将多个样本拼接为batch
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

# 调整不同层数
model_configs = [
    {
        "name": "2层128节点",
        "hidden_dims": [128, 128],
        "color": "green",
        "lr": 0.01
    },
    {
        "name": "1层128节点",
        "hidden_dims": [128],
        "color": "blue",
        "lr": 0.01
    },
    {
        "name": "3层128节点",
        "hidden_dims": [128, 128, 128],
        "color": "red",
        "lr": 0.01
    },
    {
        "name": "2层128和64节点",
        "hidden_dims": [128, 64],
        "color": "brown",
        "lr": 0.01
    },
    {
        "name": "2层64和128节点",
        "hidden_dims": [64, 128],
        "color": "yellow",
        "lr": 0.01
    },
]

def train_model(config):
    # 创建模型
    model = SimpleClassifier(vocab_size, config["hidden_dims"], output_dim)
    # 损失函数（交叉熵损失）
    criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
    # 优化器
    optimizer = optim.SGD(model.parameters(), lr=config["lr"])

    # 训练:双重for循环
    num_epochs = 10  # 将数据集整体迭代次数
    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0

        # batch： 数据集汇总为一批训练一次
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % 50 == 0:
                # print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        # 打印平均损失
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    return model

# 预测方法
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    # 将文本的前max_len个字符转换为索引
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    # 不够长的补0
    tokenized += [0] * (max_len - len(tokenized))
    # 创建一个长度为vocab_size的全0的张量
    bow_vector = torch.zeros(vocab_size)
    # 统计每个索引出现的次数
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1
    # 升维，将一维的张量变为二维，因为模型要求输入的维度为2，形状为（1，bow_vector）
    bow_vector = bow_vector.unsqueeze(0)
    # 设置为评估模式
    model.eval()
    # 创建上下文管理器，不会进行梯度计算
    with torch.no_grad():
        # 将bow_vector传入模型，得到预测结果，预测结果output的形状是（1，标签数）。output: tensor([[-0.4141, -1.0493,  0.1880,  0.6482, -1.2357,  0.6100,  0.0455,  1.2456,  0.0097,  1.4302, -0.7334,  0.1012]])
        output = model(bow_vector)
    # 获取最大值和对应的索引。max_value: tensor([-1.4302])  max_index: tensor([9])
    max_value, max_index = torch.max(output, 1)
    # 将张量索引转换为普通整数
    predicted_index = max_index.item()
    # 索引映射到字典中对应的标签
    predicted_label = index_to_label[predicted_index]
    # 返回预测的标签
    return predicted_label

# 索引-标签字典：{0: 'Calendar-Query', 1: 'Other', 2: 'FilmTele-Play', 3: 'Travel-Query', 4: 'Music-Play', 5: 'Radio-Listen', 6: 'HomeAppliance-Control', 7: 'Weather-Query', 8: 'Alarm-Update', 9: 'Audio-Play', 10: 'Video-Play', 11: 'TVProgram-Play'}
index_to_label = {i: label for label, i in label_to_index.items()}
new_text = "帮我导航到北京"
new_text_2 = "查询明天北京的天气"

# 分别训练模型并预测结果
for config in model_configs:
    model = train_model(config)

    print(f"使用模型: {config['name']}")

    predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

    predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
    print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
