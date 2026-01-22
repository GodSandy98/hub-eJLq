import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("./dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        """
        通用分类器模型
        :param input_dim: 输入维度（词汇表大小）
        :param hidden_dims: 隐藏层维度列表，例如 [128] 表示1层隐藏层，128个节点；[256, 128] 表示2层隐藏层
        :param output_dim: 输出维度（类别数）
        """

        super(Classifier, self).__init__()
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())

        # 隐藏层之间的连接
        for i in range(1, len(hidden_dims)):
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i]))
            layers.append(nn.ReLU())

        # 最后一个隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        # 将所有层组合成Sequential
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

def train_model(hidden_dims, num_epochs=10, lr=0.01):
    """
    训练指定结构的模型并返回loss记录
    :param hidden_dims: 隐藏层维度列表
    :param num_epochs: 训练轮数
    :param lr: 学习率
    :return: 每轮的平均loss列表
    """
    # 初始化模型
    output_dim = len(label_to_index)
    model = Classifier(vocab_size, hidden_dims, output_dim)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

# epoch： 将数据集整体迭代训练一次
# batch： 数据集汇总为一批训练一次
    epoch_losses = []

    for epoch in range(num_epochs): # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        epoch_losses.append(avg_loss)

        model_name = f"隐藏层{len(hidden_dims)}层_节点{'-'.join(map(str, hidden_dims))}"
        print(f"{model_name} - Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")

    return epoch_losses

# 定义要测试的模型结构组合
experiment_configs = {
    "1层_128节点": [128],  # 原模型结构
    "1层_256节点": [256],
    "2层_256-128节点": [256, 128],
    "3层_512-256-128节点": [512, 256, 128]
}

# 训练所有模型并记录loss
results = {}
num_epochs = 10  # 统一训练10轮
print("开始训练不同结构的模型...")
for config_name, hidden_dims in experiment_configs.items():
    print(f"\n========== 训练 {config_name} ==========")
    results[config_name] = train_model(hidden_dims, num_epochs=num_epochs)

def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    tokenized += [0] * (max_len - len(tokenized))

    bow_vector = torch.zeros(vocab_size)
    for index in tokenized:
        if index != 0:
            bow_vector[index] += 1

    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(bow_vector)

    predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

print("\n========== 各模型最终Loss变化 ==========")
final_losses = {name: losses[-1] for name, losses in results.items()}
# 按最终loss从小到大排序
sorted_losses = sorted(final_losses.items(), key=lambda x: x[1])
for i, (name, loss) in enumerate(sorted_losses):
    print(f"第{i + 1}名: {name} - 最终Loss: {loss:.4f}")
