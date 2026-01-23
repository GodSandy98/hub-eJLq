"""
调整 09_深度学习文本分类.py 代码中模型的层数和节点个数，对比模型的loss变化。
"""
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np


def load_data():
   dataset = pd.read_csv("../../data/dataset.csv", sep="\t", header=None)
   return dataset


def preprocess_data(dataset):
   texts = dataset[0].tolist()
   string_labels = dataset[1].tolist()
   label_to_index = {label: i for i, label in enumerate(set(string_labels))}
   numerical_labels = [label_to_index[label] for label in string_labels]
   return texts, string_labels, label_to_index, numerical_labels



def create_char_to_index(texts):
   char_to_index = {'<pad>': 0}
   for text in texts:
      for char in text:
         if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
   index_to_char = {i: char for char, i in char_to_index.items()}
   return char_to_index, index_to_char



dataset = load_data()
texts, string_labels, label_to_index, numerical_labels = preprocess_data(dataset)
char_to_index, index_to_char = create_char_to_index(texts)
index_to_label = {i: label for label, i in label_to_index.items()}
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


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers): # 层的个数
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.num_layers = num_layers


    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        for _ in range(self.num_layers - 1):
            out = self.relu(out)
        out = self.fc2(out)
        return out


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

output_dim = len(label_to_index)
criterion = nn.CrossEntropyLoss()


def train_model(vocab_size, output_dim, criterion, dataloader, num_epochs):
    results = []
    best_model = None
    best_loss = float('inf')

    for hidden_dim in [64, 128, 256]:
        for num_layers in [3, 5, 7]:
            print(f"Training with hidden_dim: {hidden_dim}, num_layers: {num_layers}")
            model = SimpleClassifier(vocab_size, hidden_dim, output_dim, num_layers)
            optimizer = optim.SGD(model.parameters(), lr=0.01)

            config_losses = []

            for epoch in range(num_epochs):
                model.train()
                running_loss = 0.0
                for idx, (inputs, labels) in enumerate(dataloader):
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item()

                epoch_loss = running_loss / len(dataloader)
                config_losses.append(epoch_loss)

            final_loss = config_losses[-1]
            results.append((hidden_dim, num_layers, final_loss, config_losses))
            print(f"Final training loss: {final_loss:.4f}")

            if final_loss < best_loss:
                best_loss = final_loss
                best_model = model

    return results, best_model



num_epochs = 10
results, trained_model = train_model(vocab_size, output_dim, criterion, dataloader, num_epochs)


def plot_loss_results(results):
    print("\n" + "="*50)
    print("Summary of Results:")
    print("="*50)
    print(f"{'Hidden Dim':<12} {'Num Layers':<12} {'Final Loss':<12}")
    print("-"*50)
    for r in sorted(results, key=lambda x: (x[0], x[1])):
        print(f"{r[0]:<12} {r[1]:<12} {r[2]:<12.4f}")


plot_loss_results(results)

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

    print(f"output: {output}")

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    print(f"predicted_index: {predicted_index}")
    predicted_label = index_to_label[predicted_index]
    print(f"predicted_label: {predicted_label}")
    return predicted_label


test_text = "帮我放一首歌曲"
predicted_label = classify_text(test_text, trained_model, char_to_index, vocab_size, max_len, index_to_label)
print(f"test_text: {test_text}, predicted_label: {predicted_label}")