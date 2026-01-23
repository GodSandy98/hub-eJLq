"""
调整 06_torch线性回归.py 构建一个sin函数，然后通过多层网络拟合sin函数，并进行可视化。
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn


X_numpy = np.random.rand(100, 1) * 10
y_numpy = np.sin(X_numpy) + np.random.randn(100, 1) * 0.1
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


class MultiLayerModel(nn.Module):
  def __init__(self, input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim):
    super(MultiLayerModel, self).__init__()
    self.network = nn.Sequential(
      nn.Linear(input_dim, hidden_dim1),
      nn.Sigmoid(),
      nn.Linear(hidden_dim1, hidden_dim2),
      nn.Sigmoid(),
      nn.Linear(hidden_dim2, hidden_dim3),
      nn.Sigmoid(),
      nn.Linear(hidden_dim3, output_dim),
    )

  def forward(self, x):
    return self.network(x)


loss_fn = torch.nn.MSELoss()


num_epochs = 1000
input_dim = 1
hidden_dim1 = 30
hidden_dim2 = 30
hidden_dim3 = 30
hidden_dim4 = 30
output_dim = 1

model = MultiLayerModel(input_dim, hidden_dim1, hidden_dim2, hidden_dim3, hidden_dim4, output_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer often works better

loss_list = []
for epoch in range(num_epochs):
  model.train()
  optimizer.zero_grad()

  y_pred = model(X)
  loss = loss_fn(y_pred, y)
  loss.backward()
  optimizer.step()
  loss_list.append(loss.item())

  if (epoch + 1) % 100 == 0:
      print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')





with torch.no_grad():
    model.eval()
    y_predicted = model(X)

sorted_indices = np.argsort(X_numpy.flatten())
X_sorted = X_numpy[sorted_indices]
y_sorted = y_numpy[sorted_indices]
y_pred_sorted = y_predicted[sorted_indices]


plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_sorted, y_pred_sorted, label='Multi-layer Model Prediction', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.title('Multi-layer Neural Network Fitting sin(x)')
plt.legend()
plt.grid(True)
plt.show()
