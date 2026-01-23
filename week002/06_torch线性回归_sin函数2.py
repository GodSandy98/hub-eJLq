import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
x = np.linspace(-2 * np.pi, 2 * np.pi, 1000)
y = np.sin(x) + 0.1 * np.random.randn(1000)

# 转换为PyTorch张量
x_tensor = torch.FloatTensor(x.reshape(-1, 1))
y_tensor = torch.FloatTensor(y.reshape(-1, 1))


# 多层网络模型
class SinNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features=1, out_features=64),
            nn.Sigmoid(),
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=16),
            nn.Sigmoid(),
            nn.Linear(in_features=16, out_features=8),
            nn.ReLU(),
            nn.Linear(in_features=8, out_features=1)
        )

    def forward(self, x):
        return self.net(x)


# 训练
model = SinNetwork()
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01) # 优化器，基于 a b 梯度 自动更新

losses = []
for epoch in range(1000):
    optimizer.zero_grad()
    output = model(x_tensor)
    loss = loss_fn(output, y_tensor)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

# 预测
with torch.no_grad():
    y_pred = model(x_tensor).numpy()

# 可视化
plt.figure(figsize=(12, 4))
plt.subplot(131)
plt.scatter(x, y, s=1, alpha=0.3, label='Noisy Data')
plt.plot(x, np.sin(x), 'g-', lw=2, label='True sin')
plt.plot(x, y_pred, 'r-', lw=2, label='Predicted')
plt.legend()
plt.title('Fitting Result')

plt.subplot(132)
plt.plot(losses)
plt.yscale('log')
plt.title('Training Loss')
plt.xlabel('Epoch')

plt.subplot(133)
plt.hist(y_pred.flatten() - np.sin(x), bins=30)
plt.title('Error Distribution')

plt.tight_layout()
plt.show()
