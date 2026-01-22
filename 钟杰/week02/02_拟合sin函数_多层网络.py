import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 1. 生成sin函数数据
X_numpy = np.random.uniform(-3*np.pi, 3*np.pi, (10000, 1))
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + 0.01*np.random.randn(10000, 1)

# 转换为PyTorch张量
X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")

#神经网络通过激活函数来学习函数，每个神经元都会有对应的权重和偏置。但无法像线性回归那样得到a/b模型的值。


# 2. 定义多层神经网络模型
class SinNet(nn.Module):
    def __init__(self, hidden_layers=[64, 128, 64]):
        """
        多层神经网络拟合sin函数
        hidden_layers: 列表，定义每层隐藏层的节点数
        """
        super(SinNet, self).__init__()

        # 创建网络层
        self.layers = nn.ModuleList()

        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(1, hidden_layers[0]))
        self.layers.append(nn.Tanh())  # 使用Tanh激活函数，适合sin函数

        # 中间隐藏层
        for i in range(1, len(hidden_layers)):
            self.layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            self.layers.append(nn.Tanh())

        # 输出层（回归问题，通常不用激活函数）
        self.output = nn.Linear(hidden_layers[-1], 1)

    def forward(self, x):
        # 前向传播
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):  # 参数：输入维度、隐藏层维度、输出维度
        # 调用父类 nn.Module 的初始化方法
        super(SimpleClassifier, self).__init__()

        # 创建多个隐藏层
        self.layers = nn.ModuleList()
        # 定义第一个全连接层：从输入维度到隐藏层维度
        #self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        # 定义 ReLU 激活函数
        #self.relu = nn.ReLU()

        # 输入层到第一个隐藏层
        self.layers.append(nn.Linear(input_dim, hidden_dims[0]))
        self.layers.append(nn.ReLU())
        # 定义第二个全连接层：从隐藏层维度到隐藏层
        for i in range(1,len(hidden_dims)):
            self.layers.append(nn.Linear(hidden_dims[i-1], hidden_dims[i]))    #0~1

            self.layers.append(nn.Tanh())

            #self.layers.append(nn.ReLU())
            #self.layers.append(nn.Dropout(0.3))
        # 定义第n个全连接层：从隐藏层到输出维度
        #self.fc2 = nn.Linear(hidden_dims[-1], output_dim)
        # 输出层
        self.output = nn.Linear(hidden_dims[-1], output_dim)
    '''
        def forward(self, x):
        # 定义前向传播过程：依次通过全连接层1、激活函数、全连接层2
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    '''

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.output(x)
        return x


# 3. 创建模型实例
#model = SinNet(hidden_layers=[64, 128, 64])
model = SimpleClassifier(1,hidden_dims=[4, 4],output_dim=1)
# 打印模型信息
print("模型架构:")
print(model)
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
print("---" * 10)

# 4. 定义损失函数和优化器
loss_fn = nn.MSELoss()  # 回归任务使用均方误差
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器

# 学习率调度器（可选，可以改善训练效果）
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.8)

# 5. 训练模型
num_epochs = 2000
loss_history = []  # 记录损失变化
best_loss = float('inf')  # 记录最佳损失
best_model_state = None  # 保存最佳模型状态

print("开始训练...")
for epoch in range(num_epochs):
    # 前向传播
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)
    loss_history.append(loss.item())

    # 保存最佳模型
    if loss.item() < best_loss:
        best_loss = loss.item()
        best_model_state = model.state_dict().copy()

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    scheduler.step()  # 更新学习率

    # 每200个epoch打印一次损失
    if (epoch + 1) % 200 == 0:
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch + 1:4d}/{num_epochs}], Loss: {loss.item():.6f}, LR: {current_lr:.6f}')

# 加载最佳模型
model.load_state_dict(best_model_state)

print("\n训练完成！")
print(f"最终损失: {loss_history[-1]:.6f}")
print(f"最佳损失: {best_loss:.6f}")
print("---" * 10)

# 6. 模型评估和可视化
model.eval()  # 切换到评估模式

# 生成更密集的测试数据用于绘制平滑曲线
X_test_numpy = np.linspace(-3 * np.pi, 3 * np.pi, 500).reshape(-1, 1)
X_test = torch.from_numpy(X_test_numpy).float()

# 计算预测结果
with torch.no_grad():
    y_test_pred = model(X_test).numpy()
    y_train_pred = model(X).numpy()

# 计算真实sin函数值（无噪声）
y_true = np.sin(X_test_numpy)

# 7. 创建高质量可视化图形
fig = plt.figure(figsize=(18, 12))

# 子图1：主要对比图
ax1 = plt.subplot(2, 3, 1)
# 绘制真实sin函数（无噪声）
ax1.plot(X_test_numpy, y_true, 'g-', linewidth=3, alpha=0.8, label='真实sin函数')

# 绘制训练数据点（带噪声）
ax1.scatter(X_numpy, y_numpy, color='blue', alpha=0.6, s=20, label='训练数据（带噪声）')

# 绘制模型预测曲线
ax1.plot(X_test_numpy, y_test_pred, 'r-', linewidth=2.5, label='神经网络预测')

plt.tight_layout()
plt.show()
