import torch
import torch.nn as nn
import torch.optim as optim

# 检查是否有 GPU 可用，选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建一个简单的线性回归模型
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # 输入一个特征，输出一个值

    def forward(self, x):
        return self.linear(x)

# 生成一些假数据：y = 2x + 1
# x 是输入，y 是目标输出
x_train = torch.randn(100, 1)  # 100 个样本，1 个特征
print(x_train)
y_train = 2 * x_train + 1  # 线性关系：y = 2x + 1

# 将数据移到 GPU（如果可用）
x_train = x_train.to(device)
y_train = y_train.to(device)

# 初始化模型，损失函数和优化器
model = LinearRegressionModel().to(device)  # 将模型加载到 GPU
criterion = nn.MSELoss()  # 均方误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.01)  # 随机梯度下降优化器

# 训练模型
epochs = 200
for epoch in range(epochs):
    model.train()  # 设置模型为训练模式

    # 前向传播
    outputs = model(x_train)
    loss = criterion(outputs, y_train)  # 计算损失

    # 后向传播和优化
    optimizer.zero_grad()  # 清空梯度
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新模型参数

    if (epoch + 1) % 10 == 0:  # 每10个epoch输出一次
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

# 打印最终的训练损失
print(f"Final loss after training: {loss.item():.4f}")

# 测试：预测一些值
with torch.no_grad():  # 测试时不需要计算梯度
    predicted = model(x_train)
    print(f"Predicted values: {predicted[:5]}")  # 打印前5个预测值
