import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class CIFAR10DataLoader:
    def __init__(self, data_path, batch_size=64, transform=None):
        self.data_path = data_path
        self.batch_size = batch_size
        self.transform = transform

    def get_data_loaders(self):
        # 确保数据转换正确
        trainset = torchvision.datasets.CIFAR10(root=self.data_path, train=True, download=True, transform=self.transform)
        testset = torchvision.datasets.CIFAR10(root=self.data_path, train=False, download=True, transform=self.transform)
        
        trainloader = DataLoader(trainset, batch_size=self.batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=self.batch_size, shuffle=False)
        
        return trainloader, testloader

class Trainer:
    def __init__(self, model, trainloader, testloader, criterion, optimizer, device):
        self.model = model
        self.trainloader = trainloader
        self.testloader = testloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=5):
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(self.trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:
                    print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

    def test(self):
        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.testloader:
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')

def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data')

    # 检查是否有可用的 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置数据预处理（转化为张量并标准化）
    transform = transforms.Compose([
        transforms.ToTensor(),  # 将 PIL 图像转换为张量
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 标准化
    ])
    
    # 加载数据
    data_loader = CIFAR10DataLoader(data_path, transform=transform)
    trainloader, testloader = data_loader.get_data_loaders()

    # 初始化模型、损失函数和优化器
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # 初始化训练器并训练
    trainer = Trainer(model, trainloader, testloader, criterion, optimizer, device)
    trainer.train(num_epochs=5)
    trainer.test()

if __name__ == "__main__":
    main()

