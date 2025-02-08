#!/usr/bin/env python
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import argparse

# 定义超参数
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=64, help='Input batch size for training')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum for SGD optimizer')
parser.add_argument('--model_path', type=str, default='simple_cnn.pth', help='Path to load/save the model')
args = parser.parse_args()

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, 2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, 2)
        x = x.view(-1, 32 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def load_data(data_path, batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
    testset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
    return trainloader, testloader

def train(model, trainloader, criterion, optimizer, device, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss: {running_loss / 100:.3f}")
                running_loss = 0.0

def test(model, testloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy of the network on the 10000 test images: {100 * correct / total:.2f}%')


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(current_dir, 'data')
    model_path = os.path.join(current_dir, args.model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    trainloader, testloader = load_data(data_path, args.batch_size)

    model = SimpleCNN().to(device)
    
    # 检查是否存在预训练模型
    if os.path.exists(model_path):
        print(f"Loading pre-trained model from {model_path}")
        model.load_state_dict(torch.load(model_path, weights_only=True))
        # 直接测试加载的模型
        test(model, testloader, device)
    else:
        print("No pre-trained model found. Training from scratch...")
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum)

        train(model, trainloader, criterion, optimizer, device, args.num_epochs)
        test(model, testloader, device)

        # 保存模型
        print(f"Saving model to {model_path}")
        torch.save(model.state_dict(), model_path)

if __name__ == "__main__":
    main()
