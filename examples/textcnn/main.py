#!/usr/bin/env python3
import torch
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# 超参数
BATCH_SIZE = 64
EMBED_DIM = 100
HIDDEN_DIM = 32
EPOCHS = 5

# 1. 加载数据集并预处理
tokenizer = get_tokenizer('basic_english')  # 基础英文分词器

# 创建词汇表
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

# 加载训练数据集并构建词汇表
train_iter = IMDB(split='train')
vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])  # 设置未知词标记

# 2. 定义数据处理管道
text_pipeline = lambda x: vocab(tokenizer(x))
label_pipeline = lambda x: 1 if x == 'pos' else 0

# 3. 创建批处理函数
def collate_batch(batch):
    text_list, label_list = [], []
    for (_label, _text) in batch:
        processed_text = torch.tensor(text_pipeline(_text), dtype=torch.int64)
        text_list.append(processed_text)
        label_list.append(label_pipeline(_label))
    
    # 对文本进行padding
    padded_text = pad_sequence(text_list, batch_first=True, padding_value=vocab["<pad>"])
    return padded_text, torch.tensor(label_list)

# 4. 创建数据加载器
train_iter, test_iter = IMDB()  # 获取训练集和测试集
train_loader = DataLoader(list(train_iter), batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_batch)
test_loader = DataLoader(list(test_iter), batch_size=BATCH_SIZE, collate_fn=collate_batch)

# 5. 定义简单模型
class TextClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_class):
        super().__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim)
        self.rnn = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = torch.nn.Linear(hidden_dim, num_class)
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.rnn(embedded)
        return self.fc(hidden.squeeze(0))

model = TextClassifier(len(vocab), EMBED_DIM, HIDDEN_DIM, 2)

# 6. 定义训练函数
def train():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(EPOCHS):
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch: {epoch+1}, Loss: {loss.item():.4f}')

# 7. 训练模型
train()

# 8. 测试函数
def test():
    total, correct = 0, 0
    with torch.no_grad():
        for texts, labels in test_loader:
            outputs = model(texts)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# 运行测试
test()
