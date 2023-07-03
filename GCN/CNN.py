import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from tqdm import trange
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
i = 0
while True:
    dir = "./run/CNN/Version_" + str(i)
    if not os.path.exists(dir):
        break
    i += 1
TensorWriter = SummaryWriter(dir)

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", default=0.001)
parser.add_argument("--num_epochs", type=int, default=10)
args = parser.parse_args()

# ------------------------------- 加载数据并分离特征和标签 ------------------------------- #
train_data = pd.read_csv('./data/mitbih_train.csv', header=None)
test_data = pd.read_csv('./data/mitbih_test.csv', header=None)
train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.long)
train_features = train_data.iloc[:, :-1].values

test_labels = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.long)
test_features = test_data.iloc[:, :-1].values


# ---------------------------------- 自定义数据集类 --------------------------------- #
class ECGDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        feature = torch.tensor(self.features[idx], dtype=torch.float).unsqueeze(0)
        label = self.labels[idx]
        return feature, label


# ---------------------------------- 定义CNN模型 --------------------------------- #
class CNN(nn.Module):
    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 43, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# ---------------------------------- 创建数据加载器 --------------------------------- #
train_dataset = ECGDataset(train_features, train_labels)
test_dataset = ECGDataset(test_features, test_labels)
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True)
num_classes = 5
# ----------------------------------- 设备设置 ----------------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------- 创建模型实例 ---------------------------------- #
model = CNN(num_classes).to(device)

# -------------------------------- 定义损失函数和优化器 -------------------------------- #
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

# ----------------------------------- 训练模型 ----------------------------------- #
for epoch in trange(args.num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    for features, labels in train_loader:
        features = features.to(device)
        labels = labels.to(device)
        
        # ----------------------------------- 前向传播 ----------------------------------- #
        outputs = model(features)
        loss = criterion(outputs, labels)

        # ---------------------------------- 反向传播和优化 --------------------------------- #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * features.size(0)
        _, predicted = torch.max(outputs.data, 1)
        train_correct += (predicted == labels).sum().item()

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100.0 * train_correct / len(train_loader.dataset)

   # --------------------------------- 在测试集上验证模型 -------------------------------- #
    model.eval()
    test_correct = 0
    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            test_correct += (predicted == labels).sum().item()

    test_accuracy = 100.0 * test_correct / len(test_loader.dataset)
    TensorWriter.add_scalars('Acc',{'train': train_accuracy, 'test':test_accuracy},epoch)
    TensorWriter.add_scalars('Loss',{'loss': train_loss},epoch)

