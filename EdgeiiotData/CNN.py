import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
loaded_data = torch.load('data.pth')

i = 0
while True:
    dir = "./run/Version_" + str(i)
    if not os.path.exists(dir):
        break
    i += 1
TensorWriter = SummaryWriter(dir)

# 通过键来访问加载的数据集
X_train = loaded_data['X_train']
X_test = loaded_data['X_test']
y_train = loaded_data['y_train']
y_test = loaded_data['y_test']



# 首先，我们需要将 numpy 数组转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)  # 可能需要在此步骤将独热编码标签转化为类别
y_test = torch.tensor(y_test, dtype=torch.float32)

# 然后，我们将张量包装为 TensorDataset 对象
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 最后，我们创建 DataLoader 对象
batch_size = 64  # 你可以选择适合你的硬件和数据的批量大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_class = 15

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv1d(96, 64, 3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, 3, padding=1)
        self.fc1 = nn.Linear(128 * 1, 256)  # 注意这里也进行了相应的修改
        self.fc2 = nn.Linear(256, 15)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 除去批处理维度的其他所有维度
        num_features = 1
        for s in size:
            num_features *= s
        return num_features



net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
num_epochs = 10
for epoch in trange(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = net(images)
        labels = torch.argmax(labels, dim=1)  # Convert from one-hot to class indices
        loss = criterion(outputs, labels)


        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        net.eval()
        TensorWriter.add_scalars('Loss',{'train': loss.item()},epoch)
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == torch.argmax(labels, dim=1)).sum().item()
        TensorWriter.add_scalars('Acc',{'Acc': 100 * correct / total},epoch)
# Testing


