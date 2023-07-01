import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from tqdm import trange
from torch.utils.tensorboard import SummaryWriter
import dgl

loaded_data = torch.load('./data.pth')

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


graph_train = dgl.DGLGraph()
graph_train.add_nodes(X_train.shape[1])
graph_train.add_edges(range(X_train.shape[1]), range(X_train.shape[1]))

graph_test = dgl.DGLGraph()
graph_test.add_nodes(X_test.shape[1])
graph_test.add_edges(range(X_test.shape[1]), range(X_test.shape[1]))


# 转换为 PyTorch 张量
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)  # 使用long类型标签
y_test = torch.tensor(y_test, dtype=torch.long)

# 然后，我们将张量包装为 TensorDataset 对象
train_dataset = TensorDataset(X_train, y_train)
test_dataset = TensorDataset(X_test, y_test)

# 最后，我们创建 DataLoader 对象
batch_size = 64  # 你可以选择适合你的硬件和数据的批量大小
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

num_class = 15

class GCN(nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        self.conv1 = dgl.nn.GraphConv(X_train.shape[1], 64)
        self.conv2 = dgl.nn.GraphConv(64, 128)
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, num_class)

    def forward(self, g, features):
        x = F.relu(self.conv1(g, features.unsqueeze(2)))  # Add an extra dimension for channels
        x = F.relu(self.conv2(g, x))
        x = torch.mean(x, dim=0)  # 使用均值池化操作
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

best_accuracy = 0.0
best_model_weights = None

gcn = GCN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(gcn.parameters(), lr=0.01, momentum=0.9)
num_epochs = 20

for epoch in trange(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # Forward pass
        outputs = gcn(graph_train, images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 在每个epoch结束后进行模型评估
    gcn.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = gcn(graph_test, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算准确度并更新最佳模型权重
        accuracy = 100 * correct / total
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_weights = gcn.state_dict()

        # 保存最后一个epoch的模型权重
        torch.save(gcn.state_dict(), './result/last.pt')

    TensorWriter.add_scalars('Loss', {'train': loss.item()}, epoch)
    TensorWriter.add_scalars('Acc', {'Acc': accuracy}, epoch)

# 保存最佳模型权重
torch.save(best_model_weights, './result/best.pt')
