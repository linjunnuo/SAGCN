import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
import torch.nn.functional as F

# ------------------------------- 加载数据并分离特征和标签 ------------------------------- #
train_data = pd.read_csv('./data/mitbih_train.csv', header=None)
test_data = pd.read_csv('./data/mitbih_test.csv', header=None)

train_labels = torch.tensor(train_data.iloc[:, -1].values, dtype=torch.long)
train_features = train_data.iloc[:, :-1].values

test_labels = torch.tensor(test_data.iloc[:, -1].values, dtype=torch.long)
test_features = test_data.iloc[:, :-1].values

# --------------------------------- 转化为图结构数据 --------------------------------- #
def data_to_graph(data, label):
    edge_index = torch.tensor([[i, i+1] for i in range(data.shape[0]-1)], dtype=torch.long)
    return Data(x=torch.tensor(data, dtype=torch.float).view(-1, 1), 
                edge_index=edge_index.t().contiguous(), 
                y=torch.full((data.shape[0],), label, dtype=torch.long))

train_graphs = [data_to_graph(data, label) for data, label in zip(train_features, train_labels)]
test_graphs = [data_to_graph(data, label) for data, label in zip(test_features, test_labels)]

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=True)


# ---------------------------------- 定义GCN模型 --------------------------------- #
class Net(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Net, self).__init__()
        self.conv1 = GCNConv(input_dim, 16)
        self.conv2 = GCNConv(16, 64)
        self.conv3 = GCNConv(64, 64)
        self.conv4 = GCNConv(64, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv4(x, edge_index)

        return F.log_softmax(x, dim=1)


# ----------------------------------- 超参数设置 ---------------------------------- #
input_dim = 1  # 输入特征维度
num_classes = 5  # 类别数量
learning_rate = 0.0001
num_epochs = 10

# ----------------------------------- 设备设置 ----------------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------------- 创建模型实例 ---------------------------------- #
model = Net(input_dim, num_classes).to(device)

# -------------------------------- 定义损失函数和优化器 -------------------------------- #
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# ----------------------------------- 训练模型 ----------------------------------- #
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * data.num_graphs
        _, predicted = torch.max(output.data, 1)
        train_correct += (predicted == data.y).sum().item()
        train_total += data.y.size(0)  # 累积样本数量

    train_loss /= len(train_loader.dataset)
    train_accuracy = 100.0 * train_correct / train_total  # 使用累积的样本数量计算准确率

    # --------------------------------- 在测试集上验证模型 -------------------------------- #
    model.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            test_correct += (predicted == data.y).sum().item()
            test_total += data.y.size(0)  # 累积样本数量

    test_accuracy = 100.0 * test_correct / test_total  # 使用累积的样本数量计算准确率

    # ---------------------------------- 打印训练结果 ---------------------------------- #
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, Test Accuracy: {test_accuracy:.2f}%')

