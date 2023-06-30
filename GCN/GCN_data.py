import pandas as pd
import numpy as np
import networkx as nx
import os
# 读取数据
train_data = pd.read_csv('./data/mitbih_train.csv', header=None)
test_data = pd.read_csv('./data/mitbih_test.csv', header=None)

# 将数据和标签分离
train_labels = train_data.iloc[:, -1].values
train_data = train_data.iloc[:, :-1].values

test_labels = test_data.iloc[:, -1].values
test_data = test_data.iloc[:, :-1].values

def ecg_to_graph(ecg_data):
    # 创建一个空的有向图
    G = nx.DiGraph()

    # 将每个时间点添加为一个节点，节点的"signal_value"属性表示心电图信号的值
    for i in range(len(ecg_data)):
        G.add_node(i, signal_value=ecg_data[i])

    # 将每两个相邻的时间点之间添加一条边
    for i in range(len(ecg_data) - 1):
        G.add_edge(i, i + 1)

    return G

train_graphs = [ecg_to_graph(ecg_data) for ecg_data in train_data]
test_graphs = [ecg_to_graph(ecg_data) for ecg_data in test_data]


def save_graphs(graphs, directory):
    # 检查目标文件夹是否存在，如果不存在，创建它
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, G in enumerate(graphs):
        nx.write_gexf(G, f"{directory}/graph_{i}.gexf")

# 存储训练数据和测试数据
save_graphs(train_graphs, './data/train_graphs')
save_graphs(test_graphs, './data/test_graphs')
