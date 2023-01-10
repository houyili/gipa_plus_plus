import torch
from data_loader import load_data, preprocess

# root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
root_path = "/data/ogb/datasets/"
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112

graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path)
graph, labels = preprocess(graph, labels, sparse_encoder="hard_30")

for id in [train_idx, val_idx, test_idx]:
    print(id)
    sparse = torch.mean(graph.ndata["sparse"][id], dim=0)
    dense = torch.mean(graph.ndata["feat"][id], dim=0)
    print(sparse.shape)
    print(dense.shape)
    # str = ""
    # for i in range(len(sparse.tolist())):
    #     str = str + ", %.03f" %sparse[i]
    # # print(str)
    # str = ""
    # for i in range(len(dense.tolist())):
    #     str = str + ", %.03f" % dense[i]
    # print(str)

# print(graph.edata["feat"][0])
# print(graph.edata["sparse"][0])