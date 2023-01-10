import torch
from data_loader import load_data, preprocess

# root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
root_path = "/data/ogb/datasets/"
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112

graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path)
graph, labels = preprocess(graph, labels, sparse_encoder="hard_30")

print(train_idx)
print(val_idx)
print(test_idx)

print(torch.mean(graph.ndata["sparse"][train_idx], dim=-1))
print(torch.mean(graph.ndata["sparse"][val_idx], dim=-1))
print(torch.mean(graph.ndata["sparse"][test_idx], dim=-1))


print(torch.mean(graph.ndata["feat"][train_idx], dim=-1))
print(torch.mean(graph.ndata["feat"][val_idx], dim=-1))
print(torch.mean(graph.ndata["feat"][test_idx], dim=-1))

# print(graph.edata["feat"][0])
# print(graph.edata["sparse"][0])