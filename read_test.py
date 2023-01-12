import torch
from data_loader import load_data, preprocess

# root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
root_path = "/data/ogb/datasets/"
dataset = "ogbn-proteins"

graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path)

print(test_idx.shape)
# graph, labels = preprocess(graph, labels, sparse_encoder="count_30")
# graph, labels = preprocess(graph, labels, sparse_encoder="hard_6")
graph, labels = preprocess(graph, labels, sparse_encoder="hard_30")

for id in [train_idx, val_idx, test_idx]:
    num = id.shape[0]
    x = graph.ndata["sparse"][id]
    non_zero_num = torch.count_nonzero(x, dim=0) / num
    str = ""
    for i in range(len(non_zero_num)):
        str = str + "%.03f, " %non_zero_num[i]
    print(str)

# for id in [train_idx, val_idx, test_idx]:
#     sparse = torch.mean(graph.ndata["sparse"][id], dim=0).tolist()
#     max_s = torch.max(graph.ndata["sparse"][id], dim=0).values.tolist()
#     str = ""
#     for i in range(len(sparse)):
#         str = str + "%.03f, " %sparse[i]
#     for i in range(len(max_s)):
#         str = str + ", %.03f" % max_s[i]
#     print(str)

# print(graph.edata["feat"][0])
# print(graph.edata["sparse"][0])