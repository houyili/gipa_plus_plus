from data_loader import load_data, preprocess

# root_path = "/Users/lihouyi/Documents/opensource/data_source/proteins"
root_path = "/data/ogb/datasets/"
dataset = "ogbn-proteins"
n_node_feats, n_edge_feats, n_classes = 0, 8, 112

graph, labels, train_idx, val_idx, test_idx, evaluator = load_data(dataset, root_path)
graph, labels = preprocess(graph, labels, use_sparse=True)

print(graph.edata["feat"][0:10])
print(graph.edata["sparse"][0:10])