import torch
import dgl.function as fn
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

def compute_norm(graph):
    degs = graph.in_degrees().float().clamp(min=1)
    deg_isqrt = torch.pow(degs, -0.5)

    degs = graph.in_degrees().float().clamp(min=1)
    deg_sqrt = torch.pow(degs, 0.5)

    return deg_sqrt, deg_isqrt


def load_data(dataset, root_path):
    data = DglNodePropPredDataset(name=dataset, root=root_path)
    evaluator = Evaluator(name=dataset)
    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]
    graph.ndata["labels"] = labels
    print(f"Nodes : {graph.number_of_nodes()}\n"
          f"Edges: {graph.number_of_edges()}\n"
          f"Train nodes: {len(train_idx)}\n"
          f"Val nodes: {len(val_idx)}\n"
          f"Test nodes: {len(test_idx)}")
    return graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph, labels, edge_agg_as_feat=True, user_adj=True, user_avg=True):
    # The sum of the weights of adjacent edges is used as node features.
    if edge_agg_as_feat:
        graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    if user_adj or user_avg:
        deg_sqrt, deg_isqrt = compute_norm(graph)
        if user_adj:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_sqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm_adjust"))

        if user_avg:
            graph.srcdata.update({"src_norm": deg_isqrt})
            graph.dstdata.update({"dst_norm": deg_isqrt})
            graph.apply_edges(fn.u_mul_v("src_norm", "dst_norm", "gcn_norm"))

    graph.create_formats_()
    print(graph.ndata.keys())
    print(graph.edata.keys())
    return graph, labels