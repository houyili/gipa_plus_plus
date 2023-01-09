import torch
import dgl.function as fn
from torch.nn import functional
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator


def transform_edge_feature_to_sparse(raw_edge_fea):
    edge_fea_list = []
    for i in range(8):
        print(raw_edge_fea[:, i].size())

        if i == 0:
            for value in [0.0010, 0.5010]:
                res = (raw_edge_fea[:, i] == value).float()
                edge_fea_list.append(res if value == 0.001 else res * raw_edge_fea[:, i])
        elif i == 6:
            for value in [0.0010, 0.9010, 0.6010, 0.6510, 0.5410]:
                res = (raw_edge_fea[:, i] == value).float()
                edge_fea_list.append(res if value == 0.001 else res * raw_edge_fea[:, i])
        else:
            edge_fea_list.append((raw_edge_fea[:, i] == 0.0010).float())
            possible = (raw_edge_fea[:, i] != 0.0010).float() * raw_edge_fea[:, i]
            print(possible.size())
            one_hot = functional.one_hot((raw_edge_fea[:, i] * 30).long()).float()
            print(one_hot.size())
            edge_fea_list.append(possible * one_hot)
    sparse = torch.concat(edge_fea_list, dim=-1)
    return sparse

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

def preprocess(graph, labels, edge_agg_as_feat=True, user_adj=False, user_avg=False, use_sparse=False):
    # The sum of the weights of adjacent edges is used as node features.
    if use_sparse:
        # graph.edata.update({"sparse": (graph.edata['feat'] * 1000).int()})
        # graph.edata.update({"sparse": one_hot((graph.edata['feat'][:,1] * 100).long())})
        edge_sparse = transform_edge_feature_to_sparse(graph.edata['feat'])
        graph.edata.update({"sparse": edge_sparse})
        graph.update_all(fn.copy_e("sparse", "sparse_c"), fn.sum("sparse_c", "sparse"))

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