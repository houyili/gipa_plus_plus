import torch
import dgl.function as fn
from torch.nn import functional
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator

def transform_edge_feature_to_sparse(raw_edge_fea, split_num:int = 30):
    edge_fea_list = []
    for i in range(8):
        print("Process edge feature == %d " %i)
        print("The edge feature size ", raw_edge_fea[:, i].size())
        if i == 0:
            for value in [0.0010, 0.5010]:
                res = torch.reshape((raw_edge_fea[:, i] == value).float(), [-1, 1])
                edge_fea_list.append(res if value == 0.001 else res * torch.reshape(raw_edge_fea[:, i], [-1,1]))
        elif i == 6:
            for value in [0.0010, 0.9010, 0.6010, 0.6510, 0.5410]:
                res = torch.reshape((raw_edge_fea[:, i] == value).float(), [-1, 1])
                edge_fea_list.append(res if value == 0.001 else res * torch.reshape(raw_edge_fea[:, i], [-1,1]))
        else:
            edge_fea_list.append(torch.reshape((raw_edge_fea[:, i] == 0.0010).float(), [-1, 1]))
            possible = (raw_edge_fea[:, i] != 0.0010).float() * raw_edge_fea[:, i]
            print("The edge possible size ", possible.size())
            one_hot = functional.one_hot((raw_edge_fea[:, i] * split_num).long()).float()
            print("The edge one hot size ", one_hot.size())
            edge_fea_list.append(one_hot * torch.reshape(possible, [-1, 1]))
    sparse = torch.concat(edge_fea_list, dim=-1)
    print(sparse.size())
    return sparse

def transform_edge_feature_to_sparse2(raw_edge_fea, split_num:int = 30):
    edge_fea_list = []
    for i in range(8):
        print("Process edge feature == %d " %i)
        this_fea = torch.reshape(raw_edge_fea[:, i], [-1,1])
        print("The edge feature size ", raw_edge_fea[:, i].size(), " transform to ", this_fea.size())
        if i == 0:
            for value in [0.0010, 0.5010]:
                res = torch.reshape((raw_edge_fea[:, i] == value).float(), [-1, 1])
                edge_fea_list.append(res * this_fea)
        elif i == 6:
            for value in [0.0010, 0.9010, 0.6010, 0.6510, 0.5410]:
                res = torch.reshape((raw_edge_fea[:, i] == value).float(), [-1, 1])
                edge_fea_list.append(res * this_fea)
        else:
            one_hot = functional.one_hot((raw_edge_fea[:, i] * split_num).long()).float()
            print("The edge one hot size ", one_hot.size())
            edge_fea_list.append(one_hot * this_fea)
    sparse = torch.concat(edge_fea_list, dim=-1)
    print(sparse.size())
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

def preprocess(graph, labels, edge_agg_as_feat=True, user_adj=False, user_avg=False, sparse_encoder:str=None):
    if edge_agg_as_feat:
        graph.update_all(fn.copy_e("feat", "feat_copy"), fn.sum("feat_copy", "feat"))

    if sparse_encoder is not None:
        if len(sparse_encoder) > 0:
            edge_sparse = transform_edge_feature_to_sparse2(graph.edata['feat'], int(sparse_encoder.split("_")[-1]))

        else:
            edge_sparse = transform_edge_feature_to_sparse(graph.edata['feat'])
        graph.edata.update({"sparse": edge_sparse})
        graph.update_all(fn.copy_e("sparse", "sparse_c"), fn.sum("sparse_c", "sparse"))
        if sparse_encoder.find("edge_reverse") != -1:
            graph.edata.update({"feat": edge_sparse})
        del graph.edata["sparse"]

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