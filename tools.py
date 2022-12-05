import random
import torch
import numpy as np
import dgl

from model_gipa import GIPA_SIMPLE
import torch.nn.functional as F

def count_model_parameters(model:torch.nn.Module):
    for name, parameters in model.named_parameters():
        print(name, ':', parameters.size())
    n_parameters = sum([np.prod(p.size()) for p in model.parameters() if p.requires_grad])
    return n_parameters

def seed(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    dgl.random.seed(seed)

def print_msg_and_write(out_msg, log_f):
    print(out_msg)
    log_f.write(out_msg)
    log_f.flush()


def get_model(args, n_node_feats, n_edge_feats, n_classes):
    model = GIPA_SIMPLE(
        n_node_feats,
        n_edge_feats,
        n_classes,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        n_hidden=args.n_hidden,
        edge_emb=args.edge_emb_size,
        activation=F.relu,
        dropout=args.dropout,
        input_drop=args.input_drop,
        edge_drop=args.edge_drop,
        use_attn_dst=not args.no_attn_dst,
        norm=args.norm,
        batch_norm=not args.disable_fea_trans_norm,
        edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode
    )
    return model

#
# def get_model2(args, n_node_feats, n_edge_feats, n_classes):
#     model = AGDN_SM(
#         n_node_feats,
#         n_edge_feats,
#         n_classes,
#         n_layers=args.n_layers,
#         n_heads=args.n_heads,
#         n_hidden=args.n_hidden,
#         edge_emb=args.edge_emb_size,
#         activation=F.relu,
#         dropout=args.dropout,
#         input_drop=args.input_drop,
#         attn_drop=args.attn_drop,
#         hop_attn_drop=args.hop_attn_drop,
#         edge_drop=args.edge_drop,
#         K=args.K,
#         use_attn_dst=not args.no_attn_dst,
#         norm=args.norm,
#         use_one_hot=args.use_one_hot_feature,
#         use_labels=args.use_labels,
#         weight_style=args.weight_style,
#         batch_norm=not args.disable_fea_trans_norm,
#         edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode,
#         conv_kernel=GIPASMConv
#     )
#     return model