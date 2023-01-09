import random
import torch
import numpy as np
import dgl


import torch.nn.functional as F
from model_gipa import GIPA_SIMPLE
from new_models import GIPASMConv, AGDN_SM


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
    if args.model == "gipa_simple":
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
            edge_att_act=args.edge_att_act,
            edge_agg_mode=args.edge_agg_mode,
            use_node_sparse = args.use_sparse_fea
        )
    else:
        model = AGDN_SM(
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
            attn_drop=0.0,
            hop_attn_drop=0.0,
            edge_drop=args.edge_drop,
            K=1,
            use_attn_dst=not args.no_attn_dst,
            norm=args.norm,
            use_one_hot=False,
            use_labels=False,
            weight_style="sum",
            batch_norm=not args.disable_fea_trans_norm,
            edge_att_act=args.edge_att_act, edge_agg_mode=args.edge_agg_mode,
            conv_kernel=GIPASMConv
        )
    return model
