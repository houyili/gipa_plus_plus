import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.ops import edge_softmax
from dgl.utils import expand_as_pair

from utils import get_act_by_str

class GIPAConv(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            out_feats,
            n_heads,
            edge_drop=0.0,
            negative_slope=0.2,
            activation=None,
            use_attn_dst=True,
            allow_zero_in_degree=True,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="both_softmax"
    ):
        super(GIPAConv, self).__init__()
        self._n_heads = n_heads
        self._in_src_feats, self._in_dst_feats = expand_as_pair(node_feats)
        self._out_feats = out_feats
        self._allow_zero_in_degree = allow_zero_in_degree
        self._norm = norm
        self._batch_norm = batch_norm
        self._edge_agg_mode = edge_agg_mode
        self._edge_att_act = edge_att_act

        # project function
        self.src_fc = nn.Linear(self._in_src_feats, out_feats * n_heads, bias=False)

        # propagation function
        self.attn_src_fc = nn.Linear(self._in_src_feats, n_heads, bias=False)
        self.attn_dst_fc = nn.Linear(self._in_src_feats, n_heads, bias=False) if use_attn_dst else None
        self.attn_edge_fc = nn.Linear(edge_feats, n_heads, bias=False) if edge_feats > 0 else None
        self.edge_norm = nn.BatchNorm1d(edge_feats) if edge_feats > 0 else None
        self.edge_att_actv = get_act_by_str(edge_att_act, negative_slope)
        self.edge_drop = edge_drop

        #  aggregation function
        self.offset = nn.Parameter(torch.zeros(size=(1, n_heads, out_feats)))
        self.scale = nn.Parameter(torch.ones(size=(1, n_heads, out_feats)))
        self.agg_fc = nn.Linear(out_feats * n_heads, out_feats * n_heads)

        # apply function
        self.dst_fc = nn.Linear(self._in_src_feats, out_feats * n_heads)
        self.activation = activation
        print("Init %s" % str(self.__class__))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.src_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.attn_src_fc.weight, gain=gain)
        if self.attn_dst_fc is not None:
            nn.init.xavier_normal_(self.attn_dst_fc.weight, gain=gain)
        if self.attn_edge_fc is not None:
            nn.init.xavier_normal_(self.attn_edge_fc.weight, gain=gain)

        nn.init.xavier_normal_(self.agg_fc.weight, gain=gain)

        if self.dst_fc is not None:
            nn.init.xavier_normal_(self.dst_fc.weight, gain=gain)

    def set_allow_zero_in_degree(self, set_value):
        self._allow_zero_in_degree = set_value

    def agg_func(self, h):
        if self._batch_norm:
            mean = h.mean(dim=-1).view(h.shape[0], self._n_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._n_heads, 1) + 1e-9
            h = (h - mean) * self.scale * torch.rsqrt(var) + self.offset
        return self.agg_fc(h.view(-1, self._out_feats * self._n_heads)).view(-1, self._n_heads, self._out_feats)

    def forward(self, graph, feat_src, feat_edge=None):
        with graph.local_scope():
            if graph.is_block:
                feat_dst = feat_src[: graph.number_of_dst_nodes()]
            else:
                feat_dst = feat_src

            # project function: source node
            feat_src_fc = self.src_fc(feat_src).view(-1, self._n_heads, self._out_feats)

            # propagation function: source node
            attn_src = self.attn_src_fc(feat_src).view(-1, self._n_heads, 1)
            graph.srcdata.update({"feat_src_fc": feat_src_fc, "attn_src": attn_src})

            # propagation function: dst node
            if self.attn_dst_fc is not None:
                attn_dst = self.attn_dst_fc(feat_dst).view(-1, self._n_heads, 1)
                graph.dstdata.update({"attn_dst": attn_dst})
                graph.apply_edges(fn.u_add_v("attn_src", "attn_dst", "attn_node"))
            else:
                graph.apply_edges(fn.copy_u("attn_src", "attn_node"))

            # propagation function: edge
            e = graph.edata["attn_node"]
            if feat_edge is not None:
                attn_edge = self.attn_edge_fc(feat_edge).view(-1, self._n_heads, 1)
                graph.edata.update({"attn_edge": attn_edge})
                e += graph.edata["attn_edge"]
            e = self.edge_att_actv(e)

            # edge drop
            if self.training and self.edge_drop > 0:
                perm = torch.randperm(graph.number_of_edges(), device=e.device)
                bound = int(graph.number_of_edges() * self.edge_drop)
                eids = perm[bound:]
            else:
                eids = torch.arange(graph.number_of_edges(), device=e.device)
            graph.edata["a"] = torch.zeros_like(e)

            # edge softmax
            if self._edge_agg_mode == "single_softmax":
                graph.edata["a"][eids] = edge_softmax(graph, e[eids], eids=eids, norm_by='dst')
            else:
                graph.edata["a"][eids] = e[eids]

            # graph normalize
            if self._norm == "adj":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["gcn_norm_adjust"][eids].view(-1, 1, 1)
            if self._norm == "avg":
                graph.edata["a"][eids] = graph.edata["a"][eids] * graph.edata["gcn_norm"][eids].view(-1, 1, 1)

            # aggregation
            graph.update_all(fn.u_mul_e("feat_src_fc", "a", "m"), fn.sum("m", "feat_src_fc"))
            agg_msg = self.agg_func(graph.dstdata["feat_src_fc"])

            # apply part
            if self.dst_fc is not None:
                feat_dst_fc = self.dst_fc(feat_dst).view(-1, self._n_heads, self._out_feats)
                rst = agg_msg + feat_dst_fc  # apply = fc(concat([h_{k-1}, msg]))
            else:
                rst = agg_msg

            if self.activation is not None:
                rst = self.activation(rst, inplace=True)

            return rst

class GIPA_SIMPLE(nn.Module):
    def __init__(
            self,
            node_feats,
            edge_feats,
            n_classes,
            n_layers,
            n_heads,
            n_hidden,
            edge_emb,
            activation,
            dropout,
            input_drop,
            edge_drop,
            use_attn_dst=True,
            allow_zero_in_degree=False,
            norm="none",
            batch_norm=True,
            edge_att_act="leaky_relu",
            edge_agg_mode="none_softmax",
            first_hidden = 150
    ):
        super().__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, first_hidden)

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_norms = nn.ModuleList()

        for i in range(n_layers):
            in_hidden =  n_heads * n_hidden if i > 0 else first_hidden
            out_hidden = n_hidden

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_norms.append(nn.BatchNorm1d(edge_emb))
            self.convs.append(
                GIPAConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    batch_norm=batch_norm,
                    edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        print("The new parameter are %s,%s,%s" % (batch_norm, edge_att_act, edge_agg_mode))
        print("Init %s" % str(self.__class__))

    def forward(self, g):
        if not isinstance(g, list):
            subgraphs = [g] * self.n_layers
        else:
            subgraphs = g

        h = subgraphs[0].srcdata["feat"]


        h = self.node_encoder(h)
        h = F.relu(h, inplace=True)
        h = self.input_drop(h)

        h_last = None

        for i in range(self.n_layers):

            if self.edge_encoder is not None:
                efeat = subgraphs[i].edata["feat"]
                efeat_emb = self.edge_encoder[i](efeat)
                efeat_emb = F.relu(efeat_emb, inplace=True)
            else:
                efeat_emb = None

            h = self.convs[i](subgraphs[i], h, efeat_emb).flatten(1, -1)

            if h_last is not None:
                h += h_last[: h.shape[0], :]

            h_last = h
            h = self.norms[i](h)
            h = self.activation(h, inplace=True)
            h = self.dropout(h)

        h = self.pred_linear(h)
        return h