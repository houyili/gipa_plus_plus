from models import AGDN, AGDNConv, EdgeAttentionLayer
import torch
import torch.nn as nn

class AGDNSMConv(AGDNConv):
    def feat_trans(self, h, idx):
        if self._batch_norm:
            mean = h.mean(dim=-1).view(h.shape[0], self._n_heads, 1)
            var = h.var(dim=-1, unbiased=False).view(h.shape[0], self._n_heads, 1) + 1e-9
            h = (h - mean) * self.scale[idx] * torch.rsqrt(var) + self.offset[idx]
        return h

class GIPASMConv(AGDNSMConv):
    def __init__(self,node_feats,
        edge_feats,
        out_feats,
        n_heads=1,
        K=3,
        attn_drop=0.0,
        hop_attn_drop=0.0,
        edge_drop=0.0,
        negative_slope=0.2,
        residual=True,
        activation=None,
        use_attn_dst=True,
        allow_zero_in_degree=True,
        norm="none",
        batch_norm=True,
        weight_style="HA", edge_att_act="leaky_relu", edge_agg_mode="both_softmax"):
        super(GIPASMConv, self).__init__(node_feats, edge_feats, out_feats, n_heads,
        K, attn_drop, hop_attn_drop, edge_drop, negative_slope, residual, activation, use_attn_dst,
        allow_zero_in_degree, norm, batch_norm, weight_style, edge_att_act, edge_agg_mode)
        self.agg_fc = nn.Linear(out_feats * n_heads, out_feats * n_heads)
        self.reset_other_parameters()

    def reset_other_parameters(self):
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_normal_(self.agg_fc.weight, gain=gain)
        print("Reset %s" % str(self.agg_fc.__class__))

    def feat_trans(self, h, idx):
        h = super(GIPASMConv, self).feat_trans(h, idx)
        return self.agg_fc(h.view(-1, self._out_feats * self._n_heads)).view(-1, self._n_heads, self._out_feats)


class AGDN_MA(AGDN):
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
            attn_drop,
            hop_attn_drop,
            edge_drop,
            K=3,
            use_attn_dst=True,
            allow_zero_in_degree=False,
            norm="none",
            use_one_hot=False,
            use_labels=False,
            edge_attention=False,
            weight_style="HA", batch_norm=True, edge_att_act="leaky_relu", edge_agg_mode="both_softmax"
    ):
        # super(GIPA, self).__init__(node_feats, edge_feats,
        #     n_classes, n_layers, n_heads, n_hidden, edge_emb, activation, dropout, input_drop,
        #     attn_drop, hop_attn_drop, edge_drop, K, use_attn_dst, allow_zero_in_degree,
        #     norm, use_one_hot, use_labels, edge_attention,
        #     weight_style, batch_norm, edge_att_act, edge_agg_mode)
        super(AGDN, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, 150)
        if edge_attention:
            self.pre_aggregator = EdgeAttentionLayer(edge_feats, n_heads)
        else:
            self.pre_aggregator = None
        if use_one_hot:
            self.one_hot_encoder = nn.Linear(8, 8)
        else:
            self.one_hot_encoder = None

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_norms = nn.ModuleList()
        else:
            self.edge_encoder = None

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else 150
            out_hidden = n_hidden
            # bias = i == n_layers - 1

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_norms.append(nn.BatchNorm1d(edge_emb))
            self.convs.append(
                AGDNConv(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    K=K,
                    attn_drop=attn_drop,
                    hop_attn_drop=hop_attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    residual=True,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    weight_style=weight_style, batch_norm=batch_norm, edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

class AGDN_SM(AGDN):
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
            attn_drop,
            hop_attn_drop,
            edge_drop,
            K=3,
            use_attn_dst=True,
            allow_zero_in_degree=False,
            norm="none",
            use_one_hot=False,
            use_labels=False,
            edge_attention=False,
            weight_style="HA",
            batch_norm=True, edge_att_act="leaky_relu", edge_agg_mode="both_softmax", conv_kernel=AGDNSMConv
    ):
        super(AGDN, self).__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_hidden = n_hidden
        self.n_classes = n_classes

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.node_encoder = nn.Linear(node_feats, 150)
        if edge_attention:
            self.pre_aggregator = EdgeAttentionLayer(edge_feats, n_heads)
        else:
            self.pre_aggregator = None
        if use_one_hot:
            self.one_hot_encoder = nn.Linear(8, 8)
        else:
            self.one_hot_encoder = None

        if edge_emb > 0:
            self.edge_encoder = nn.ModuleList()
            self.edge_norms = nn.ModuleList()
        else:
            self.edge_encoder = None

        for i in range(n_layers):
            in_hidden = n_heads * n_hidden if i > 0 else 150
            out_hidden = n_hidden

            if edge_emb > 0:
                self.edge_encoder.append(nn.Linear(edge_feats, edge_emb))
                self.edge_norms.append(nn.BatchNorm1d(edge_emb))
            self.convs.append(
                conv_kernel(
                    in_hidden,
                    edge_emb,
                    out_hidden,
                    n_heads=n_heads,
                    K=K,
                    attn_drop=attn_drop,
                    hop_attn_drop=hop_attn_drop,
                    edge_drop=edge_drop,
                    use_attn_dst=use_attn_dst,
                    residual=True,
                    allow_zero_in_degree=allow_zero_in_degree,
                    norm=norm,
                    weight_style=weight_style, batch_norm=batch_norm, edge_att_act=edge_att_act,
                    edge_agg_mode=edge_agg_mode
                )
            )
            self.norms.append(nn.BatchNorm1d(n_heads * out_hidden))

        self.pred_linear = nn.Linear(n_heads * n_hidden, n_classes)

        self.input_drop = nn.Dropout(input_drop)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation