import torch
import numpy as np
import dgl
from dgl.base import EID,NID
from dgl.dataloading import NeighborSampler

def random_subgraph(num_clusters, graph, shuffle=True, save_e=[]):
    if shuffle:
        cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes())
    else:
        if not save_e:
            cluster_id = np.random.randint(low=0, high=num_clusters, size=graph.num_nodes())
            save_e.append(cluster_id)
        else:
            cluster_id = save_e[0]
    perm = np.arange(0, graph.num_nodes())
    batch_no = 0
    while batch_no < num_clusters:
        batch_nodes = perm[cluster_id == batch_no]
        batch_no += 1
        sub_g = graph.subgraph(batch_nodes)
        yield batch_nodes, sub_g

class EdgeSampleMultiLayerNeighborSampler(NeighborSampler):
    def __init__(self, fanouts, edge_sample_rate:list, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        if not isinstance(fanouts, list):
            fan = fanouts * len(edge_sample_rate)
        else:
            fan = fanouts
            assert len(edge_sample_rate) == len(fanouts)
        for sample_rate in edge_sample_rate:
            assert sample_rate > 0.001 and sample_rate <= 1
        self._sample_rate = edge_sample_rate
        super().__init__(fanouts=fan, edge_dir=edge_dir, prob=prob, replace=replace,
                         prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        print("Init %s" % str(self.__class__))
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        layer = len(self._sample_rate) - 1
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            random_num = torch.rand(frontier.edata['_ID'].size())
            edge_id = torch.arange(frontier.num_edges())
            random_edges = edge_id[random_num < self._sample_rate[layer]]
            layer = layer - 1
            sub_frontier = dgl.edge_subgraph(frontier, random_edges, relabel_nodes=False, store_ids=False)
            eid = sub_frontier.edata[EID]
            block = dgl.to_block(sub_frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks

class EdgeSampleNeighborSampler(NeighborSampler):
    def __init__(self, fanouts, edge_sample_rate, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None, min_fanout=-1):
        super().__init__(fanouts=fanouts, edge_dir=edge_dir, prob=prob, replace=replace,
                        prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        assert edge_sample_rate > 0.001 and edge_sample_rate <= 1
        self._sample_rate = edge_sample_rate
        self._min_fanout = min_fanout
        print("Init %s" % str(self.__class__))
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            random_num = torch.rand(frontier.edata['_ID'].size())
            edge_id = torch.arange(frontier.num_edges())
            random_edges = edge_id[random_num < self._sample_rate]
            sub_frontier = dgl.edge_subgraph(frontier, random_edges, relabel_nodes=False, store_ids=False)
            if self.edge_dir == "in" and self._min_fanout >= 1:
                sub_in_degree = sub_frontier.in_degrees(seed_nodes)
                small_src_in_sub =  seed_nodes[sub_in_degree < self._min_fanout]
                if small_src_in_sub.size()[0] > 0:
                    index_in_frontier = torch.nonzero(torch.isin(frontier.all_edges()[1], small_src_in_sub))
                    if index_in_frontier.size()[0] >= 1:
                        random_num[index_in_frontier] = 0
                        random_edges = edge_id[random_num < self._sample_rate]
                        sub_frontier = dgl.edge_subgraph(frontier, random_edges, relabel_nodes=False, store_ids=False)
            eid = sub_frontier.edata[EID]
            block = dgl.to_block(sub_frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks

class InSeedNodeNeighborSampler(NeighborSampler):
    def __init__(self, fanouts, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        super().__init__(fanouts=fanouts, edge_dir=edge_dir, prob=prob, replace=replace,
                        prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        print("Init %s" % str(self.__class__))

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, edge_dir=self.edge_dir, prob=self.prob,
                replace=self.replace, output_device=self.output_device,
                exclude_edges=exclude_eids)
            edge_id = torch.nonzero(torch.isin(frontier.all_edges()[0], seed_nodes)).view(-1)
            sub_frontier = dgl.edge_subgraph(frontier, edge_id, relabel_nodes=False, store_ids=False)
            eid = sub_frontier.edata[EID]
            block = dgl.to_block(sub_frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks

class InSeedNodeFullNeighborSampler(NeighborSampler):
    def __init__(self, layer_num, edge_dir='in', prob=None, replace=False,
                 prefetch_node_feats=None, prefetch_labels=None, prefetch_edge_feats=None,
                 output_device=None):
        fanouts = [-1] * layer_num
        self._layer_num = layer_num
        super().__init__(fanouts=fanouts, edge_dir=edge_dir, prob=prob, replace=replace,
                        prefetch_node_feats=prefetch_node_feats,
                         prefetch_labels=prefetch_labels,
                         prefetch_edge_feats=prefetch_edge_feats,
                         output_device=output_device)
        print("Init %s" % str(self.__class__))
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        frontier = g.sample_neighbors(
            seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
            replace=self.replace, output_device=self.output_device,
            exclude_edges=exclude_eids)
        edge_id = torch.nonzero(torch.isin(frontier.all_edges()[0], seed_nodes)).view(-1)
        sub_frontier = dgl.edge_subgraph(frontier, edge_id, relabel_nodes=False, store_ids=False)
        eid = sub_frontier.edata[EID]
        for i in range(self._layer_num):
            block = dgl.to_block(sub_frontier, seed_nodes)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks

class InSeedNodeFullNeighborSampler2(InSeedNodeFullNeighborSampler):
    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        frontier = g.sample_neighbors(
            seed_nodes, -1, edge_dir=self.edge_dir, prob=self.prob,
            replace=self.replace, output_device=self.output_device,
            exclude_edges=exclude_eids)
        edge_id = torch.nonzero(torch.isin(frontier.all_edges()[0], seed_nodes)).view(-1)
        sub_frontier = dgl.edge_subgraph(frontier, edge_id, relabel_nodes=False, store_ids=False)
        eid = sub_frontier.edata[EID]
        block = dgl.to_block(sub_frontier, seed_nodes)
        block.edata[EID] = eid
        seed_nodes = block.srcdata[NID]
        for i in range(self._layer_num):
            blocks.insert(0, block)
        return seed_nodes, output_nodes, blocks