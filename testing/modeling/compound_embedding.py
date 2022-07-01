from abc import ABC
from typing import List
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.nn import global_max_pool

from yacs.config import CfgNode as ConfigurationNode

from testing.modeling.registry import Registry

COMPOUND_EMBEDDER_REGISTRY = Registry()


class CompoundEmbedder(nn.Module):
    def __init__(self, in_features: int):
        super(CompoundEmbedder, self).__init__()
        self.in_features = in_features
        self.out_features = None


class GNNEmbedder(CompoundEmbedder, ABC):
    def __init__(self, in_features: int,
                 activation: str):
        super(GNNEmbedder, self).__init__(in_features)

        assert activation in ['relu', 'elu'], f'Activation {activation} not defined'

        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'elu':
            self.activation = F.elu

        self.layer_count: int = 0

    def forward(self, x, edge_index, batch):
        for i in range(self.layer_count):
            L = self._modules[f'gnn_{i + 1}']
            x = L(x, edge_index)
            x = self.activation(x)
        return x, batch


class GCNEmbedder(GNNEmbedder):
    def __init__(self,
                 in_features: int,
                 d_layers: List[int],
                 activation: str):
        super(GCNEmbedder, self).__init__(in_features, activation)

        # graph drug layers
        prv = in_features
        for i, d in enumerate(d_layers):
            self.add_module(f'gnn_{i + 1}', GCNConv(prv, d))
            prv = d
        self.layer_count = len(d_layers)
        self.out_features = prv


class GATEmbedder(GNNEmbedder):
    def __init__(self,
                 in_features: int,
                 nheads: int,
                 out_features: int,
                 activation: str,
                 dropout: float):
        super(GATEmbedder, self).__init__(in_features, activation)

        self.add_module(f'gnn_1', GATConv(in_features, out_features, heads=nheads, dropout=dropout))
        self.add_module(f'gnn_2', GATConv(out_features * nheads, out_features, dropout=dropout))
        self.out_features = out_features
        self.layer_count = 2


class GlobalPoolingEmbedder(nn.Module):
    def __init__(self, out_features, compound_embedder: GNNEmbedder):
        super(GlobalPoolingEmbedder, self).__init__()
        h = compound_embedder.out_features
        self.fc1 = nn.Linear(h, out_features // 4)
        self.fc2 = nn.Linear(out_features // 4, out_features // 2)
        self.compound_embedder = compound_embedder

    def forward(self, x, edge_index, batch):
        x, batch = self.compound_embedder(x, edge_index, batch)
        x = global_max_pool(x, batch)
        x = self.fc1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.fc2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)

        return x


def build_gnn_embedder(cfg_hyper_parameters: ConfigurationNode):
    if cfg_hyper_parameters.GNN_TYPE == 'GAT':
        return GATEmbedder(in_features=cfg_hyper_parameters.IN_FEATURES,
                           nheads=cfg_hyper_parameters.N_HEADS,
                           out_features=cfg_hyper_parameters.OUT_FEATURES,
                           activation=cfg_hyper_parameters.ACTIVATION,
                           dropout=cfg_hyper_parameters.DROPOUT)
    elif cfg_hyper_parameters.GNN_TYPE == 'GCN':
        return GCNEmbedder(in_features=cfg_hyper_parameters.IN_FEATURES,
                           d_layers=cfg_hyper_parameters.D_LAYERS,
                           activation=cfg_hyper_parameters.ACTIVATION)
    else:
        raise NotImplementedError(f"GNN type {cfg_hyper_parameters.GNN_TYPE} not implemented!")


@COMPOUND_EMBEDDER_REGISTRY.register('gnn-global-pooling')
def build_gnn_global_pooling_embedder(cfg_hyper_parameters: ConfigurationNode):
    compound_embedder = build_gnn_embedder(cfg_hyper_parameters)
    return GlobalPoolingEmbedder(out_features=cfg_hyper_parameters.POOLING_OUT_FEATURES,
                                 compound_embedder=compound_embedder)


@COMPOUND_EMBEDDER_REGISTRY.register('gnn-no-global-pooling')
def build_gnn_with_no_pooling_embedder(cfg_hyper_parameters: ConfigurationNode):
    return build_gnn_embedder(cfg_hyper_parameters)
