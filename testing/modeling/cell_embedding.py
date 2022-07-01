import torch
import torch.nn as nn
import torch.nn.functional as F

from testing.modeling.registry import Registry
from yacs.config import CfgNode as ConfigurationNode

CELL_EMBEDDER_REGISTRY = Registry()


class CellEmbedder(nn.Module):
    def __init__(self, dropout: float = 0.2,
                 out_features: int = 100):
        super(CellEmbedder, self).__init__()
        self.dropout_rate = dropout
        self.out_features = out_features


class MLPCellEmbedder(CellEmbedder):
    def __init__(self, dropout: float,
                 out_features: int,
                 in_features: int,
                 h1: int = 2048,
                 h2: int = 512):
        super(MLPCellEmbedder, self).__init__(dropout, out_features)
        self.expression_layer1 = nn.Linear(in_features, h1, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(h1)
        self.expression_layer2 = nn.Linear(h1, h2, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(h2)
        self.expression_layer3 = nn.Linear(h2, out_features)

    def forward(self, expression_encodings):
        x = self.expression_layer1(expression_encodings)
        x = F.relu(x)
        x = self.batchnorm1(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.expression_layer2(x)
        x = F.relu(x)
        x = self.batchnorm2(x)
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.expression_layer3(x)
        x = F.relu(x)
        return x


@CELL_EMBEDDER_REGISTRY.register('mlp-cell-embedder')
def build_simple_cell_embedder_with_expression(cfg_hyper_parameters: ConfigurationNode):
    return MLPCellEmbedder(dropout=cfg_hyper_parameters.DROPOUT,
                           out_features=cfg_hyper_parameters.OUT_FEATURES,
                           in_features=cfg_hyper_parameters.IN_FEATURES,
                           h1=cfg_hyper_parameters.H1,
                           h2=cfg_hyper_parameters.H2)
