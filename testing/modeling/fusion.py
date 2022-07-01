from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from testing.modeling.registry import Registry
from yacs.config import CfgNode as ConfigurationNode

FUSION_REGISTRY = Registry()


class DrugConcentrationFusion(nn.Module):
    def __init__(self, conc_in_features, drug_in_features):
        super(DrugConcentrationFusion, self).__init__()
        self.conc_in_features = conc_in_features
        self.drug_in_features = drug_in_features
        self.out_features = None


# Fusions using MLP and concatenation
class MLPBasedFusion(DrugConcentrationFusion):
    def __init__(self, conc_in_features: int, drug_in_features: int,
                 hidden_units: List[int],
                 dropout: float):
        super(MLPBasedFusion, self).__init__(conc_in_features, drug_in_features)
        prv = conc_in_features + drug_in_features
        for i, h in enumerate(hidden_units):
            self.add_module(f'fc_{i + 1}', nn.Linear(prv, h))
            prv = h
            if i % 2 == 0:
                self.add_module(f'batchnorm_{i + 1}', nn.BatchNorm1d(h))
        self.dropout_rate = dropout
        self.layer_count = len(hidden_units)
        self.out_features = prv


def _custom_forward(module: MLPBasedFusion, x):
    for i in range(module.layer_count):
        L = module._modules[f'fc_{i + 1}']
        x = L(x)
        x = F.relu(x)
        if i % 2 == 0:
            L = module._modules[f'batchnorm_{i + 1}']
            x = L(x)
        if i < module.layer_count - 1:
            x = F.dropout(x, p=module.dropout_rate, training=module.training)
    return x


class ConcatMLPConcentrationFusion(MLPBasedFusion):
    def __init__(self, conc_in_features: int, drug_in_features: int, hidden_units: List[int], dropout: float):
        super(ConcatMLPConcentrationFusion, self).__init__(conc_in_features, drug_in_features, hidden_units, dropout)

    def forward(self, drugs, concentrations):
        # drugs : [B x drug_embeddings]
        # concentrations : [B x concentration embeddings]
        x = torch.cat([drugs, concentrations], axis=1)
        return _custom_forward(self, x)


class ConcatMLPConcentrationFusionNoCompress(MLPBasedFusion):
    def __init__(self, conc_in_features: int, drug_in_features: int, hidden_units: List[int], dropout: float):
        super(ConcatMLPConcentrationFusionNoCompress, self).__init__(conc_in_features, drug_in_features, hidden_units,
                                                                     dropout)

    def forward(self, compound_nodes: torch.Tensor, batch_index: torch.Tensor, concentrations: torch.Tensor) -> [
        torch.Tensor, torch.Tensor]:
        # compound_nodes -> All the nodes of the compound collated
        # batch_index -> An index array the size of compound_nodes.shape[0] which defines
        # the corresponding batch of each node
        # Concentrations -> [B x conc_in_features]
        all_conc_nodes = concentrations[batch_index, :]
        joined = torch.cat([compound_nodes, all_conc_nodes], axis=1)
        ret = _custom_forward(self, joined)
        return ret, batch_index


@FUSION_REGISTRY.register('mlp-concat-fusion')
def build_mlp_concant_fusion(cfg_hyper_parameters: ConfigurationNode, drug_in_features: int):
    return ConcatMLPConcentrationFusion(
        dropout=cfg_hyper_parameters.DROPOUT,
        conc_in_features=cfg_hyper_parameters.CONC_IN_FEATURES,
        drug_in_features=drug_in_features,
        hidden_units=cfg_hyper_parameters.HIDDEN_UNITS
    )


@FUSION_REGISTRY.register('mlp-concat-fusion-no-compression')
def build_mlp_concat_fusion_with_no_compress(cfg_hyper_parameters: ConfigurationNode, drug_in_features: int):
    return ConcatMLPConcentrationFusionNoCompress(
        dropout=cfg_hyper_parameters.DROPOUT,
        conc_in_features=cfg_hyper_parameters.CONC_IN_FEATURES,
        drug_in_features=drug_in_features,
        hidden_units=cfg_hyper_parameters.HIDDEN_UNITS
    )

