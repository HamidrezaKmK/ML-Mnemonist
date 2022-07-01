from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_batch

from testing.modeling.registry import Registry
from yacs.config import CfgNode as ConfigurationNode

INTEGRATION_REGISTRY = Registry()


class DrugCellIntegrationModel(nn.Module):
    def __init__(self, drug_in_features: int, cell_in_features: int):
        super(DrugCellIntegrationModel, self).__init__()
        self.drug_in_features = drug_in_features
        self.cell_in_features = cell_in_features
        self.out_features = None


class SimpleConcatModel(DrugCellIntegrationModel):
    # TODO: add documentation
    def __init__(self, drug_in_features: int, cell_in_features: int, max_drug_count: int):
        super(SimpleConcatModel, self).__init__(drug_in_features, cell_in_features)
        self.k = max_drug_count
        self.out_features = self.k * drug_in_features + cell_in_features
        # self.batchnorm = nn.BatchNorm1d(self.out_features)

    def forward(self, all_drugs: List, cells):
        # all_drugs:
        #   [(d1 x drug_in_features), (d2 x drug_in_features), ..., (dB x drug_in_features)]
        # all di's should be less than or equal to max_drug_count
        # cells:
        #   (B x cell_in_features)
        B = len(all_drugs)
        all_padded = torch.stack([F.pad(d, (0, 0, 0, self.k - d.shape[0]), 'constant', 0) \
                                  for d in all_drugs])
        # all_padded.shape = (B * k * drug_in_features)
        all_padded = all_padded.reshape((B, -1))
        return torch.cat((all_padded, cells), dim=1)


class CellDrugAttentionModel(DrugCellIntegrationModel):
    def __init__(self, drug_in_features: int, cell_in_features: int, h: int, nheads: int):
        super(CellDrugAttentionModel, self).__init__(drug_in_features, cell_in_features)

        self.nheads = nheads
        self.attention_module = nn.MultiheadAttention(embed_dim=h * nheads,
                                                      num_heads=nheads,
                                                      kdim=drug_in_features,
                                                      vdim=drug_in_features,
                                                      batch_first=True)
        self.cell_query_fc = nn.Linear(cell_in_features, h * nheads)
        self.out_features = h * nheads + cell_in_features

    def forward(self, x_nodes, guide, cell):
        """
        x_nodes: all the nodes
        guide: A long tensor indicating each node belongs to which combination
        cells: B x cell
        """

        # Extract query, key, and value from drug and turn them into a batch_first sequence
        drug_key, mask = to_dense_batch(x_nodes, guide)

        # View cell lines as queries and drugs as key and value
        cell_query = self.cell_query_fc(cell.unsqueeze(1))
        cell_drug_integrated, _ = self.attention_module(query=cell_query,
                                                        key=drug_key,
                                                        value=drug_key,
                                                        need_weights=False,
                                                        key_padding_mask=torch.logical_not(mask)
                                                        )
        # B x M x h -> B x h
        cell_drug_integrated = cell_drug_integrated.squeeze()

        # Concatenate the max pooled feature vector from both sides
        integrated_all = torch.cat((cell_drug_integrated, cell), dim=1)
        return integrated_all


@INTEGRATION_REGISTRY.register('concatenation-method')
def build_concatenation_based_integration(cfg_hyper_parameters: ConfigurationNode, drug_in_features: int,
                                          cell_in_features: int):
    return SimpleConcatModel(
        drug_in_features=drug_in_features,
        cell_in_features=cell_in_features,
        max_drug_count=cfg_hyper_parameters.MAX_DRUG_COUNT
    )


@INTEGRATION_REGISTRY.register('multi-head-attention-method')
def build_attention_based_integration(cfg_hyper_parameters: ConfigurationNode, drug_in_features: int,
                                      cell_in_features: int):
    return CellDrugAttentionModel(
        drug_in_features=drug_in_features,
        cell_in_features=cell_in_features,
        nheads=cfg_hyper_parameters.N_HEADS,
        h=cfg_hyper_parameters.H
    )
