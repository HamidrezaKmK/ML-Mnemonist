import torch.nn as nn

from testing.modeling.cell_embedding import CellEmbedder, CELL_EMBEDDER_REGISTRY
from testing.modeling.compound_embedding import CompoundEmbedder, COMPOUND_EMBEDDER_REGISTRY
from testing.modeling.fusion import DrugConcentrationFusion, FUSION_REGISTRY
from testing.modeling.integration import DrugCellIntegrationModel, INTEGRATION_REGISTRY
from testing.modeling.registry import Registry
from yacs.config import CfgNode as ConfigurationNode

DEEP_DDR_REGISTRY = Registry()


class DrugDoseResponseModel(nn.Module):
    def __init__(self,
                 compound_embedder: CompoundEmbedder,
                 cell_embedder: CellEmbedder,
                 drug_concentration_fusion: DrugConcentrationFusion,
                 drug_cell_integrator: DrugCellIntegrationModel,
                 dropout: float
                 ):
        super(DrugDoseResponseModel, self).__init__()
        if cell_embedder.out_features is None:
            raise AttributeError("cell embedder out feature dimension not specified!")

        if compound_embedder.in_features is None:
            raise AttributeError("molecule embedder out feature dumension no specified!")

        # TODO: add more exceptions for dimensions

        self.compound_embedder = compound_embedder
        self.cell_embedder = cell_embedder
        self.drug_cell_integrator = drug_cell_integrator
        self.drug_concentration_fusion = drug_concentration_fusion

        self.dropout_rate = dropout
        self.linear1 = nn.Linear(self.drug_cell_integrator.out_features, 1000, bias=False)
        self.batchnorm1 = nn.BatchNorm1d(1000)
        self.linear2 = nn.Linear(1000, 500, bias=False)
        self.batchnorm2 = nn.BatchNorm1d(500)
        self.linear3 = nn.Linear(500, 1)


class NoPoolingDDRM(DrugDoseResponseModel):
    def __init__(self, *args, **kwargs):
        super(NoPoolingDDRM, self).__init__(*args, **kwargs)


def prepare_models(cfg_hyper_parameters: ConfigurationNode):
    cell_embedder_type = cfg_hyper_parameters.CELL_EMBEDDER.TYPE
    cell_embedder_cfg = cfg_hyper_parameters.CELL_EMBEDDER.HYPER_PARAMETERS
    cell_embedder = CELL_EMBEDDER_REGISTRY[cell_embedder_type](cell_embedder_cfg)

    drug_embedder_type = cfg_hyper_parameters.DRUG_EMBEDDER.TYPE
    drug_embedder_cfg = cfg_hyper_parameters.DRUG_EMBEDDER.HYPER_PARAMETERS
    drug_embedder = COMPOUND_EMBEDDER_REGISTRY[drug_embedder_type](drug_embedder_cfg)

    fusion_type = cfg_hyper_parameters.FUSION.TYPE
    fusion_cfg = cfg_hyper_parameters.FUSION.HYPER_PARAMETERS
    fusion_model = FUSION_REGISTRY[fusion_type](fusion_cfg, drug_embedder.out_features)

    integration_type = cfg_hyper_parameters.INTEGRATION.TYPE
    integration_cfg = cfg_hyper_parameters.INTEGRATION.HYPER_PARAMETERS
    integration_model = INTEGRATION_REGISTRY[integration_type](integration_cfg, fusion_model.out_features,
                                                               cell_embedder.out_features)
    return cell_embedder, drug_embedder, fusion_model, integration_model


@DEEP_DDR_REGISTRY.register('deep-ddr-v1')
def build_deep_ddr(cfg_hyper_parameters: ConfigurationNode):
    cell_embedder, drug_embedder, fusion_model, integration_model = prepare_models(cfg_hyper_parameters)

    return DrugDoseResponseModel(compound_embedder=drug_embedder,
                                 cell_embedder=cell_embedder,
                                 drug_concentration_fusion=fusion_model,
                                 drug_cell_integrator=integration_model,
                                 dropout=cfg_hyper_parameters.DROPOUT)


@DEEP_DDR_REGISTRY.register('deep-ddr-v2')
def build_deep_ddr_large_vertex_set(cfg_hyper_parameters: ConfigurationNode):
    cell_embedder, drug_embedder, fusion_model, integration_model = prepare_models(cfg_hyper_parameters)

    return NoPoolingDDRM(
        compound_embedder=drug_embedder,
        cell_embedder=cell_embedder,
        drug_concentration_fusion=fusion_model,
        drug_cell_integrator=integration_model,
        dropout=cfg_hyper_parameters.DROPOUT
    )
