
from yacs.config import CfgNode as ConfigurationNode

# YACS overwrite these settings using YAML
__C = ConfigurationNode()

# Dummy dataset variables only for testing
__C.DATASET = ConfigurationNode()
__C.DATASET.TRAIN_NAME = 'not-specified!'
__C.DATASET.TEST_NAME = 'not-specified!'

__C.SOLVER = ConfigurationNode()
__C.SOLVER.LR = 0.0
__C.SOLVER.OPTIMIZER_TYPE = 'not-specified!'
__C.SOLVER.DEVICE = 'not-specified!'
__C.SOLVER.METHOD = 'not-specified!'
__C.SOLVER.BATCH_SIZE = -1

__C.MODEL = ConfigurationNode()
__C.MODEL.HYPER_PARAMETERS = ConfigurationNode()
__C.MODEL.HYPER_PARAMETERS.IN_FEATURES = -1
__C.MODEL.HYPER_PARAMETERS.H1 = -1
__C.MODEL.HYPER_PARAMETERS.H2 = -1


def get_cfg_defaults() -> ConfigurationNode:
    """
    Get a yacs CfgNode object with default values
    """
    # Return a clone so that the defaults will not be altered
    # It will be subsequently overwritten with local YAML.
    return __C.clone()
