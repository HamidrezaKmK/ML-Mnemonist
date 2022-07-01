import sys

from config.config_drug import get_cfg_defaults
from mlmnemonist.validation_tools.cfg_utils import _expand_cfg, expand_cfg
from yacs.config import CfgNode as ConfigurationNode


def test1():
    cfg = ConfigurationNode()
    cfg.A = ConfigurationNode()
    cfg.B = ConfigurationNode()
    cfg.C = ConfigurationNode()
    cfg.A.A = ConfigurationNode()
    cfg.A.A.MLM_BRANCH_1 = 100
    cfg.A.A.MLM_BRANCH_2 = 200
    cfg.A.A.MLM_BRANCH_3 = 300
    cfg.A.B = ConfigurationNode()
    cfg.A.B.MLM_BRANCH_1 = 1000
    cfg.A.B.MLM_BRANCH_2 = 3000

    cfg.B.MLM_BRANCH_1 = ConfigurationNode()
    cfg.B.MLM_BRANCH_2 = ConfigurationNode()
    cfg.B.MLM_BRANCH_1.A = ConfigurationNode()
    cfg.B.MLM_BRANCH_1.A.MLM_BRANCH_1 = 111
    cfg.B.MLM_BRANCH_1.A.MLM_BRANCH_2 = 222
    cfg.B.MLM_BRANCH_2.T = ConfigurationNode()
    cfg.B.MLM_BRANCH_2.T.MLM_BRANCH_1 = 111
    cfg.B.MLM_BRANCH_2.T.MLM_BRANCH_2 = 222

    cfg.C.MLM_BRANCH_1 = 11
    cfg.C.MLM_BRANCH_2 = 13
    for cfg_i, code in _expand_cfg(cfg):
        print(f"-- {code} --")
        print(cfg_i)
        print("---")


def test2():
    sys.setrecursionlimit(10000)

    mp = expand_cfg(get_cfg_defaults(),
                    cfg_dir='conf-test-branches.yaml',
                    save_directory='all-confs')
    for key in mp.keys():
        print(key)
        print(mp[key])
        print("-------------------")


def test3():
    sys.setrecursionlimit(10000)

    mp = expand_cfg(get_cfg_defaults(),
                    cfg_dir='conf-drug-palette.yaml',
                    save_directory='config/drug-tuning')
    for key in mp.keys():
        print(key)
        print(mp[key])
        print("-------------------")


if __name__ == '__main__':
    test3()
