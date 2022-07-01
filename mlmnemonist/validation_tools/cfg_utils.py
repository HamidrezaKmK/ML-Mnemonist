import copy
import os
import sys

from dotenv import load_dotenv, find_dotenv
from yacs.config import CfgNode as ConfigurationNode
import re


def _is_branch_type(cfg: ConfigurationNode):
    """
    :param cfg:
    A ConfigurationNode
    :return:
    returns true if the children are of type MLM_BRANCH_i
    returns false otherwise
    """
    branch_count = 0
    for k in cfg.keys():
        if re.match(r'MLM_BRANCH_\d+', k):
            branch_count += 1

    if branch_count != 0 and branch_count != len(cfg.keys()):
        raise Exception(f"MLM branches incorrect at:\n{cfg}")
    return branch_count > 0


def _merge_cfg(cfg1: ConfigurationNode, cfg2: ConfigurationNode, key_list=None) -> ConfigurationNode:
    if key_list is None:
        key_list = []

    if not isinstance(cfg1, ConfigurationNode):
        return cfg2
    if not isinstance(cfg2, ConfigurationNode):
        return cfg1

    if _is_branch_type(cfg2):
        leaf = False
        if not isinstance(cfg1, ConfigurationNode):
            leaf = True
        base_cfg = copy.deepcopy(cfg1)
        cfg1 = ConfigurationNode()
        for k in cfg2.keys():
            if leaf:
                cfg1[k] = copy.deepcopy(cfg2[k])
            else:
                new_cfg = copy.deepcopy(base_cfg)
                new_cfg = _merge_cfg(new_cfg, cfg2[k], key_list)
                cfg1[k] = new_cfg
    else:
        for k in cfg2.keys():
            if not isinstance(cfg1, ConfigurationNode) or k not in cfg1:
                raise KeyError("Non-existent config key: {}".format(key_list + [k]))
            if not isinstance(cfg2[k], ConfigurationNode):
                cfg1[k] = copy.deepcopy(cfg2[k])
            else:
                cfg1[k] = _merge_cfg(cfg1[k], cfg2[k], key_list + [k])
    return cfg1


def _expand_cfg(cfg: ConfigurationNode, key_list=None):
    if key_list is None:
        key_list = []

    if not isinstance(cfg, ConfigurationNode):
        return [[copy.deepcopy(cfg), []]]
    elif not _is_branch_type(cfg):
        old_variants = []
        for k in cfg.keys():
            new_variants = _expand_cfg(cfg[k], key_list + [k])
            if len(old_variants) == 0:
                for cfg_, code in new_variants:
                    new_cfg = ConfigurationNode()
                    new_cfg[k] = copy.deepcopy(cfg_)
                    old_variants.append([new_cfg, code])
            else:
                accumulate = []
                for cfg_new, code_new in new_variants:
                    for cfg_old, code_old in old_variants:
                        cfg_combined = copy.deepcopy(cfg_old)
                        cfg_combined[k] = copy.deepcopy(cfg_new)
                        accumulate.append([cfg_combined, copy.deepcopy(code_old) + copy.deepcopy(code_new)])
                old_variants = accumulate
        return old_variants
    else:
        variants = []
        for i, k in enumerate(cfg.keys()):
            if isinstance(cfg[k], ConfigurationNode):
                new_variants = _expand_cfg(cfg[k], key_list+[k])
                for var in new_variants:
                    var[1] += [i]
                variants += new_variants
            else:
                variants.append([copy.deepcopy(cfg[k]), [i]])
        return variants


def expand_cfg(cfg_base, cfg_dir: str, save_directory: str):

    load_dotenv(find_dotenv(), verbose=True)  # Load .env
    cfg_dir = os.path.join(os.getenv('MLM_CONFIG_DIR'), cfg_dir)
    save_directory = os.path.join(os.getenv('MLM_EXPERIMENT_DIR'), save_directory)

    with open(cfg_dir, "r") as f:
        other_cfg = cfg_base.load_cfg(f)
    grid_cfg = copy.deepcopy(cfg_base)
    grid_cfg = _merge_cfg(grid_cfg, other_cfg)

    variants = _expand_cfg(grid_cfg)
    if not os.path.exists(save_directory):
        sv_dir = os.getcwd()
        parent_dir, dir_name = os.path.split(save_directory)
        parent_dir = os.path.join(parent_dir)
        os.chdir(parent_dir)
        os.mkdir(dir_name)
        os.chdir(sv_dir)

    mx_code = 1
    for _, code in variants:
        mx_code = max(mx_code, len(code))

    ret = {}
    for cfg, code in variants:
        name = '-'.join([str(c) for c in code])
        for _ in range(mx_code - len(code)):
            name += '-0'
        pth = os.path.join(save_directory, f'{name}-MLM-{os.path.basename(cfg_dir)}')
        ret[name] = pth
        cfg.dump(stream=open(pth, 'w'))
    return ret
