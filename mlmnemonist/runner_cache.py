import os
import shutil
import warnings

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List, Any
import pickle
from yacs.config import CfgNode as ConfigurationNode
import yacs
import yacs.config

MAX_CACHE_SIZE: int = 20


def get_all_tokens(directory: str) -> List[str]:
    all_names = set()
    for f in os.listdir(directory):
        if '-MLM-CACHE-TOK' not in f:
            continue
        name = f.split('-MLM-CACHE-TOK')[0]
        all_names.add(name)
    return list(all_names)


def get_new_token(directory: str, pref: Optional[str] = '') -> str:
    all_tokens = get_all_tokens(directory)
    mex = 0
    while f'{mex}{pref}' in all_tokens:
        mex += 1
    return f'{mex}{pref}'

def get_new_meta_token(directory: str) -> str:
    return get_new_token(directory, pref='-META')

class RunnerCache:

    def __init__(self, directory: str, token: Optional[str] = None):
        all_tokens = get_all_tokens(directory)

        if token not in all_tokens and len(all_tokens) >= MAX_CACHE_SIZE:
            raise Exception("Maximum cache limit reached!\n"
                            f"Remove files from {directory}")
        self._cache_token = token if token is not None else get_new_token(directory)
        self._cache_token += '-MLM-CACHE-TOK'

        self._cached_primitives: Dict[str, Any] = {}
        self._cached_models: Dict[str, nn.Module] = {}
        self._cached_configs: Dict[str, ConfigurationNode] = {}

        self.directory = directory
        # Create the logfile
        log_dir = os.path.join(self.directory, f'{self._cache_token}-logs')
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)

    @property
    def LOGS_DIR(self) -> str:
        return os.path.join(self.directory, f'{self._cache_token}-logs')

    @property
    def TOKEN(self) -> str:
        return self._cache_token.split('-MLM-CACHE-TOK')[0]

    def RESET(self, prompt=True) -> None:
        """
        This function removes all the cached data from disk.
        By defeault, it will prompt the user to ask and make sure they agree to this.
        However, one can use it by setting 'prompt = False' to avoid prompting.
        """
        while prompt:
            resp = input("This will remove all of the variables from your runner's cache ..."
                         "\nAre you sure you want to proceed? [y/n] ")
            if resp == 'y':
                break
            elif resp == 'n':
                return
            else:
                continue

        # Remove everything from the checkpoint
        # directory that starts with the runner prefix
        for f in os.listdir(self.directory):
            real_dir = os.path.join(self.directory, f)
            if f.startswith(self._cache_token):
                if os.path.isfile(real_dir):
                    os.remove(os.path.join(self.directory, f))
                elif f == f'{self._cache_token}-logs':
                    shutil.rmtree(real_dir)

        # Clear up the cached_primitives
        self._cached_primitives = {}
        self._cached_models = {}

    def SAVE(self) -> None:
        """
        Save the whole cache into the disk.
        """
        try:
            # Save the primitives
            with open(os.path.join(self.directory, f'{self._cache_token}-runner_checkpoint_primitives.pkl'), 'wb') as f:
                pickle.dump(self._cached_primitives, f)

            # Save all of the current models
            for name in self._cached_models.keys():
                torch.save(self._cached_models[name].state_dict(),
                           os.path.join(self.directory, f'{self._cache_token}-model.{name}.pth'))
            # Save all the configurations
            for name in self._cached_configs.keys():
                cfg = self._cached_configs[name]
                cfg.dump(stream=open(os.path.join(self.directory, f'{self._cache_token}-cfg.{name}.yaml'), 'w'))

        except KeyboardInterrupt as e:
            warnings.warn("No keyboard interrupt allowed in between cache saving ...")

    def SET_IFN(self, name: str, value: Any) -> Any:
        if name not in self._cached_primitives:
            self._cached_primitives[name] = value
        return self._cached_primitives[name]

    def SET(self, name: str, value: Any) -> Any:
        self._cached_primitives[name] = value
        return self._cached_primitives[name]

    def GET(self, name: str) -> Any:
        if name not in self._cached_primitives:
            raise ModuleNotFoundError(f"primitive with name {name} not found!")
        return self._cached_primitives[name]

    def SET_CFG(self, name: str, value: ConfigurationNode):
        self._cached_configs[name] = value
        return self._cached_configs

    def SET_IFN_CFG(self, name: str, value: Optional[ConfigurationNode]) -> Optional[ConfigurationNode]:
        if name not in self._cached_configs:
            self._cached_configs[name] = value
        return self._cached_configs[name]

    def GET_CFG(self, name: str):
        if name not in self._cached_configs:
            raise Exception(f"Config with name {name} not found!")
        return self._cached_configs[name]

    def SET_IFN_M(self, name: str, model: Optional[nn.Module]) -> Optional[nn.Module]:
        if '.' in name:
            raise NameError(f"Name should not contain dots {name}")
        if name not in self._cached_models:
            self._cached_models[name] = model
        return self._cached_models[name]

    def SET_M(self, name: str, model: nn.Module):
        if '.' in name:
            raise NameError(f"Name should not contain dots {name}")
        self._cached_models[name] = model
        return self._cached_models[name]

    def GET_M(self, name: str):
        if name not in self._cached_models:
            raise ModuleNotFoundError(f"model {name} not found!")
        return self._cached_models[name]

    def LOAD(self):
        for dir in os.listdir(self.directory):
            real_dir = os.path.join(self.directory, dir)
            if dir == f'{self._cache_token}-runner_checkpoint_primitives.pkl':
                # Handle loading the primitives
                with open(real_dir, 'rb') as f:
                    self._cached_primitives = pickle.load(f)
            elif dir.startswith(f'{self._cache_token}-model'):
                # Handle loading models
                model_name = real_dir.split('.')[-2]
                if model_name not in self._cached_models:
                    print(f"Model with name {model_name} has not been initialized in preprocess!")
                self._cached_models[model_name].load_state_dict(torch.load(real_dir))
            elif dir.startswith(f'{self._cache_token}-cfg'):
                cfg_name = real_dir.split('.')[-2]
                self._cached_configs[cfg_name] = yacs.config.load_cfg(open(real_dir, 'r'))
