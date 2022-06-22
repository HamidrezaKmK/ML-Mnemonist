import os

import torch
import torch.nn as nn
from typing import Optional, Union, Dict, List, Any
import random
import pickle


class RunnerCache:

    def __init__(self, directory: str, token: Optional[str] = None):
        self._cache_token = token if token is not None else f'tok-{random.randint(1, 300)}'
        self._cached_primitives: Dict[str, Any] = {}
        self._cached_models: Dict[str, nn.Module] = {}
        self.directory = directory

    @property
    def TOKEN(self) -> str:
        return self._cache_token

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
            if f.startswith(self._cache_token):
                os.remove(os.path.join(self.directory, f))

        # Clear up the cached_primitives
        self._cached_primitives = {}
        self._cached_models = {}

    def SAVE(self) -> None:
        """
        Save the whole cache into the disk.
        """
        # Save the primitives
        with open(os.path.join(self.directory, f'{self._cache_token}-runner_checkpoint_primitives.pkl'), 'wb') as f:
            pickle.dump(self._cached_primitives, f)

        # Save all of the current models
        for name in self._cached_models.keys():
            torch.save(self._cached_models[name].state_dict(),
                       os.path.join(self.directory, f'{self._cache_token}-model.{name}.pth'))

    def SET_IFN(self, name: str, value: Any):
        if name not in self._cached_primitives:
            self._cached_primitives[name] = value
        return self._cached_primitives[name]

    def SET(self, name: str, value: Any):
        self._cached_primitives[name] = value
        return self._cached_primitives[name]

    def GET(self, name: str):
        if name not in self._cached_primitives:
            raise ModuleNotFoundError(f"primitive with name {name} not found!")
        return self._cached_primitives[name]

    def SET_IFN_M(self, name: str, model: nn.Module):
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

    def _load_cache(self):
        for dir in os.listdir(self.directory):
            real_dir = os.path.join(self.directory, dir)
            if dir == f'{self._cache_token}-runner-checkpoint_primitives.pkl':
                # Handle loading the primitives
                with open(real_dir, 'rb') as f:
                    self._cached_primitives = pickle.load(f)
            elif dir.startswith(f'{self._cache_token}-model'):
                # Handle loading models
                model_name = real_dir.split('.')[-2]
                if model_name not in self._cached_models:
                    print(f"Model with name {model_name} has not been initialized in preprocess!")
                self._cached_models[model_name].load_state_dict(torch.load(real_dir))
