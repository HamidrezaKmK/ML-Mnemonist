import json
import os
import warnings
from typing import List, Optional, Dict, Any, Tuple

from mlmnemonist.runner_cache import RunnerCache
from yacs.config import CfgNode as ConfigurationNode

from mlmnemonist import ExperimentRunner
from mlmnemonist.validation_tools import expand_cfg
from mlmnemonist.validation_tools.grid_search import _run_grid_search
from mlmnemonist.validation_tools.utils import get_all_codes


class HyperExperimentRunner:
    def __init__(self,
                 cfg_palette_dir: str,
                 cfg_base: ConfigurationNode,
                 hyper_experiment_path: str,
                 experiment_runners: List[ExperimentRunner],
                 checkpoint_dir: str,
                 cache_token: Optional[str] = None,
                 verbose: Optional[int] = 0,
                 secret_root: Optional[str] = None
                 ):
        self.cfg_base = cfg_base
        self._secret_root = secret_root
        self.hyper_experiment_path = hyper_experiment_path
        if not os.path.exists(self.hyper_experiment_path):
            raise FileNotFoundError(
                f"File {self.hyper_experiment_path} not found! Maybe you have deleted the experiment ...\n"
                f"Make sure to delete everything in checkpoint directory {checkpoint_dir} too before re-running")
        self.all_cfgs_dir = os.path.join(self.hyper_experiment_path, 'all-cfgs')
        if not os.path.exists(self.all_cfgs_dir):
            os.mkdir(self.all_cfgs_dir)
        expand_cfg(cfg_base, cfg_palette_dir, self.all_cfgs_dir)

        self.runners = experiment_runners
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir

        self.CACHE = RunnerCache(self.checkpoint_dir, f'{cache_token}')

    def reveal_true_path(self, path: str) -> str:
        return os.path.join(self._secret_root, path)

    def preprocess(self, keep: bool = True) -> None:
        for runner in self.runners:
            runner.preprocess(keep=keep)

    def full_search(self,
                    exception_list: Tuple = (),
                    with_preprocess=False,
                    *args, **kwargs) -> Dict[str, Any]:
        # Setup remaining codes
        self.CACHE.LOAD()
        all_codes = self.CACHE.SET_IFN('all_codes', None)
        iteration_i = self.CACHE.SET_IFN('iteration_i', 0)
        codes_dict = self.CACHE.SET_IFN('codes_dict', None)
        score_dict = self.CACHE.SET_IFN('score_dict', {})
        if all_codes is None:
            codes_dict = get_all_codes(self.all_cfgs_dir)
            all_codes = list(codes_dict.keys())
        self.CACHE.SET('all_codes', all_codes)
        self.CACHE.SET('iteration_i', iteration_i)
        self.CACHE.SET('codes_dict', codes_dict)
        self.CACHE.SET('score_dict', score_dict)
        self.CACHE.SAVE()

        # Iterate over all the remaining codes
        while iteration_i < len(all_codes):
            current_code = all_codes[iteration_i]
            self.runners[0].merge_cfg(codes_dict[current_code])
            print("---")
            print(self.runners[0].cfg)
            print("---")

            self.runners[0].verbose = max(0, self.verbose - 1)
            if self.verbose > 0:
                print(f"Iteration no. [{iteration_i + 1}/{len(all_codes)}] "
                      f"-- Running {os.path.split(self.runners[0].cfg_path)[-1]}")
            try:
                if with_preprocess:
                    self.runners[0].preprocess()
                score = self.runners[0].run(*args, **kwargs)
            except exception_list as e:
                warnings.warn(f"Exception caught {e}")
                score = None
            score_dict[codes_dict[current_code]] = score
            self.runners[0].CACHE.RESET(prompt=False)
            if self.verbose > 0:
                print("Final score:", score)

            with open(os.path.join(self.all_cfgs_dir, 'seen_scores.json'), 'w') as f:
                json.dump(score_dict, f, indent=4)

            iteration_i += 1

            self.CACHE.SET('all_codes', all_codes)
            self.CACHE.SET('iteration_i', iteration_i)
            self.CACHE.SET('codes_dict', codes_dict)
            self.CACHE.SET('score_dict', score_dict)
            self.CACHE.SAVE()

        return score_dict

    def grid_search(self, with_preprocess: bool = False, *args, **kwargs):
        return _run_grid_search(self.runners[0],
                                self.CACHE,
                                self.verbose,
                                self.cfg_base,
                                self.all_cfgs_dir,
                                with_preprocess, *args, **kwargs)
