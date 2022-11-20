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
        """
        :param cfg_palette_dir: Path to the directory containing the configuration palettes
        :param cfg_base: Base configuration node that will be used to merge the configuration palettes
        :param hyper_experiment_path: Path to the directory where the hyper-experiment will store results
        :param experiment_runners: List of experiment runners that will be used to run the hyper-experiment
        :param checkpoint_dir: Path to the directory where the hyper-experiment will store cached data regarding itself
        :param cache_token: Token to be used to identify the hyper-experiment in the cache
        :param verbose: Verbosity level
        :param secret_root: The secret root that is being used per machine
        """
        self.cfg_base = cfg_base
        self._secret_root = secret_root
        self.hyper_experiment_path = hyper_experiment_path
        if not os.path.exists(self.hyper_experiment_path):
            raise FileNotFoundError(
                f"File {self.hyper_experiment_path} not found! Maybe you have deleted the experiment ...\n"
                f"Make sure to delete everything in checkpoint directory {checkpoint_dir} too before re-running")
        # Store all the configurations to run in a directory under hyper_experiment_path
        self.all_cfgs_dir = os.path.join(self.hyper_experiment_path, 'all-cfgs')
        if not os.path.exists(self.all_cfgs_dir):
            os.mkdir(self.all_cfgs_dir)
        expand_cfg(cfg_base, cfg_palette_dir, self.all_cfgs_dir)

        # raise exception if experiment_runners has length larger than 1
        if len(experiment_runners) > 1:
            raise NotImplementedError("Currently only one experiment runner is supported")
        self.runners = experiment_runners
        self.verbose = verbose
        self.checkpoint_dir = checkpoint_dir
        
        # Create a cache for the hyper-experiment
        self.CACHE = RunnerCache(self.checkpoint_dir, f'{cache_token}')

    def reveal_true_path(self, path: str) -> str:
        return os.path.join(self._secret_root, path)

    def preprocess(self, keep: bool = True) -> None:
        # run preprocess for all runners
        for runner in self.runners:
            runner.preprocess(keep=keep)

    def get_runner(self, runner_id: int = 0) -> ExperimentRunner:
        """
        Get an available experiment runner
        """
        return self.runners[runner_id]

    def run_code(self, code: str, with_preprocess: bool = False,
            exception_list: Tuple = (), *args, **kwargs):
        """
        Run a specific configuration on one of the runners that is available
        The runner might encounter an internal exception, in which case it will
        be caught from exception_list and ignored.

        This function updates the `score_dict` value available in the cache
        and returns the score of the configuration.
        """
        # Get the path to the configuration file
        all_codes = get_all_codes(self.all_cfgs_dir)
        cfg_path = all_codes[code]
        # Run the experiment
        runner = self.get_runner()
        runner.merge_cfg(cfg_path)
        # Setup the outputs
        current_experiment_output_dir = os.path.join(self.hyper_experiment_path, f'exp-{code}')
        if not os.path.exists(current_experiment_output_dir):
            os.mkdir(current_experiment_output_dir)
        runner.set_output_dir(current_experiment_output_dir)

        # Run the experiment
        if self.verbose > 0:
            print("---")
            print("This the configuration that will be used:")
            print(runner.cfg)
            print("---")
        runner.verbose = max(0, self.verbose - 1)
        try:
            if with_preprocess:
                runner.preprocess()
            score = runner.run(*args, **kwargs)
        except exception_list as e:
            warnings.warn(f"Exception caught {e}")
            score = None
        self.CACHE.LOAD()
        score_dict = self.CACHE.SET_IFN('score_dict', {})
        score_dict[code] = score
        self.CACHE.SET('score_dict', score_dict)
        self.CACHE.SAVE()
        runner.CACHE.RESET(prompt=False)
        return score


    def save_score_dict(self) -> None:
        """
        Save the score dictionary to the hyper-experiment directory
        """
        self.CACHE.LOAD()
        score_dict = self.CACHE.GET('score_dict')
        with open(os.path.join(self.hyper_experiment_path, 'score_dict.json'), 'w') as f:
            json.dump(score_dict, f)

    def full_search(self,
                    exception_list: Tuple = (),
                    with_preprocess=False,
                    *args, **kwargs) -> None:
        """
        :param exception_list: Some of the configurations might have inhenerent
        problems that will cause the experiment to crash. This list contains
        the codes of the configurations that should be skipped.

        :param with_preprocess: Whether to run the preprocess function before
        running the experiment. This is useful if the preprocess function
        generates some data that is needed for the experiment and we don't
        want to run the preprocess function again for each and every runner.

        :param args: Arguments to be passed to the experiment runner(s)
        :param kwargs: Keyword arguments to be passed to the experiment runner(s)
        """

        # Setup remaining codes
        self.CACHE.LOAD()
        iteration_i = self.CACHE.SET_IFN('iteration_i', 0)
        codes_dict = get_all_codes(self.all_cfgs_dir)
        all_codes = list(codes_dict.keys())
        self.CACHE.SAVE()

        # Iterate over all the codes and run the experiment while saving the end result in a json file
        while iteration_i < len(all_codes):
            current_code = all_codes[iteration_i]
            if self.verbose > 0:
                print(f"Iteration no. [{iteration_i + 1}/{len(all_codes)}]")
            self.run_code(current_code, with_preprocess=with_preprocess,exception_list=exception_list, *args, **kwargs)
            self.save_score_dict()
            iteration_i += 1
            self.CACHE.SET('iteration_i', iteration_i)
            self.CACHE.SAVE()

    def _grid_search(self, with_preprocess: bool = False, *args, **kwargs):
        # This version is unstable and probably buggy!
        return _run_grid_search(self.get_runner(),
                                self.CACHE,
                                self.verbose,
                                self.cfg_base,
                                self.all_cfgs_dir,
                                with_preprocess, *args, **kwargs)
