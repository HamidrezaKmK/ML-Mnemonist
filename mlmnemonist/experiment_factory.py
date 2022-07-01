import copy
import os
import pickle
import shutil
import warnings
from typing import Optional, Callable
from datetime import date
from dotenv import load_dotenv, find_dotenv
from mlmnemonist.runner_cache import RunnerCache

from mlmnemonist.experiment_runner import ExperimentRunner
from yacs.config import CfgNode as ConfigurationNode


class ExperimentRunnerFactory:
    """
    An experiment runner factory that creates experiment runners and customizes them.

    Use the 'create' method first to create a generic runner and then add on new features to the runner.

    - load_combo_dataset: loads a dataset according to the config file given to it; this directly calls
                            ExperimentRunner.load_in_dataframe
    - load_cv_fold: loads the cv_folds according to the paths given to it in the environment. This function
                        directly calls ExperimentRunner.load_cv_fold
    - implement_run: This function
    """

    def __init__(self,
                 cfg_base: Optional[ConfigurationNode] = None,
                 experiment_dir: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 config_dir: Optional[str] = None,
                 secret_dir: Optional[str] = None):
        """
        Loads data from .env and fills up the following:
        - MLM_EXPERIMENT_DIR=The directory containing the experiments
        - MLM_CHECKPOINT_DIR=Directory containing all the checkpoints
        - MLM_CONFIG_DIR=directory containing all the .yaml files
        - MLM_SECRET_ROOT_DIR=secret prefix used for secret paths
        """
        self.cfg_base = copy.deepcopy(cfg_base)

        load_dotenv(find_dotenv(), verbose=True)  # Load .env

        all_args = [experiment_dir, 'MLM_EXPERIMENT_DIR',
                    checkpoint_dir, 'MLM_CHECKPOINT_DIR',
                    config_dir, 'MLM_CONFIG_DIR',
                    secret_dir, 'MLM_SECRET_ROOT_DIR']
        for i in range(0, len(all_args), 2):
            if all_args[i] is None:
                all_args[i] = os.getenv(all_args[i + 1])
        if all_args[0] is None:
            raise RuntimeError("No experiment directory defined in constructor or .env!\n"
                               "Define in .env using MLM_EXPERIMENT_DIR=/PATH/TO/DIR")
        self.experiment_dir = os.path.join(all_args[0], 'mnemonic-experiments')
        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)

        if all_args[2] is None:
            raise RuntimeError("No checkpoint directory defined in constructor or .env!\n"
                               "Define in .env using MLM_CHECKPOINT_DIR=/PATH/TO/CHECKPOINTS")
        self.checkpoint_dir = os.path.join(all_args[2], '.mnemonic-checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.config_dir = all_args[4]
        if self.config_dir is None:
            warnings.warn("MLM_CONFIG_DIR not defined!")
        self.secret_root = all_args[6]
        if self.secret_root is None:
            warnings.warn("MLM_SECRET_ROOT_DIR not defined!")

    def _get_new_experiment_path(self, experiment_name: Optional[str]):
        """
        Get the full directory of the experiment, if experiment_name is not specified
        then come up with the currently unavailable name!
        """
        # setup experiment name
        root_experiments_path = self.experiment_dir
        middle_name = 'exp' if experiment_name is None else experiment_name

        mex = 0
        while os.path.exists(os.path.join(root_experiments_path, f'{date.today()}-{middle_name}-{mex}')):
            mex += 1
        exp_name = f'{date.today()}-{middle_name}-{mex}'

        experiment_path = os.path.join(root_experiments_path, exp_name)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)
        return experiment_path

    def clear_caches(self, prompt=True):
        """
        :param prompt:
        Prompt the user for decision

        Deletes everything in the checkpoints directory
        """
        while prompt:
            resp = input(f"This will delete everything in {self.checkpoint_dir}! Are you sure? [y/n] ")
            if resp == 'n':
                return
            elif resp == 'y':
                break
        for f in os.listdir(self.checkpoint_dir):
            real_dir = os.path.join(self.checkpoint_dir, f)
            if os.path.isfile(real_dir):
                os.remove(real_dir)
            else:
                shutil.rmtree(real_dir)

    def retrieve(self, cache_token: str):
        """
        Based on the cache token available at the checkpoints directory
        it retrieves a runner with all its cached data using cache_token
        """

        cache = RunnerCache(directory=self.checkpoint_dir,
                            token=f'{cache_token}-META')
        cache.LOAD()
        runner_meta = cache.SET_IFN('RUNNER_CONSTRUCTOR_META', None)
        if runner_meta is None:
            raise FileNotFoundError(f"No runner with cache token {cache_token} found in {self.checkpoint_dir}")

        return ExperimentRunner(experiment_dir=runner_meta['experiment_dir'],
                                checkpoint_dir=self.checkpoint_dir,
                                verbose=runner_meta['verbose'],
                                cache_token=cache_token,
                                cfg_path=runner_meta['cfg_path'],
                                cfg_base=copy.deepcopy(self.cfg_base),
                                secret_root=self.secret_root)

    def create(self, verbose: int = 0,
               description: str = 'description not specified!',
               experiment_name: Optional[str] = None,
               cfg_dir: Optional[str] = None,
               cache_token: Optional[str] = None) -> ExperimentRunner:

        experiment_path = self._get_new_experiment_path(experiment_name)
        with open(os.path.join(experiment_path, 'DESCRIPTION.txt'), 'w') as f:
            f.write(description)

        cfg_path = None
        if cfg_dir is not None:
            if self.config_dir is not None:
                cfg_path = os.path.join(self.config_dir, cfg_dir)
            else:
                cfg_path = cfg_dir

        return ExperimentRunner(experiment_dir=experiment_path,
                                checkpoint_dir=self.checkpoint_dir, verbose=verbose,
                                cache_token=cache_token, cfg_path=cfg_path,
                                cfg_base=copy.deepcopy(self.cfg_base),
                                secret_root=self.secret_root)
