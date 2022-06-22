import functools
import os
import warnings
from typing import Optional, Callable

from dotenv import load_dotenv, find_dotenv

from src.experiment_runner import ExperimentRunner
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

    def __init__(self):
        """
        Loads data from .env and fills up the following:
        - DATA_DIR=The directory containing all the data
        - EXPERIMENT_DIR=The directory containing the experiments
        - CHECKPOINT_DIR=Directory containing all the checkpoints
        - CONFIG_DIR=directory containing all the .yaml files
        """
        self.runner: ExperimentRunner

        load_dotenv(find_dotenv(), verbose=True)  # Load .env

        self.data_dir = os.path.abspath(os.getenv('DATA_DIR'))
        self.experiment_dir = os.path.abspath(os.getenv('EXPERIMENT_DIR'))
        self.checkpoint_dir = os.path.abspath(os.getenv('CHECKPOINT_DIR'))
        self.config_dir = os.path.abspath(os.getenv('CONFIG_DIR'))

    def _get_experiment_path(self, experiment_name: Optional[str] = None):
        # setup experiment name
        root_experiments_path = self.experiment_dir
        exp_name = experiment_name
        if exp_name is None:
            mex = 0
            while os.path.exists(os.path.join(root_experiments_path, f'exp_{mex}')):
                mex += 1
            exp_name = f'exp_{mex}'
        experiment_path = os.path.join(root_experiments_path, exp_name)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)
        return experiment_path

    def create(self, cfg_dir: str, experiment_name: Optional[str] = None, verbose: int = 0,
               cache_token: Optional[str] = None) -> ExperimentRunner:

        experiment_path = self._get_experiment_path(experiment_name)
        # setup cfg
        cfg_path = os.path.join(self.config_dir, cfg_dir)
        self.runner = ExperimentRunner(data_dir=self.data_dir,
                                       cfg_path=cfg_path, experiment_dir=experiment_path,
                                       checkpoint_dir=self.checkpoint_dir, verbose=verbose,
                                       cache_token=cache_token)

        return self.runner


FACTORY = ExperimentRunnerFactory()
