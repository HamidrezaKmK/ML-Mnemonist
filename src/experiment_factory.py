import os
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

    def __init__(self,
                 experiment_dir: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 config_dir: Optional[str] = None,
                 config_default_builder: Optional[Callable[[], ConfigurationNode]] = None):
        """
        Loads data from .env and fills up the following:
        - EXPERIMENT_DIR=The directory containing the experiments
        - CHECKPOINT_DIR=Directory containing all the checkpoints
        - CONFIG_DIR=directory containing all the .yaml files
        """
        self.runner: ExperimentRunner
        self.cfg_builder = config_default_builder

        load_dotenv(find_dotenv(), verbose=True)  # Load .env

        all_args = [experiment_dir, 'EXPERIMENT_DIR',
                    checkpoint_dir, 'CHECKPOINT_DIR',
                    config_dir, 'CONFIG_DIR']
        for i in range(0, len(all_args), 2):
            if all_args[i] is None:
                all_args[i] = os.getenv(all_args[i + 1])
        if all_args[0] is None:
            raise RuntimeError("No experiment directory defined in constructor or .env!\n"
                               "Define in .env using EXPERIMENT_DIR=/PATH/TO/DIR")
        self.experiment_dir = os.path.join(all_args[0], 'mnemonic-experiments')
        if all_args[2] is None:
            raise RuntimeError("No checkpoint directory defined in constructor or .env!\n"
                               "Define in .env using CHECKPOINT_DIR=/PATH/TO/CHECKPOINTS")
        self.checkpoint_dir = os.path.join(all_args[2], '.mnemonic-checkpoints')

        self.config_dir = all_args[4]

    def _get_experiment_path(self, experiment_name: Optional[str]):
        """
        Get the full directory of the experiment, if experiment_name is not specified
        then come up with the currently unavailable name!
        """
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

    def create(self, experiment_name: Optional[str] = None, verbose: int = 0,
               cfg_dir: Optional[str] = None, cache_token: Optional[str] = None) -> ExperimentRunner:

        experiment_path = self._get_experiment_path(experiment_name)

        cfg_path = None
        if cfg_dir is not None:
            if self.config_dir is not None:
                cfg_path = os.path.join(self.config_dir, cfg_dir)
            else:
                cfg_path = cfg_dir

        self.runner = ExperimentRunner(experiment_dir=experiment_path,
                                       checkpoint_dir=self.checkpoint_dir, verbose=verbose,
                                       cache_token=cache_token, cfg_path=cfg_path, cfg_builder=self.cfg_builder)

        return self.runner


FACTORY = ExperimentRunnerFactory()
