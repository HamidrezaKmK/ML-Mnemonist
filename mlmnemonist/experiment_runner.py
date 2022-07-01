from __future__ import annotations

import copy
import os
import shutil
from typing import Optional, Callable, Any, Dict

from yacs.config import CfgNode as ConfigurationNode
from mlmnemonist.processing_pipeline import Pipeline
from mlmnemonist.runner_cache import RunnerCache


class ExperimentRunner:
    """
    This object type is the basic object you use to run an experiment.
    It is flexible in terms of how the experiment is implemented and it is created for interactive running of experiments.
    You can separate the dataloading process from you training or testing process and you can implement a run function
    using the factory method.

    Each experiment runner has an experiment configuration file associated with it that feeds the information needed
    to conduct an experiment. You can interactively reload the configuration file mid-experiment and re-run everything.
    You can also reload the data using the load_in_dataframe option.

    As an example, checkout the 'RandomForestRuntime.ipynb' in the notebooks section to checkout an experiment runner
    that runs cv_folds on a random forest.

    This object is coupled with a FACTORY and is better to be instantiated using the factory object defined in this
    file. That way, many of the configurations will be preset to the values in the .env.
    """

    def add_config(self,
                   cfg_base: Optional[ConfigurationNode] = None,
                   cfg_path: Optional[str] = None):
        self._has_cfg = False
        self.cfg_path = None
        if cfg_base is not None:
            self._has_cfg = True
            self.cfg = copy.deepcopy(cfg_base)
            if cfg_path is not None:
                self.cfg_path = cfg_path
                self.cfg.merge_from_file(self.cfg_path)

    def __init__(self, experiment_dir: str, checkpoint_dir: str, verbose: int = 0,
                 cache_token: Optional[str] = None, cfg_path: Optional[str] = None,
                 cfg_base: Optional[ConfigurationNode] = None,
                 secret_root: Optional[str] = None,
                 meta_cache: Optional[RunnerCache] = None) -> None:
        self._secret_root = secret_root

        self.add_config(cfg_base, cfg_path)

        self.verbose = verbose

        self.experiment_dir = experiment_dir
        if not os.path.exists(self.experiment_dir):
            raise FileNotFoundError(
                f"File {self.experiment_dir} not found! Maybe you have deleted the experiment ...\n"
                f"Make sure to delete everything in checkpoint directory {checkpoint_dir} too before re-running")

        self.checkpoint_dir = checkpoint_dir

        self._outputs: Dict[str, str] = {}

        self._implemented_run: Optional[Callable[[ExperimentRunner, ...], Any]] = None
        self._META_CACHE = meta_cache
        self.recurring_pipeline = Pipeline(self._META_CACHE,
                                           'RUNNER_RECURRING_PIPELINE_META')
        self.preprocessing_pipeline = Pipeline(self._META_CACHE,
                                               'RUNNER_PREP_PIPELINE_META')

        self.CACHE = RunnerCache(directory=checkpoint_dir, token=cache_token)

    def __str__(self) -> str:
        """
        return a string containing the type of runner
        their cache token,
        configuration (if available)
        the prepcrocessing pipeline
        the recurring pipeline
        and the run function (if implemented)
        """
        ret = f'Runner at {self.experiment_dir}\n'
        ret += f'\t - cache token: {self.CACHE.TOKEN}\n'

        if self._has_cfg:
            ret += f'\t - configurations at: {self.cfg_path}\n'
        else:
            ret += f'\t - no configuration file specified!'

        ret += f'\t - preprocessings functions {self.preprocessing_pipeline}\n'
        ret += f'\t - recurring pipeline {self.recurring_pipeline}\n'
        if self._implemented_run is None:
            ret += f'\t - Run function not implemented!'
        else:
            ret += f'\t - Run function: {self._implemented_run.__name__}\n'
        return ret

    def reveal_true_path(self, path: str) -> str:
        return os.path.join(self._secret_root, path)

    def merge_cfg(self, cfg_path: str) -> None:
        self.cfg_path = cfg_path
        self.reload_cfg()

    def reload_cfg(self) -> None:
        """
        Run this module whenever you have made a change to the cfg but you do not wish
        to redo all the previous steps on the runner. For example, when the data is loaded
        onto an experiment runner and some training configurations have changed, there is no
        need to reload the experiment runner and you can simply change the hyperparameters in
        the yaml file and reload.
        """
        if self._has_cfg and self.cfg_path is not None:
            self.cfg.merge_from_file(self.cfg_path)

    def export_logs(self) -> str:
        ret = os.path.join(self.experiment_dir, 'logs-export')
        shutil.make_archive(ret,
                            'zip',
                            self.CACHE.LOGS_DIR)
        if self.verbose > 0:
            print(f"Files being archived in {ret}.zip ...")
        return ret + '.zip'

    def preprocess(self, keep=True) -> None:
        """
        Run all the functions specified for preprocessing in an orderly fasion
        """
        if self.preprocessing_pipeline.function_count == 0 and self.verbose > 0:
            print("No functions in the preprocessing pipeline!")
        self.reload_cfg()
        self.preprocessing_pipeline.run(keep=keep, verbose=self.verbose, runner=self)

    def implement_run(self, run: Callable[[ExperimentRunner, ...], Any]) -> None:
        """
        Use this function to
        """
        self._implemented_run = run
        self._META_CACHE.SET('RUNNER_RUN_META', self._implemented_run)
        self._META_CACHE.SAVE()

    def ADD_OUTPUT(self, file_dir: str, description: str):
        self._outputs[file_dir] = description

    def run(self, *args, **kwargs):
        """
        This function runs an arbitrary method that is specified in the
        factory function. After running it, all the values of the config file plus a timer of the whole runtime
        will be stored in a .yaml file in the experiments directory. This will give us the possibility to re-run
        our experiments by simply specifying the output yaml file as input again.
        """
        if self._implemented_run is None:
            raise NotImplementedError("The run function is not implemented yet!")

        if self.verbose > 0 and self.recurring_pipeline.function_count == 0:
            print("No functions in the recurring pipeline ...")
        self.reload_cfg()
        self.recurring_pipeline.run(keep=True, verbose=self.verbose, runner=self)

        self.CACHE.LOAD()
        ret = self._implemented_run(self, *args, **kwargs)

        # Save files
        if self.verbose > 0:
            print("[DONE] running over!")
            print("\t - saving files ...")

        with open(os.path.join(self.experiment_dir, 'readme.txt'), 'w') as f:
            all_lines = ['This file contains a description on the files available in the experiment\n']
            # Save configurations
            if self._has_cfg and self.cfg is not None and self.cfg_path is not None:
                name = '.'.join(os.path.basename(self.cfg_path).split('.')[:-1])
                self.cfg.dump(
                    stream=open(os.path.join(self.experiment_dir, f'{name}-output.yaml'), 'w'))
                shutil.copyfile(self.cfg_path, os.path.join(self.experiment_dir, f'{name}-input.yaml'))
                all_lines.append(f"\t{name}-output.yaml : Contains the configurations after ending the runner.\n")
                all_lines.append(f"\t{name}-input.yaml : Contains the configurations when starting the runner.\n"
                                 f"\t You can feed the same file again as configurations and gain the same results.\n")
            # Save file descriptions in readme.txt
            all_lines.append("Output files and their descriptions:\n")
            all_lines += [f'\t{x} : {self._outputs[x]}\n' for x in self._outputs.keys()]
            f.writelines(all_lines)
        return ret
