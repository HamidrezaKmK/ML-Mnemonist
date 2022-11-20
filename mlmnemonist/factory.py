import copy
import os
import pickle
import shutil
import warnings
from typing import Optional, Callable, List
from datetime import date
from dotenv import load_dotenv, find_dotenv
from mlmnemonist.runner_cache import RunnerCache, get_all_tokens, get_new_meta_token, get_new_token

from mlmnemonist.experiment_runner import ExperimentRunner
from yacs.config import CfgNode as ConfigurationNode

from mlmnemonist.validation_tools.hyper_experiment_runner import HyperExperimentRunner


def _get_new_experiment_path(root_experiments_path, experiment_name: Optional[str], overwrite: bool):
    """
    Get the full directory of the experiment, if experiment_name is not specified
    then come up with the currently unavailable name!
    """
    # setup experiment name
    if experiment_name is None:
        overwrite = False
        middle_name = 'exp'
    else:
        middle_name = experiment_name

    # See if you want to overwrite or not and if not find a unique name via mex
    if overwrite is False:
        mex = 0
        while os.path.exists(os.path.join(root_experiments_path, f'{date.today()}-{middle_name}-{mex}')):
            mex += 1
        exp_name = f'{date.today()}-{middle_name}-{mex}'
    else:
        exp_name = f'{date.today()}-{middle_name}'

    experiment_path = os.path.join(root_experiments_path, exp_name)
    if not os.path.exists(experiment_path):
        os.mkdir(experiment_path)
    return experiment_path


class RunnerFactory:
    """
    An experiment runner and hyper experiment runner factory that creates experiment runners, 
    hyper-experiment runners, customizes them, and keeps track of their cached values

    This class is a singleton and FACTORY is the only instance of it which is defined here

    - Use the create method to create a new experiment runner or hyper experiment runner
    - Use the retrieve method to retrieve an experiment runner or hyper experiment from the cache
    """

    _instance_count: int = 0

    def __init__(self,
                 experiment_dir: Optional[str] = None,
                 checkpoint_dir: Optional[str] = None,
                 config_dir: Optional[str] = None,
                 secret_dir: Optional[str] = None,
                 override_singleton: bool = False):

        """
        Loads data from .env and fills up the following:
        - MLM_EXPERIMENT_DIR=The directory containing the experiments
        - MLM_CHECKPOINT_DIR=Directory containing all the checkpoints
        - MLM_CONFIG_DIR=directory containing all the .yaml files
        - MLM_SECRET_ROOT_DIR=secret prefix used for secret paths
        You can also create an instance of factory by passing the above arguments and this will override the .env file
        """
        if RunnerFactory._instance_count > 0 and not override_singleton:
            raise Exception(
                "Singleton class RunnerFactory cannot be instantiated more than once!")
        RunnerFactory._instance_count += 1

        load_dotenv(find_dotenv(), verbose=True)  # Load .env

        all_args = [experiment_dir, 'MLM_EXPERIMENT_DIR',
                    checkpoint_dir, 'MLM_CHECKPOINT_DIR',
                    config_dir, 'MLM_CONFIG_DIR',
                    secret_dir, 'MLM_SECRET_ROOT_DIR']
                    
        for i in range(0, len(all_args), 2):
            if all_args[i] is None:
                all_args[i] = os.getenv(all_args[i + 1])
        # all_args[2i] contains the actual value for the i-th argument

        if all_args[0] is None:
            raise RuntimeError("No experiment directory defined in constructor or .env!\n"
                               "Define in .env using MLM_EXPERIMENT_DIR=/PATH/TO/DIR")
        self.experiment_dir = os.path.join(all_args[0], 'mnemonic-experiments')
        self.hyper_experiment_dir = os.path.join(
            all_args[0], 'mnemonic-hyper-experiments')

        if not os.path.exists(self.experiment_dir):
            os.mkdir(self.experiment_dir)
        if not os.path.exists(self.hyper_experiment_dir):
            os.mkdir(self.hyper_experiment_dir)

        if all_args[2] is None:
            raise RuntimeError("No checkpoint directory defined in constructor or .env!\n"
                               "Define in .env using MLM_CHECKPOINT_DIR=/PATH/TO/CHECKPOINTS")
        self.checkpoint_dir = os.path.join(
            all_args[2], '.mnemonic-checkpoints')
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

        self.config_dir = all_args[4]
        if self.config_dir is None:
            warnings.warn("MLM_CONFIG_DIR not defined!")
        self.secret_root = all_args[6]
        if self.secret_root is None:
            warnings.warn("MLM_SECRET_ROOT_DIR not defined!")

    def delete_experiment(self, experiment_name: str):
        shutil.rmtree(os.path.join(self.experiment_dir, experiment_name))

    def delete_hyper_experiment(self, hyper_experiment_name: str):
        shutil.rmtree(os.path.join(
            self.hyper_experiment_dir, hyper_experiment_name))

    def cached_runners(self):
        """
        Returns a list of tokens of all the cached runners
        """
        all_tokens = []
        for f in os.listdir(self.checkpoint_dir):
            token = f.split('_')[0]
            all_tokens.append(token)  
        return list(set(all_tokens))
            
    def delete_from_cache(self, cache_prefix: str):
        for f in os.listdir(self.checkpoint_dir):
            real_dir = os.path.join(self.checkpoint_dir, cache_prefix)
            if f.startswith(cache_prefix):
                if os.path.isfile(real_dir):
                    os.remove(real_dir)
                else:
                    shutil.rmtree(real_dir)

    def clear_caches(self, prompt=True):
        """
        :param prompt:
        Prompt the user for decision

        Deletes everything in the checkpoints directory
        """
        while prompt:
            resp = input(
                f"This will delete everything in {self.checkpoint_dir}! Are you sure? [y/n] ")
            if resp == 'n':
                return
            elif resp == 'y':
                break
        self.delete_from_cache('')

    def retrieve_experiment_runner(self, cache_token: str) -> ExperimentRunner:
        """
        Based on the cache token available at the checkpoints directory
        it retrieves a runner with all its cached data using cache_token
        """

        if f'{cache_token}-META' not in get_all_tokens(self.checkpoint_dir):
            raise FileNotFoundError(
                f"No runner with cache token {cache_token} found in {self.checkpoint_dir}")
        meta_cache = RunnerCache(directory=self.checkpoint_dir,
                                 token=f'{cache_token}-META')

        meta_cache.LOAD()

        meta = meta_cache.SET_IFN('RUNNER_CONSTRUCTOR_META', None)
        cfg_base = meta_cache.SET_IFN_CFG('RUNNER_CFG_BASE_META', None)

        runner = ExperimentRunner(experiment_name=meta['experiment_name'],
                                  experiment_dir=meta['experiment_dir'],
                                  checkpoint_dir=self.checkpoint_dir,
                                  verbose=meta['verbose'],
                                  cache_token=cache_token,
                                  cfg_path=meta['cfg_path'],
                                  cfg_base=cfg_base,
                                  secret_root=self.secret_root,
                                  meta_cache=meta_cache)

        preprocessing_pipeline = meta_cache.SET_IFN(
            'RUNNER_PREP_PIPELINE_META', [])
        recurring_pipeline = meta_cache.SET_IFN(
            'RUNNER_RECURRING_PIPELINE_META', [])
        run_func = meta_cache.SET_IFN('RUNNER_RUN_META', None)
        for func in preprocessing_pipeline:
            runner.preprocessing_pipeline.update_function(func)
        for func in recurring_pipeline:
            runner.recurring_pipeline.update_function(func)
        if run_func is not None:
            runner.implement_run(run_func)
        return runner

    def create_experiment_runner(self,
                                 experiment_name: Optional[str] = None,
                                 description: str = 'description not specified!',
                                 cfg_base: Optional[ConfigurationNode] = None,
                                 cfg_dir: Optional[str] = None,
                                 cache_token: Optional[str] = None,
                                 verbose: int = 0,
                                 overwrite: bool = True) -> ExperimentRunner:
        """
        This function should be used to create experiment runners
        """
        # Get a new path and assign it to the experiment, all logs will be stored in it
        experiment_path = _get_new_experiment_path(
            self.experiment_dir, experiment_name, overwrite=overwrite)
        experiment_name = os.path.basename(experiment_path)

        # Create a description file and store in the experiment's path
        with open(os.path.join(experiment_path, 'DESCRIPTION.txt'), 'w') as f:
            f.write(description)

        # Store the metadata used to construct the experiment runner
        cache_token_meta = f'{cache_token}-META' if cache_token is not None else get_new_meta_token(
            self.checkpoint_dir)
        meta_cache = RunnerCache(
            directory=self.checkpoint_dir, token=cache_token_meta)
        meta_cache.LOAD()

        # Create a string called cfg_path which contains the configuration directory
        # This string might be set to None which means that the experiment does not take
        # in any configurations
        cfg_path = None
        if cfg_dir is not None:
            if self.config_dir is not None:
                cfg_path = os.path.join(self.config_dir, cfg_dir)
            else:
                cfg_path = cfg_dir

        # Construct the experiment runner itself and store it to return ultimately
        ret = ExperimentRunner(experiment_name=experiment_name,
                               experiment_dir=experiment_path,
                               checkpoint_dir=self.checkpoint_dir,
                               verbose=verbose,
                               cache_token=cache_token if cache_token is not None else get_new_token(
                                   self.checkpoint_dir),
                               cfg_path=cfg_path,
                               cfg_base=None if cfg_base is None else copy.deepcopy(
                                   cfg_base),
                               secret_root=self.secret_root,
                               meta_cache=meta_cache)

        # Store the fields used for the constructor in the cache as a dictionary
        meta = {
            'experiment_name': ret.experiment_name,
            'experiment_dir': ret.experiment_dir,
            'verbose': ret.verbose,
            'cfg_path': ret.cfg_path
        }
        meta_cache.SET('RUNNER_CONSTRUCTOR_META', meta)
        if ret.get_cfg() is not None:
            # Store the configuration file itself
            meta_cache.SET_CFG('RUNNER_CFG_BASE_META', ret.get_cfg())

        meta_cache.SAVE()
        return ret

    def retrieve_hyper_experiment_runner(self, cache_token: str) -> HyperExperimentRunner:
        cache = RunnerCache(directory=self.checkpoint_dir,
                            token=f'{cache_token}-HYPER-META')
        cache.LOAD()
        meta = cache.SET_IFN('HYPER_CONSTRUCTOR_META', None)
        cfg_base = cache.SET_IFN_CFG('CFG_BASE_HYPER_META', None)

        if meta is None or cfg_base is None:
            raise FileNotFoundError(
                f"Metadata for hyper runner not found in {self.checkpoint_dir}")

        # Retrieve all the experiment runners using the cache tokens that have been
        # saved in the meta-cache
        runners = []
        for runner_tokens in meta['runner_cache_tokens']:
            runners.append(self.retrieve_experiment_runner(runner_tokens))

        return HyperExperimentRunner(
            cfg_palette_dir=meta['cfg_palette_dir'],
            cfg_base=cfg_base,
            hyper_experiment_path=meta['hyper_experiment_path'],
            cache_token=f'{cache_token}-HYPER',
            verbose=meta['verbose'],
            secret_root=self.secret_root,
            checkpoint_dir=self.checkpoint_dir,
            experiment_runners=runners
        )

    def create_hyper_experiment_runner(self,
                                       cfg_base: ConfigurationNode,
                                       cfg_palette_path: str,
                                       experiment_runners: List[ExperimentRunner],
                                       verbose: int = 0,
                                       description: str = 'description of hyper experiment not specified!',
                                       cache_token: Optional[str] = None,
                                       experiment_name: Optional[str] = None,
                                       overwrite: bool = True) -> HyperExperimentRunner:
        # Get a new path and assign it to the experiment, all logs will be stored in it
        hyper_experiment_path = _get_new_experiment_path(
            self.hyper_experiment_dir, experiment_name, overwrite=overwrite)
        experiment_name = os.path.basename(hyper_experiment_path)

        # Create a description file and store in the experiment's path
        with open(os.path.join(hyper_experiment_path, 'DESCRIPTION.txt'), 'w') as f:
            f.write(description)

        # Store the metadata used to construct the experiment runner
        cache_token = cache_token if cache_token is not None else get_new_token(self.checkpoint_dir)
        cfg_palette_path = os.path.join(self.config_dir, cfg_palette_path)
        ret = HyperExperimentRunner(cfg_palette_dir=cfg_palette_path,
                                    cfg_base=cfg_base,
                                    hyper_experiment_path=hyper_experiment_path,
                                    cache_token=f'{cache_token}-HYPER',
                                    verbose=verbose,
                                    secret_root=self.secret_root,
                                    checkpoint_dir=self.checkpoint_dir,
                                    experiment_runners=experiment_runners)
        meta_data = {
            'hyper_experiment_path': hyper_experiment_path,
            'cfg_palette_dir': cfg_palette_path,
            'verbose': verbose,
            'runner_cache_tokens': [x.CACHE.TOKEN for x in experiment_runners]
        }
        meta_cache = RunnerCache(
            directory=ret.checkpoint_dir, token=f'{ret.CACHE.TOKEN}-META')
        meta_cache.SET('HYPER_CONSTRUCTOR_META', meta_data)
        meta_cache.SET_CFG('CFG_BASE_HYPER_META', cfg_base)
        meta_cache.SAVE()
        return ret


FACTORY = RunnerFactory()
