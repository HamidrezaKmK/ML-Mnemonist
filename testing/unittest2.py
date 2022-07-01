import random

from config.config_drug import get_cfg_defaults
from mlmnemonist import ExperimentRunnerFactory, ExperimentRunner
from mlmnemonist.validation_tools import grid_search, grid_search_from_palette
from testing.modeling.drug_combo import DEEP_DDR_REGISTRY


def decoy_prep(runner: ExperimentRunner):
    pass


def decoy_recurring(runner: ExperimentRunner):
    pass


def decoy_run(runner: ExperimentRunner):
    print("Running!")
    for i in range(10):
        print(i)


def config_based_run(runner: ExperimentRunner):
    model = DEEP_DDR_REGISTRY[runner.cfg.DEEP_DDR.MODEL.TYPE](runner.cfg.DEEP_DDR.MODEL.HYPER_PARAMETERS)
    return random.randint(0, 10)


if __name__ == '__main__':
    factory = ExperimentRunnerFactory(cfg_base=get_cfg_defaults())
    try:
        runner = factory.retrieve('unittest2')
    except FileNotFoundError:
        runner = factory.create(experiment_name='build-a-large',
                                verbose=1,
                                cache_token='unittest2')

    runner.preprocessing_pipeline.update_function(decoy_prep)
    runner.preprocess()
    runner.recurring_pipeline.update_function(decoy_recurring)
    # runner.implement_run(decoy_run)
    # runner.run()
    runner.implement_run(config_based_run)
    grid_search_from_palette(runner,
                             cache_token='chiz',
                             verbose=1,
                             save_directory='config/all-conf-drug-tuning',
                             cfg_base=get_cfg_defaults(),
                             cfg_palette_dir='conf-drug-palette.yaml')
