from mlmnemonist.factory import FACTORY
from testing.config.config_drug import get_cfg_defaults
from testing.unittest2 import decoy_recurring, decoy_prep, decoy_run

if __name__ == '__main__':
    description = "This is unittest3 runner"
    try:
        runner = FACTORY.retrieve_experiment_runner('0')
        runner.implement_run(decoy_run)
    except FileNotFoundError as e:
        runner = FACTORY.create_experiment_runner(verbose=4,
                                                  description="This is the runner for unittest 3",
                                                  cfg_base=get_cfg_defaults(),
                                                  cache_token='0')
        runner.preprocessing_pipeline.update_function(decoy_prep)
        runner.preprocess()
        runner.recurring_pipeline.update_function(decoy_recurring)
        runner.implement_run(decoy_run)

    print("--- Now running hyper-runner ---")

    try:
        hyper_runner = FACTORY.retrieve_hyper_experiment_runner('1')
    except FileNotFoundError as e:
        print("CREATING NEW HYPERRUNNER!")
        hyper_runner = FACTORY.create_hyper_experiment_runner(cfg_base=get_cfg_defaults(),
                                                              cfg_palette_path='conf-drug-palette.yaml',
                                                              experiment_runners=[runner],
                                                              verbose=1,
                                                              description='This is a hyperrunner for unittest 3',
                                                              cache_token='1'
                                                              )

    score_dict = hyper_runner.full_search()
    print(score_dict)
