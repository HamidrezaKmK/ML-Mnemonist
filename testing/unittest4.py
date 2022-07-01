from mlmnemonist.factory import FACTORY
from testing.config.config_drug import get_cfg_defaults
from testing.unittest2 import decoy_recurring, decoy_prep, decoy_run

if __name__ == '__main__':
    description = "This is unittest3 runner"
    try:
        runner = FACTORY.retrieve_experiment_runner('10')
    except FileNotFoundError as e:
        print("CREATING NEW!")
        runner = FACTORY.create_experiment_runner(verbose=4,
                                                  description="sth sth",
                                                  cfg_base=get_cfg_defaults(),
                                                  cache_token='10')
        runner.preprocessing_pipeline.update_function(decoy_prep)
        runner.recurring_pipeline.update_function(decoy_recurring)
        runner.implement_run(decoy_run)

    runner.preprocess()

    print("PREP DONE!")

    runner.run()