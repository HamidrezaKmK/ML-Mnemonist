
## Experiments


ML-Mnemonist uses concepts such as experiments that have a config file associated with them and objects such as `ExperimentRunner`s are able to run this experiment while doing systematic caching in the software-level. This caching feature helps work with services such as Google Colab that do not provide seamless backup and recovery and a session might get disconnected alongside all of the progress with it.


## Hyper Experiments

Additionally, ML-Mnemonist facilitates objects called `HyperExperimentRunners` that can be used to run multiple experiments. For tasks such as hyper-parameter tuning, these hyper-experiment runners get a meta-config file and run experiments according to each configuration. 
