# ML-Mnemonist

ML-Mnemonist is an open-source lightweight framework for simple ML operations and it pairs up easily with free online cloud services that do not support session backup and recovery because the framework implements these in the hardware level. To that end, some of the tutorials here are purposefully created to work with Google Colab.
<p align="center">
<img src="/figures/logo.png"/>
</p>

Furthermore, ML-Mnemonist takes a systematic approach in defining entities known as experiments to track every ML operation. Configuration files are associated with each experiments and these configuration files are logged for later recreation of the experiments. In addition, ML-Mnemonist also introduces entities which we call "Hyper Experiments" that can run multiple experiments and compare them. This can be useful for systematic search over different hyper parameter settings with custom model selection algorithms.

For interactive Jupyter notebooks check out the [tutorials](/tutorial/) section that contains multiple use-cases of the functionalities introduced here. For an extensive documentation of the functionalities check out the [documentation](/tutorial/documentation.md) section.

## Installation
ML-Mnemonist is available as a PyPI package and you can simply install it using the following command:
```
pip install mlmnemonist
```
## License
ML-Mnemonist is released under [MIT License](/LICENCE.rst).