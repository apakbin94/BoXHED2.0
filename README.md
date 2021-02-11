# BoXHED2.0Main

For information on the functionalities of BoXHED2.0, please refer to [BoXHED1.0 Paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) published in ICML 2020.

In order to install BoXHED2.0 locally, the steps are as follows:

1. **cloning the repo**: first BoXHED2.0Main needs to be cloned. For doing so, you may run the following command:
```
git clone --recursive https://github.com/BoXHED/BoXHED2.0Main.git
```
2. **setting up conda**: We highly recommend devoting a conda environment to BoXHED2.0. This step makes sure BoXHED2.0 will not interfere with XGBoost (the library we have borrowed from extensively) when installed. This implementation uses python 3.8. You may create a conda environment named boxhed2.0. To do so, you can go to the cloned BoXHED2.0Main repository, and run the following:
```
conda create -n boxhed2.0 python=3.8
```
Having created the environment, you may run:
```
conda activate boxhed2.0
```
3. **BoXHED2.0 Installation**: You may run the setup bash script by:
```
bash setup.sh
```
3. **running BoXHED2.0**: we have included our synthetic data simulation as an example of how to use BoXHED2.0 in *main.py*. You may run it as:
```
python main.py
```
