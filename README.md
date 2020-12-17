# BoXHED2.0Main

For information on the functionalities of BoXHED2.0, please refer to [BoXHED1.0 Paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) published in ICML 2020.

In order to install BoXHED2.0 locally, the steps are as follows:

1. **cloning the repos**: first BoXHED2.0Main and BoXHED2.0 need to be cloned. For doing so, you may run the following commands:
```
git clone https://github.com/BoXHED/BoXHED2.0.git
git clone https://github.com/BoXHED/BoXHED2.0Main.git
```
2. **setting up conda**: We highly recommend devoting a conda environment to BoXHED2.0. This implementation uses python 3.8. We have provided the package list of the conda environment used for the development in the repository under the name *packageslist.txt*. 

You may create a conda environment named boxhed2.0 and load it with the package list mentioend above. To do so, you can go to the cloned BoXHED2.0 repository, and run the following:
```
conda create -n boxhed2.0 --file packageslist.txt python=3.8
```
Having created the environment, you may run:
```
conda activate boxhed2.0
```
3. **BoXHED2.0 Installation**: This stage combines the two repositories, namely, BoXHED2.0 and BoXHED2.0Main. 

You need to provide the address of the cloned BoXHED2.0 in the file named *config.txt* in BoXHED2.0Main. It is set to *./BoxHED2.0* by default so if you have cloned both the repositories in the same folder (i.e. you have run the cloning lines of codes as presented) you do not need to change anything in *config.txt*.

Having set that, you may run:
```
python setup_boxhed.py build
```
3. **running BoXHED2.0**: we have included our synthetic data simulation as an example of how to use BoXHED2.0 in *boxhed_main.py*. You may run it as:
```
python boxhed_main.py
```
