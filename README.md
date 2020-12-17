# BoXHED2.0Main

For information on the functionalities of BoXHED2.0, please refer to [BoXHED1.0 Paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) published in ICML 2020.

In order to install BoXHED2.0 locally, the steps are as follows:

1. **cloning the repos**: first BoXHED2.0Main and BoXHED2.0 need to be cloned. For doing so, you may run the following command:
```
git clone *REPO_LINK*
```
please note that it needs to be done for both repositories.
2. **setting up conda**: We highly recommend devoting a conda environment to BoXHED2.0. This implementation uses python 3.8. We have provided the package list of the conda environment used for the development in the repository under the name *condapackagelist.txt*. 

You may create a conda environment named boxhed2.0 and load it with the package list mentioend above. To do so, you can go to the cloned BoXHED2.0 repository, and run the following:
```
conda create -n boxhed2.0 --file condapackagelist.txt python=3.8
```
Having created the environment, you may run:
```
conda activate boxhed2.0
```
3. ****
