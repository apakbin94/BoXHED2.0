# BoXHED2.0

What’s new (over BoXHED 1.0):
 - Allows for survival data far beyond right censoring (specifically, Aalen’s multiplicative intensity setting). Examples include left truncation and recurrent events.
 - Significant speed improvement
 - multicore CPU and GPU support

Please refer to [BoXHED2.0 Paper](https://arxiv.org/abs/2103.12591) for details, which builds on [BoXHED1.0 Paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) (ICML 2020).

## Prerequisites
The software developed and tested in Linux and Mac OS environments. The requirements are the following:
- cmake  (>=3.18.2)
- Python (>=3.8)
- conda

We highly recommend devoting a conda environment to BoXHED 2.0. This step makes sure BoXHED 2.0 will not interfere with XGBoost (the library we have borrowed from extensively) when installed. This implementation uses python 3.8.
Installing the conda environment should be done prioer to opening this notebook. Therefore, you need to set up the environment as instructed here and then reopen this notebook. So, please open a terminal and do the following:

First create the conda environment:
```
conda create -n boxhed2.0 python=3.8
```

then activate it
```
conda activate boxhed2.0
```

now install numpy, pandas, scikit-learn, pytz, py3nvml, matplotlib and jupyter notebook by:
```
bash conda_install_packages.sh
```

Subsequently, you can install BoXHED2.0 by running:
```
bash setup.sh
```

then run jupyter notebook
```
jupyter notebook 
``` 

Now open the *tutorial.ipynb* file for a quick demonstration of how to train/test a BoXHED model on a synthetic dataset.

Please note that everytime you relocate the code, you need to run bash setup.sh again.
