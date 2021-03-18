# BoXHED2.0Main

For information on the functionalities of BoXHED2.0, please refer to [BoXHED1.0 Paper](http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) published in ICML 2020.

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

Now open the demo.ipynb file for a quick demonstration of how to train/test a BoXHED model.
