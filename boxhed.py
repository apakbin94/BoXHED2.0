import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin 
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from collections.abc import Iterable

#TODO: change this?
import subprocess                                                           
import os

OLD_OR_NEW  =  ""

if False:
    OLD_OR_NEW = "old_xgb/"

p = subprocess.Popen(['make', '-j4'], cwd=os.path.join(os.path.expanduser("~"), "survival_analysis/BoXHED2.0/"+OLD_OR_NEW+"xgboost/build/"))
p.wait()

python_setup_log = open("./CACHE/python_setup_log.txt", "w")
p = subprocess.Popen(['python', 'setup.py', 'install'], cwd=os.path.join(os.path.expanduser("~"), "survival_analysis/BoXHED2.0/"+OLD_OR_NEW+"xgboost/python-package/"), stdout = python_setup_log, stderr = python_setup_log)
p.wait()
python_setup_log.close()


import sys
import os
sys.path.append(os.path.join(os.path.expanduser("~"), "survival_analysis/BoXHED2.0/"+OLD_OR_NEW+"xgboost/python-package/"))

import xgboost as xgb

from xgboost import plot_tree
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


class boxhed(BaseEstimator, RegressorMixin):#ClassifierMixin, 

    def __init__(self, max_depth=1, n_estimators=100, eta=0.1, gpu_id = -1, nthread = -1):
        self.max_depth     = max_depth
        self.n_estimators  = n_estimators
        self.eta           = eta
        self.gpu_id        = gpu_id
        self.nthread       = nthread


    def _X_y_to_dmat(self, X, y=None, dt=None):
        dmat = xgb.DMatrix(X)

        if (y is not None):
            dmat.set_float_info('label',  y)
            dmat.set_float_info('weight', dt)
    
        return dmat
        

    def fit (self, X, y, dt=None):

        #TODO: could I do the type checking better?
        check_array(y, ensure_2d = False)

        le = LabelEncoder()
        y  = le.fit_transform(y)
        X, y       = check_X_y(X, y)

        if len(set(y)) <= 1:
            raise ValueError("Classifier can't train when only one class is present. All deltas are either 0 or 1.")
    
        if dt is None:
            dt = np.ones_like(y)

        f0_   = np.log(np.sum(y)/np.sum(dt))
        dmat_ = self._X_y_to_dmat(X, y, dt)

        if self.gpu_id>=0:
            self.objective_   = 'survival:boxhed_gpu'
            self.tree_method_ = 'gpu_hist'
        else:
            self.objective_   = 'survival:boxhed'
            self.tree_method_ = 'hist'

        self.params_         = {'objective':        self.objective_,
                                'tree_method':      self.tree_method_,
                                'booster':         'gbtree', 
                                'min_child_weight': 0,
                                'max_depth':        self.max_depth,
                                'eta':              self.eta,
                                'grow_policy':     'lossguide',

                                'base_score':       f0_,
                                'gpu_id':           self.gpu_id,
                                'nthread':          self.nthread
                                }
    
        self.boxhed_ = xgb.train( self.params_, 
                                  dmat_, 
                                  num_boost_round = self.n_estimators) 
        return self

        
    def plot_tree(self, nom_trees):
                        
        def print_tree(i):
            print("printing tree:", i+1)
            plot_tree(self.boxhed_, num_trees = i)
            fig = plt.gcf()
            fig.set_size_inches(30, 20)
            fig.savefig("./RESULTS/"+
                str(self.tree_method)+"_"+str(i)+'.jpg')

        for th_id in range(min(nom_trees, self.n_estimators)):
            print_tree(th_id)


    def predict(self, X, ntree_limit = 0):
        check_is_fitted(self)
        X = check_array(X)

        return self.boxhed_.predict(self._X_y_to_dmat(X), ntree_limit = ntree_limit)


    def get_params(self, deep=True):
        return {"max_depth":     self.max_depth, 
                "n_estimators":  self.n_estimators,
                "eta":           self.eta, 
                "gpu_id":        self.gpu_id,
                "nthread":       self.nthread}


    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self


    def score(self, X, y, dt=None, ntree_limit=0):
        X, y    = check_X_y(X, y)
        if dt is None:
            dt = np.zeros_like(y)

        preds = self.predict(X, ntree_limit = ntree_limit)
        return -(np.inner(preds, dt)-np.inner(np.log(preds), y))
