import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin, TransformerMixin 
from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from preprocessor import preprocessor

#TODO: maybe this can change once I figure out the BoXHED/XGB distinction while installing

import xgboost as xgb

from xgboost import plot_tree
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


def _left_ind_f(split_col, split_val, include_missing):
    missing_ind_f = np.is_nan if include_missing else (lambda x: np.zeros_like(x, dtype=bool))
    return (lambda X: np.logical_or(X[:, split_col]<split_val, missing_ind_f(X[:, split_col])))


def _find_idx_series(series, val):
    return series[series == val].index[0]


class pred_node:
    def recursive_build(self, tree_df, col_to_idxs, row):
        tree_row = tree_df.iloc[row]
        
        if tree_row['Feature']=="Leaf":
            self.weight = tree_row['Gain']
            return
        
        self.left_ind_f = _left_ind_f(
                                split_col       = col_to_idxs[tree_row['Feature']], 
                                split_val       = tree_row['Split'], 
                                include_missing = tree_row['Yes'    ] == tree_row['Missing'])

        self.left_node, self.rigt_node = pred_node(), pred_node()
        self.left_node.recursive_build(tree_df, col_to_idxs, _find_idx_series(tree_df['ID'], tree_row['Yes']))
        self.rigt_node.recursive_build(tree_df, col_to_idxs, _find_idx_series(tree_df['ID'], tree_row['No' ]))


    def recursive_pred(self, X):
        if hasattr(self, 'weight'):
            return self.weight

        left_ind = self.left_ind_f(X)
        return  left_ind            * self.left_node.recursive_pred(X) + \
                np.invert(left_ind) * self.rigt_node.recursive_pred(X)


class pred_tree:
    def __init__(self):
        self.root = None

    def build(self, tree_df, col_to_idxs):
        self.root = pred_node()
        self.root.recursive_build(tree_df, col_to_idxs, 0)

    def pred(self, X):
        return self.root.recursive_pred(X)



class iboxhed_pred_trees:
    def build(self, trees_df, col_to_idx):
        self.col_to_idx = col_to_idx
        pred_trees = []
        feature_trees = defaultdict(list)
        for tree, tree_df in tqdm(trees_df.groupby('Tree'), desc="building pred trees"):
            tree_df.reset_index(drop=True, inplace=True)
            t = pred_tree()
            t.build(tree_df, col_to_idx)
            pred_trees.append(t)
            feature_set = set(tree_df['Feature'])
            feature_set.remove('Leaf')
            for feature_idx in feature_set:
                tree_df_ = tree_df.copy()
                tree_df_.loc[~tree_df_['Feature'].isin([feature_idx, 'Leaf', 'time']), 'Gain'] = 0
                tree_df_.loc[~tree_df_['Feature'].isin([feature_idx, 'Leaf', 'time']), 'Feature'] = 'Leaf'
                t = pred_tree()
                t.build(tree_df_, col_to_idx)
                feature_trees[feature_idx].append(t)

        self.pred_trees = pred_trees
        self.feature_trees = feature_trees
        
    def predict(self, X, f0):
        return f0+sum([tree.pred(X) for tree in tqdm(self.pred_trees, desc="predicting")])

    def contrib_predict(self, X, col):
        if not isinstance(X, np.ndarray):
            X = X.values
        return sum([tree.pred(X) for tree in tqdm(self.feature_trees[col], desc=f"predicting column {col} contribution")])


class boxhed(BaseEstimator, RegressorMixin):#ClassifierMixin, 

    def __init__(self, max_depth=1, n_estimators=100, eta=0.1, gpu_id = -1, nthread = 1):
        self.max_depth     = max_depth
        self.n_estimators  = n_estimators
        self.eta           = eta
        self.gpu_id        = gpu_id
        self.nthread       = nthread


    def _X_y_to_dmat(self, X, y=None, w=None):
        if not hasattr(self, 'X_colnames'):
            self.X_colnames = None #model probably created for CV, no need for data name matching
        dmat = xgb.DMatrix(pd.DataFrame(X, columns=self.X_colnames))

        if (y is not None):
            dmat.set_float_info('label',  y)
            dmat.set_float_info('weight', w)
    
        return dmat
        
    def preprocess(self, data, is_cat=[], num_quantiles=20, weighted=False, nthread=-1):
        self.prep = preprocessor()
        IDs, X, w, delta =  self.prep.preprocess(
            data             = data, 
            is_cat           = is_cat,
            num_quantiles    = num_quantiles, 
            weighted         = weighted, 
            nthread          = nthread)

        self.X_colnames = X.columns.values.tolist()
        self.X_colnames = [item if item!='t_start' else 'time' for item in self.X_colnames]

        return IDs, X, w, delta


    def _get_time_cov_only_interatctions(self, cols):
        time_idx = cols.index('t_start')
        return [[time_idx, idx] for idx in range(len(cols)) if idx!=time_idx ]


    def fit (self, X, y, w=None):

        #TODO: could I do the type checking better?
        check_array(y, ensure_2d = False)
        #TODO: make sure prep exists
        # or: if does not exist, create it now and train on preprocessed

        self.train_X_cols = X.columns.tolist()
        self.interactions = self._get_time_cov_only_interatctions(X.columns.tolist())

        le = LabelEncoder()
        y  = le.fit_transform(y)
        X, y       = check_X_y(X, y, force_all_finite='allow-nan')

        if len(set(y)) <= 1:
            raise ValueError("Classifier can't train when only one class is present. All deltas are either 0 or 1.")
    
        if w is None:
            w = np.ones_like(y)

        f0_   = np.log(np.sum(y)/np.sum(w))
        self.f0_ = f0_
        dmat_ = self._X_y_to_dmat(X, y, w)

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
                                'interaction_constraints': self.interactions,
                                'base_score':       f0_,
                                'gpu_id':           self.gpu_id,
                                'nthread':          self.nthread
                                }
    
        self.boxhed_ = xgb.train( self.params_, 
                                  dmat_, 
                                  num_boost_round = self.n_estimators) 
        
        self.VarImps = self.boxhed_.get_score(importance_type='total_gain')
        return self

        
    def plot_tree(self, nom_trees):
                        
        def print_tree(i):
            print("printing tree:", i+1)
            plot_tree(self.boxhed_, num_trees = i)
            fig = plt.gcf()
            fig.set_size_inches(30, 20)
            fig.savefig("tree"+"_"+str(i+1)+'.jpg')

        for th_id in range(min(nom_trees, self.n_estimators)):
            print_tree(th_id)


    def predict(self, X, ntree_limit = 0):
        check_is_fitted(self)
        '''
        self.prep.shift_left(X)
        '''
        try:
            X = self.prep.shift_left(X)
        except:
            pass
        X = check_array(X, force_all_finite='allow-nan')

        return self.boxhed_.predict(self._X_y_to_dmat(X), ntree_limit = ntree_limit)

    def get_survival(self, X, t, ntree_limit = 0): #TODO no ind_exp
        def truncate_to_t(data, t):
            def _truncate_to_t(data_id):
                data_id                   = data_id[data_id['t_start']<t]
                data_id['t_end']          = data_id['t_end'].clip(upper=t)
                if len(data_id)>0:
                    data_id['t_end'].iloc[-1] = t
                return data_id
            return data.groupby('ID').apply(_truncate_to_t).reset_index(drop=True)

        check_is_fitted(self)
        X                              = truncate_to_t(X, t)
        cte_hazard_epoch_df            = self.prep.epoch_break_cte_hazard(X)
        cte_hazard_epoch               = check_array(cte_hazard_epoch_df.drop(columns=["ID", "dt", "delta"]), 
                                            force_all_finite='allow-nan')
        cte_hazard_epoch               = self._X_y_to_dmat(cte_hazard_epoch)
        preds                          = self.boxhed_.predict(cte_hazard_epoch, ntree_limit = ntree_limit)
        cte_hazard_epoch_df ['preds']  = preds
        cte_hazard_epoch_df ['surv']   = -cte_hazard_epoch_df ['dt'] * cte_hazard_epoch_df ['preds']
        surv_t                         = np.exp(cte_hazard_epoch_df.groupby('ID')['surv'].sum()).reset_index()
        surv_t.rename(columns={'surv':f'surv_at_t={t}'}, inplace=True)
        return surv_t.set_index('ID')
        


    def get_params(self, deep=True):
        return {"max_depth":     self.max_depth, 
                "n_estimators":  self.n_estimators,
                "eta":           self.eta, 
                "gpu_id":        self.gpu_id,
                "nthread":       self.nthread}


    def set_params(self, **params):
        for param, val in params.items():
            setattr(self, param, val)
        return self


    def score(self, X, y, w=None, ntree_limit=0):
        X, y    = check_X_y(X, y, force_all_finite='allow-nan')
        if w is None:
            w = np.zeros_like(y)

        preds = self.predict(X, ntree_limit = ntree_limit)
        return -(np.inner(preds, w)-np.inner(np.log(preds), y))

    def iboxhed_build(self):
        check_is_fitted(self)
        assert hasattr(self, "interactions"), "ERROR: iBoXHED not fitted with interaction constraints!"
        trees_df = self.boxhed_.trees_to_dataframe()
        cols = [col if col != 't_start' else 'time' for col in self.train_X_cols if col not in ["ID", "t_end", "delta"]]
        col_to_idx = {col:idx for idx, col in enumerate(cols)}
        #trees_df["Feature"].replace({col:idx for idx, col in enumerate(cols)}, inplace=True)
        self.iboxhed_pred_trees = iboxhed_pred_trees()
        self.iboxhed_pred_trees.build(trees_df, col_to_idx)


    def iboxhed_predict(self, X):
        check_is_fitted(self)
        assert hasattr(self, "iboxhed_pred_trees"), "ERROR: iBoXHED prediction trees not built!"
        return np.exp(self.iboxhed_pred_trees.predict(X.values, self.f0_))