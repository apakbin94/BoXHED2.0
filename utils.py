from datetime import datetime
from pytz import timezone
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from joblib import Parallel, delayed
import itertools
from collections import namedtuple

from boxhed import boxhed

import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd
from pathlib import Path
import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "survival_analysis/BoXHED2.0/xgboost/python-package/"))

CACHE_ADDRESS = './tmp/'

#log("loading XGB")
import xgboost as xgb
import copy
from joblib import Parallel, delayed
import numpy as np
#'''
import os
print ("CAN WE DO ANYTHING ABOUT OMP_NUM_THREADS")
os.environ['OMP_NUM_THREADS'] = "1"
#'''
from itertools import product
from sklearn.base import clone
from multiprocessing import Manager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
from py3nvml import get_free_gpus
from sklearn.model_selection import GroupKFold, GridSearchCV 



def curr_dat_time ():
    curr_dt = datetime.now(timezone("US/Central"))
    return curr_dt.strftime("%a, %b %d, %H:%M:%S")

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

def exec_if_not_cached(func):
    @functools.wraps(func)

    def _exec_if_not_cached(file_name, func, *args, **kwargs):

        file_name_ = file_name+'.pkl'
        file_path = os.path.join(CACHE_ADDRESS, file_name_)

        if Path(file_path).is_file():
            with open( file_path, "rb" ) as file_handle:
                return pickle.load(file_handle)

        else:
            create_dir_if_not_exist(os.path.dirname(file_path))
            output = func(*args, **kwargs)

            with open( file_path, "wb" ) as file_handle:
                pickle.dump(output, file_handle)

            return output


    def _func_args_to_str(func, *args, **kwargs):
        output = func.__name__
        for arg in args:
            output += "__"+str(arg)

        for key, val in kwargs.items():
            output += "__"+str(key)+"_"+str(val)

        return output

    def exec_if_not_cached(*args, **kwargs):

        file_name = _func_args_to_str(func, *args, **kwargs)
        return _exec_if_not_cached(file_name, func, *args, **kwargs)

    return exec_if_not_cached


import time
def time_now():
    return time.time()

class timer:

    def __init__(self):
        self.t_start = time_now()

    def get_dur(self):
        return round(time_now()-self.t_start, 3)

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

import functools
from multiprocessing import Process, Queue


def run_as_process(func):
    @functools.wraps(func)
    def run_as_process(*args, **kwargs):

        def _func(queue, func, *args, **kwargs):
            queue.put(func(*args, **kwargs))

        queue = Queue()
        p = Process(target=_func, args=(queue, func, *args), kwargs=kwargs)
        p.start()
        p.join()
        p.terminate()
        return queue.get()
    return run_as_process


def _free_gpu_list():
    free_gpus = get_free_gpus()
    nom_free_gpus = free_gpus.count(True)
    free_gpu_list_ = [gpu_id for (gpu_id, gpu_free) in enumerate(free_gpus) if gpu_free == True]
    return free_gpu_list_

def _get_free_gpu_list(nom):
    GPU_LIST = _free_gpu_list()
    if len(GPU_LIST) < nom:
        raise RuntimeError("ERROR: Not enough GPUs available!")

    return GPU_LIST[:nom]

from sklearn.utils import indexable
import functools
import numpy as np
from multiprocessing import Array#, RawArray
from multiprocessing.sharedctypes import RawArray
import multiprocessing as mp
from contextlib import closing
import ctypes


def _to_shared_mem(smm, arr_np):
    shared_mem       = smm.SharedMemory(size=arr_np.nbytes)
    arr_np_shared    = np.ndarray(shape = arr_np.shape, dtype = arr_np.dtype, buffer = shared_mem.buf)
    arr_np_shared[:] = arr_np[:]
    return shared_mem


def _run_batch_process(param_dict_, rslts):

    #[X, y] = child_data_conn.recv()

    param_dicts_train = param_dict_['param_dicts_train']
    param_dicts_test  = param_dict_['param_dicts_test']
    data_idx          = param_dict_['data_idx']
    X_shared_name     = param_dict_['X_shared_name']
    X_shape           = param_dict_['X_shape']
    X_dtype           = param_dict_['X_dtype']
    y_shared_name     = param_dict_['y_shared_name']
    y_shape           = param_dict_['y_shape']
    y_dtype           = param_dict_['y_dtype']
    batch_size        = param_dict_['batch_size']
    batch_idx         = param_dict_['batch_idx']
    test_block_size   = param_dict_['test_block_size']

    smem_x = SharedMemory(X_shared_name)
    smem_y = SharedMemory(y_shared_name)

    X = np.ndarray(shape = X_shape, dtype = X_dtype, buffer = smem_x.buf)
    y = np.ndarray(shape = y_shape, dtype = y_dtype, buffer = smem_y.buf)


    def _fit_single_model(param_dict):
        estimator = boxhed()
        estimator.set_params(**param_dict)
        
        fold = param_dict["fold"]
        estimator.fit(X[data_idx[fold]['train'], :], 
                      y[data_idx[fold]['train'], 0],
                      y[data_idx[fold]['train'], 1])

        return estimator


    def _fill_rslt(rslt_idx):
        est_idx   = rslt_idx // test_block_size
        test_idx  = rslt_idx %  test_block_size
        abs_idx   = batch_idx*batch_size+est_idx

        est       = trained_models[est_idx]
        test_dict = param_dicts_test[test_block_size*abs_idx + test_idx]

        n_trees = test_dict['n_estimators']
        fold    = test_dict['fold']

        score = est.score(
            X[data_idx[fold]['test'], :], 
            y[data_idx[fold]['test'], 0],
            y[data_idx[fold]['test'], 1],
            ntree_limit = n_trees)

        rslts[test_block_size*abs_idx+test_idx] = score


    def _fill_rslts():

        Parallel(n_jobs = -1, prefer = "threads")(delayed (_fill_rslt)(rslt_idx) for rslt_idx in range(test_block_size*len(trained_models)))


    trained_models = Parallel(n_jobs=-1, prefer="threads")(delayed(_fit_single_model)(param_dict) 
            for param_dict in param_dicts_train[batch_idx*batch_size:
                (batch_idx+1)*batch_size])


    _fill_rslts()


class collapsed_gs_:

    def _remove_keys(self, dict_, keys):
        dict_ = copy.copy(dict_)
        for key in keys:
            dict_.pop(key, None)
        return dict_

    def _dict_to_tuple(self, dict_):
        list_ = []
        for key in sorted(dict_.keys()):
            list_ += [key, dict_[key]]
        return tuple(list_)


    def _fill_data_idx(self):
        for idx, (train_idx, test_idx) in enumerate(self.cv):
            self.data_idx[idx] = {"train":train_idx ,"test":test_idx}
            

    def __init__(self, estimator, param_grid, cv, n_jobs, GPU_LIST, model_per_gpu):

        self.estimator  = estimator
        self.param_grid = copy.copy(param_grid)
        self.param_grid["fold"] = [x for x in range(len(cv))]

        self.model_per_gpu = model_per_gpu
        
        self.collapsed     = 'n_estimators'
        self.not_collapsed = 'max_depth'

        self.param_grid_train = copy.copy(self.param_grid)
        #self.param_grid_fit["fold"] = [x for x in range(len(cv))]

        self.param_grid_train[self.collapsed] = [
                max(self.param_grid_train[self.collapsed])
                ]

        self.GPU_LIST = GPU_LIST

        self.param_dicts_train = self._make_indiv_dicts(
                self.param_grid_train, train=True)       

        self.param_dicts_test = self._make_indiv_dicts(self.param_grid)

        self.param_dicts_test.sort(key=lambda dict_: 
                (dict_[self.not_collapsed], dict_['fold']))

        self.test_block_size = len(self.param_grid[self.collapsed])

        #raise
        self.cv         = cv
        self.n_jobs     = n_jobs

        self.data_idx   = {}
        self._fill_data_idx()

        manager                = Manager()
        self.rslts             = manager.list(['0']*len(self.param_dicts_test))


    def _make_indiv_dicts(self, dict_, train=False):
    
        keys, values =zip(*dict_.items())
        dicts = [dict(zip(keys, x)) for x in product(*values)]

        if train:
            for idx, dict_ in enumerate(dicts):
                dict_.update({'gpu_id':
                    self.GPU_LIST[(idx // self.model_per_gpu) 
                        % len(self.GPU_LIST)]
                    })

        return dicts


    #TODO: what if it was on CPU?
    def _batched_train_test(self):

        batch_size = len(self.GPU_LIST)*self.model_per_gpu

        smm = SharedMemoryManager()
        smm.start()

        X_shared = _to_shared_mem(smm, self.X)
        y_shared = _to_shared_mem(smm, self.y)

        with Manager() as manager:
            param_dict_mngd =     manager.dict({
                'param_dicts_train': self.param_dicts_train,
                'param_dicts_test':  self.param_dicts_test,
                'data_idx':          self.data_idx,
                'X_shared_name':     X_shared.name,
                'X_shape':           self.X.shape,
                'X_dtype':           self.X.dtype,
                'y_shared_name':     y_shared.name,
                'y_shape':           self.y.shape,
                'y_dtype':           self.y.dtype,
                'batch_size':        batch_size,
                'batch_idx':         None,
                'test_block_size':   self.test_block_size
                        })

            rslt_mngd =          manager.list([0]*len(self.param_dicts_test))

            for batch_idx in range(int(len(self.param_dicts_train)/batch_size)+1):

                param_dict_mngd['batch_idx'] = batch_idx

                '''
                _run_batch_process(param_dict_mngd, rslt_mngd)
                print (rslt_mngd)
                raise
                '''
                p = mp.Process(target = _run_batch_process, args = (param_dict_mngd, rslt_mngd))

                p.start() 
                p.join()
                p.terminate()

            smm.shutdown()
            
            self.rslts = list(rslt_mngd)

    def _calculate_output_statistics(self):
        def sort_pred(dict_):
            return (dict_['max_depth'], dict_['n_estimators'], dict_['fold'])

        #self.param_dicts_test = list(self.param_dicts_test)
        #self.rslts            = list(self.rslts)

        rslt__param_dict_test = \
            [(rslt, param_dict_test) for rslt, param_dict_test in \
                sorted(zip(self.rslts,self.param_dicts_test), 
                key=lambda pair: sort_pred(pair[1]))
                    ]
        srtd_rslts, srtd_param_dict_test = zip(*rslt__param_dict_test)
        srtd_rslts = np.array(srtd_rslts)
        srtd_rslts = srtd_rslts.reshape(-1, len(self.cv))

        self.srtd_param_dict_test_scores = srtd_rslts.mean(axis=1)
        self.srtd_param_dict_test_std    = srtd_rslts.std(axis=1)
 
        self.srtd_param_dict_test = srtd_param_dict_test[0::len(self.cv)]
        for srtd_param_dict_test in self.srtd_param_dict_test:
            srtd_param_dict_test.pop('fold')


    def fit(self, X,y):
        self.X = np.array(X)
        self.y = np.array(y)

        self._batched_train_test()

        self._calculate_output_statistics()


        return {
            "params":          self.srtd_param_dict_test,
            "mean_test_score": self.srtd_param_dict_test_scores,
            "std_test_score":  self.srtd_param_dict_test_std,
            "se_test_score":   self.srtd_param_dict_test_std/np.sqrt(len(self.cv)),
            "best_params":     self.srtd_param_dict_test[np.argmax(self.srtd_param_dict_test_scores)]
        }


def collapsed_ntree_gs(estimator, param_grid, x, y, groups, n_splits, gpu_list, model_per_gpu, n_jobs = -1):
    x, y, groups = indexable(x,y,groups)

    gkf = list(GroupKFold(n_splits=n_splits).split(x,y['delta'],groups))

    collapsed_ntree_gs_  = collapsed_gs_(estimator, param_grid, gkf, n_jobs, gpu_list, model_per_gpu)
 
    results     = collapsed_ntree_gs_.fit(x,y)
    means       = results['mean_test_score']
    stds        = results['std_test_score']
    params      = results['params']
    best_params = results['best_params']

    return list(zip(params, means, stds)), best_params


