from boxhed import boxhed
from utils import timer
from preprocessor import preprocessor 
from sklearn.model_selection import GroupKFold, GridSearchCV 

import sys
sys.path.insert(0, '/home/grads/a/a.pakbin/survival_analysis/kernel/')
from kernel_smoothing import TrueHaz
from tqdm import tqdm
from utils import curr_dat_time

#print ("warning! set OMP_NUM_THREADS")
from utils import run_as_process

#TODO: make a new utils from this
CACHE_ADDRESS = "./CACHE"
import os
from pathlib import Path
import pickle
def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)

from sklearn.utils import indexable
import functools
import numpy as np
from multiprocessing import Array#, RawArray
from multiprocessing.sharedctypes import RawArray
import multiprocessing as mp
from contextlib import closing
import ctypes


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

@exec_if_not_cached
def _read_synth(ind_exp, num_irrelevant):
    import pandas as pd
    import numpy as np
    datname = 'data_exp' + str(ind_exp) + '_numIrr' + str(num_irrelevant)

    dir = '/home/grads/a/a.pakbin/survival_analysis/FlexSurv/exp' + str(ind_exp) + '/Feb_19_irrelevant_features/'
    long_lotraj = pd.read_csv(dir + 'train_long_lotraj_exp%d_numIrr_40.csv'%(ind_exp), sep=',', header=None) 

    long_lotraj = long_lotraj.iloc[:,0:num_irrelevant+3]#.head(20)
    long_lotraj.columns = ['patient', 't_start']+["X_%i"%i for i in range(1, num_irrelevant+2)]

    long_lotraj['t_end'] = long_lotraj['t_start'].shift(-1)
    long_lotraj['delta'] = 0

    delta = pd.read_csv(dir + 'train_delta_exp%d_numIrr_40.csv'%(ind_exp), sep=',', header=None).values.reshape(-1)

    TOTALTIME = 0.0
    pat_data_list = []
    for pat, pat_data in long_lotraj.groupby('patient'):
        pat_data.drop(pat_data.tail(1).index,inplace=True)
        pat_data['delta'].iloc[-1] = delta[pat-1]
        
        pat_data_list.append(pat_data)
        TOTALTIME += pat_data['t_end'].iloc[-1]-pat_data['t_start'].iloc[0]
    
    data = pd.concat(pat_data_list, ignore_index = True)
    data = data[['patient', 't_start', 't_end'] + ["X_%i"%i for i in range(1, num_irrelevant+2)] + ['delta']]

    F0 = np.log(np.sum(delta)/TOTALTIME)

    return data

@exec_if_not_cached
def _read_synth_test(ind_exp, num_irrelevant):
    import pandas as pd
    dataname =  'exp' + str(ind_exp) + '_numIrr_%d'
    data = pd.read_csv('../../BoXHED/test_data/' + 'test_random_pick_' + dataname%40 +  '.csv', sep=',', header=None)

    data = data.iloc[:,0:num_irrelevant+2]
    data.columns = ['t_start']+["X_%i"%i for i in range(1, num_irr+2)]
    
    true_haz = TrueHaz(data.values, ind_exp)

    return true_haz, data
    
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


#TODO: put them in different files


def _to_shared_mem(smm, arr_np):
    shared_mem       = smm.SharedMemory(size=arr_np.nbytes)
    arr_np_shared    = np.ndarray(shape = arr_np.shape, dtype = arr_np.dtype, buffer = shared_mem.buf)
    arr_np_shared[:] = arr_np[:]
    return shared_mem

#TODO: why doesn't it work as a function?
'''
def _from_shared_mem(shared_mem_name, shape, dtype): 
    arr_np_shared    = np.ndarray(shape = shape, dtype = dtype, buffer = shared_mem.buf)
    return arr_np_shared
'''

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

    '''
    print (X)
    print (y)


    print (np.array(X[data_idx[0]['test'], :]))
    print (X)
    print (type(X))
    print (np.array(np.array(np.array(X))))
    print ("BOOOOMMMM")
    #raise
    '''
    #print (pd.DataFrame(X_).iloc[])
    #exit()


    def _fit_single_model(param_dict):
        estimator = boxhed()
        estimator.set_params(**param_dict)
        
        fold = param_dict["fold"]
        estimator.fit(X[data_idx[fold]['train'], :], 
                      y[data_idx[fold]['train'], :])

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
            y[data_idx[fold]['test'], :],
            ntree_limit = n_trees)

        rslts[test_block_size*abs_idx+test_idx] = score


    def _fill_rslts():

        '''
        batch_idx = 2
        print ([(est_idx, batch_idx*batch_size+est_idx) for est_idx in range(len(trained_models))])
        raise
        '''
        Parallel(n_jobs = -1, prefer = "threads")(delayed (_fill_rslt)(rslt_idx) for rslt_idx in range(test_block_size*len(trained_models)))

        '''
        for est_idx, est in enumerate(trained_models):
            abs_idx    = batch_idx*batch_size+est_idx
            _fill_rslt(est, abs_idx)
        '''

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

        '''
        if 'OMP_NUM_THREADS' in os.environ:
            self.orig_omp_num_threads = os.environ['OMP_NUM_THREADS'] 
        else:
            self.orig_omp_num_threads = None

        os.environ['OMP_NUM_THREADS'] = "1"
        '''

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

        '''
        if self.orig_omp_num_threads is None:
            os.environ.pop('OMP_NUM_THREADS')
        else:
            os.environ['OMP_NUM_THREADS'] = self.orig_omp_num_threads
        '''

        return {
            "params":          self.srtd_param_dict_test,
            "mean_test_score": self.srtd_param_dict_test_scores,
            "std_test_score":  self.srtd_param_dict_test_std,
            "se_test_score":   self.srtd_param_dict_test_std/np.sqrt(len(self.cv)),
            "best_params":     self.srtd_param_dict_test[np.argmax(self.srtd_param_dict_test_scores)]
        }


#TODO: now there is nowhere to set GPU and CPU
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


hyperparams = {
        "41_0" : {'max_depth':1, 'n_estimators':300},
        "41_20": {'max_depth':1, 'n_estimators':200},
        "41_40": {'max_depth':1, 'n_estimators':250},
        "42_0" : {'max_depth':1, 'n_estimators':300},
        "42_20": {'max_depth':1, 'n_estimators':250},
        "42_40": {'max_depth':1, 'n_estimators':250},
        "43_0" : {'max_depth':2, 'n_estimators':150},
        "43_20": {'max_depth':2, 'n_estimators':50},
        "43_40": {'max_depth':2, 'n_estimators':50},
        "44_0" : {'max_depth':1, 'n_estimators':300},
        "44_20": {'max_depth':1, 'n_estimators':150},
        "44_40": {'max_depth':1, 'n_estimators':200}
}

def free_gpu_list():
    free_gpus = get_free_gpus()
    nom_free_gpus = free_gpus.count(True)
    free_gpu_list_ = [gpu_id for (gpu_id, gpu_free) in enumerate(free_gpus) if gpu_free == True]
    return free_gpu_list_

def _get_free_gpu_list(nom):
    GPU_LIST = free_gpu_list()
    if len(GPU_LIST) < nom:
        raise RuntimeError("ERROR: Not enough GPUs available!")

    return GPU_LIST[:nom]

@run_as_process
def grid_search_test_synth(ind_exp, num_irr, nom_gpu, model_per_gpu):

    #N: nom quant
    #TODO: old prep or WQ?
    N = 256

    
    
    #from sklearn.utils.estimator_checks import check_estimator
    #check_estimator(boxhed())
    #raise
    

    param_grid = {'max_depth':    [1, 2, 3, 4, 5],
                  'n_estimators': [50, 100, 150, 200, 250, 300]}

    #param_grid = {'max_depth':    [1, 5],
    #              'n_estimators': [50, 300]}

    rslt = {'ind_exp':       ind_exp, 
            'num_irr':       num_irr, 
            'nom_gpu':       nom_gpu, 
            'model_per_gpu': model_per_gpu}

    data = _read_synth(ind_exp, num_irr)
    
    #TODO: make sure running twice = once?
    prep = preprocessor()
    #TODO: maybe change how to feed "data"?
    
    rslt['nom_quant'] = N
    prep_timer = timer()

    #data = data.head(20)
    ####data = pd.read_csv('____TEST____.csv')
    ####N = 3
    ####print (data)

    pats, X, y = prep.preprocess(data, N, True)
    ####print (X)
    ####print (y)

    ####X_test = pd.read_csv("____TEST____X.csv")
    ####print (X_test)
    ####print (prep.fix_data_on_boundaries(X_test))


    ####raise
    #print (X.shape)
    
    '''for i in range(X.shape[1]):
        str_to_print = ""
        for j in sorted(X.iloc[:,i].unique()):
            if str_to_print != "":
                str_to_print += ", "
            str_to_print += "%.5f"%j
        print (str_to_print, "\n")
    print ("\n\n")'''

    rslt["prep_time"] = prep_timer.get_dur()
    
    #gpu_list = _get_free_gpu_list(nom_gpu)
    gpu_list = [-1]
    #TODO: change the following
    #gpu_list = [-1]
     
    #model_per_gpu = len(param_grid)

    '''
    gridsearch_timer = timer()
    #TODO: handle memory exception if model_per_gpu too large
    cv_rslts, best_params = collapsed_ntree_gs(boxhed(), 
                                  param_grid, 
                                  X, 
                                  y, 
                                  pats, 
                                  5,
                                  gpu_list,
                                  model_per_gpu,
                                  -1)
    
    rslt["GS_time"] = gridsearch_timer.get_dur()
    '''
    #TODO: handle this shit
    #TODO: what if the model is not using GPU at all?

    best_params = hyperparams["%d_%d"%(ind_exp, num_irr)]
    #best_params = {'max_depth': 1, 'n_estimators': 250}
    best_params['gpu_id'] = gpu_list[0]
    #TODO for GPU it's better to be -1 but for CPU 1. Make that specification possible
    best_params['nthread'] = -1

    rslt.update(best_params)
     
    boxhed_ = boxhed(**best_params)

    fit_timer = timer()
    #boxhed_.fit (X,y)
    boxhed_.fit (X,y.iloc[:,0], y.iloc[:,1])
    rslt["fit_time"] = fit_timer.get_dur()

    #TODO: estimator check sklearn
    #TODO: set __all__ for the scripts

    true_haz, test_X = _read_synth_test(ind_exp, num_irr) 

    pred_timer = timer()
    #TO_DO: change
    #test_X = test_X.iloc[0,:].values.reshape(1, -1)
    #test_X[0,1] = 10
    #print (test_X)
    #import math
    test_x = prep.fix_data_on_boundaries(test_X)
    preds = boxhed_.predict(test_X)
    #print (preds[0])
    #print (math.exp(-0.100697))
    rslt["pred_time"] = pred_timer.get_dur()

    rslt["rmse"] = "%.3f"%(np.sqrt(np.square(true_haz - preds)).mean(axis=None))
    return rslt


import pandas as pd

if __name__ == "__main__":

    #TODO: fix the names for when there is CPU instead of GPU
    nom_gpu = 1#2
    for model_per_gpu in [6]:#[1, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 25]:

        rslts = []
        for ind_exp in [41, 42, 43, 44]:
            for num_irr in [0,20,40]:

                print ('    exp:    ', ind_exp)
                print ('    num_irr:', num_irr)
                print ('    nom GPU:', nom_gpu)
                print ('    /GPU:   ', model_per_gpu)
                print ("")

                rslt = grid_search_test_synth(ind_exp, 
                                              num_irr,
                                              nom_gpu, 
                                              model_per_gpu)

                #print ('t=',rslt['GS_time'], "  ",rslt['rmse'], '\n', "~"*5, "\n"*2, sep="")
                print (rslt, "\n"*3, rslt["rmse"], "\n"*2, sep="")
                rslts.append(rslt)


        #print ("\n"*2, "__"+curr_dat_time()+"__")
        #TODO: get the day and save the result in the results file
        print (pd.DataFrame(rslts))
        #pd.DataFrame(rslts).to_csv("./RESULTS/NEW_IMP__CPU_%d_gpus_6_per_gpu.csv"%nom_gpu, index = None)
        pd.DataFrame(rslts).to_csv("./RESULTS/OLD_PREP_265_HYPER__GPU__nom_GPU_%d__model_per_gpu_%d__nthread_-1.csv"%(nom_gpu, model_per_gpu), index = None)
