from boxhed import boxhed
from utils import timer, curr_dat_time, run_as_process, exec_if_not_cached, _get_free_gpu_list, collapsed_ntree_gs
from preprocessor import preprocessor 

import sys
sys.path.insert(0, '/home/grads/a/a.pakbin/survival_analysis/kernel/')
from kernel_smoothing import TrueHaz

#print ("warning! set OMP_NUM_THREADS")

#TODO: make a new utils from this
CACHE_ADDRESS = "./CACHE"
import os
from pathlib import Path
import pickle
import numpy as np


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


@run_as_process
def grid_search_test_synth(ind_exp, num_irr, nom_gpu, model_per_gpu):

    #N: nom quant
    #TODO: old prep or WQ?
    N = 20

    
    
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

    #'''
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
    #'''
    #TODO: handle this shit
    #TODO: what if the model is not using GPU at all?

    #best_params = hyperparams["%d_%d"%(ind_exp, num_irr)]
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
