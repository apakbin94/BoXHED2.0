import numpy as np
import pandas as pd
import os
from boxhed import boxhed
from utils import timer, curr_dat_time, run_as_process, exec_if_not_cached, _get_free_gpu_list, create_dir_if_not_exist
from model_selection import cv
import math

from scipy.stats import beta # beta distribution.
from scipy.stats import norm # normal distribution.

DATA_ADDRESS = "./synth_files/"
RSLT_ADDRESS = "./results/"

for addr in [RSLT_ADDRESS]:
    create_dir_if_not_exist(addr)

#TODO: get these from command line?
num_quantiles   = 256
do_CV           = True
use_gpu         = False

# when CPU hist is used, the batch size would be num_gpu * model_per_gpu
num_gpus = [1]#[4, 6]
batch_sizes = [10]#[8, 10]

# calc_L2: calculate L2 distance and its 95% confidence interval.
# Input:
#      @ pred: a numpy array of a column indicates predicted hazards at testing data.
#      @ true: a numpy array of a column indicates true hazards at testing data
# Return:
#      @ meanL2: point estimator of L2 distance.
#      @ CIL2: 95% confidence interval 
def calc_L2(pred, true):
    L2 = (pred-true)**2
    N = pred.shape[0]
    meanL2_sqr = sum(L2)/N # L2 distance
    sdL2_sqr = math.sqrt(sum((L2-meanL2_sqr)**2)/(N-1))
    meanL2 = math.sqrt(meanL2_sqr)
    #print (meanL2, N, sdL2_sqr)
    CIL2 = [meanL2-1.96*sdL2_sqr/2/meanL2/math.sqrt(N), meanL2+1.96*sdL2_sqr/2/meanL2/math.sqrt(N)]
    return([meanL2, CIL2])


#%%
# Function: TrueHaz
# calculate values of hazard function on testing data.
# Input:
#      @ traj: test data, a numpy array where each row is a testing data point.
#      @ ind_exp: index of hazard function. Details refer to writeup. 
#                 User can add new hazard function by creating new ind_exp.
# Return: A numpy array of hazards on testing data. 
def TrueHaz(traj, ind_exp):
    N = traj.shape[0]
    result = np.zeros((N,))
    if ind_exp==2:
        for i in range(N):
            t, x1, x2, x3 = traj[i,0:4]
            if x3<=0.5:
                temp = (x1+2*x2+2)**2*t
            else:
                temp = (2*x1+x2+1)**2*t**2
            result[i] = temp
    elif ind_exp == 3:
        for i in range(N):
            t, x1, x2, x3 = traj[i,0:4]
            if x3<=0.5:
                temp = 4*t
            else:
                temp = 10*t**2
            result[i] = temp
    elif ind_exp == 31:
        for i in range(N):
            t, x1 = traj[i,0:2]
            if x1<=0.5:
                temp = 4*t
            else:
                temp = 10*t**2
            result[i] = temp   
    elif ind_exp == 41:
        result = beta.pdf(traj[:,0], 2, 2)*beta.pdf(traj[:,1], 2, 2) 
    elif ind_exp == 42:
        result = beta.pdf(traj[:,0], 4, 4)*beta.pdf(traj[:,1], 4, 4) 
    elif ind_exp == 43:
        logt = np.array([math.log(t) for t in traj[:,0]])
        result = norm.pdf(logt-traj[:,1])/(traj[:,0]*norm.cdf(traj[:,1] - logt))            
    elif ind_exp == 44:
        cos2piz = np.array([math.cos(2*math.pi*x1) for x1 in traj[:,1]])    
        result = 1.5*traj[:,0]**(1/2)*np.exp(-0.5*cos2piz-1.5)
    elif ind_exp == 51:
        result = beta.pdf(traj[:,0], 2, 2)*beta.pdf(traj[:,1], 2, 2)
        for i in range(traj.shape[0]):
            if traj[i,2]>0.5 and traj[i,3]>0.5:
                result[i] = result[i]+1
    elif ind_exp == 52:
        result = beta.pdf(traj[:,0], 2, 2)*beta.pdf(traj[:,1], 2, 2)
        for i in range(traj.shape[0]):
            if traj[i,2]>0.5 and traj[i,3]>0.5:
                result[i] = result[i]+1
            if traj[i,4]>0.5:
                result[i] = result[i]-0.5
    elif ind_exp == 53:
        result = beta.pdf(traj[:,0], 2, 2)*beta.pdf(traj[:,1], 2, 2)
    elif ind_exp == 54:
        result = beta.pdf(traj[:,0], 4, 4)*beta.pdf(traj[:,1], 4, 4) 
    elif ind_exp == 55:
        logt = np.array([math.log(t) for t in traj[:,0]])
        result = norm.pdf(logt-traj[:,1])/(traj[:,0]*norm.cdf(traj[:,1] - logt))            
    elif ind_exp == 56:
        cos2piz = np.array([math.cos(2*math.pi*x1) for x1 in traj[:,1]])    
        result = 1.5*traj[:,0]**(1/2)*np.exp(-0.5*cos2piz-1.5)
    if ind_exp>=53 and ind_exp<=56:
        for i in range(traj.shape[0]):
            result[i] = result[i]+0.5*math.sin(math.pi*traj[i,2]*traj[i,3])
            if traj[i,4]>0.5:
                result[i] = result[i]+0.5
 
    return result


def cols_to_include (col_names, num_irr):
    col_included = []
    for col in col_names:
        if not col.startswith("X_"):
            col_included.append(True)
        else:
            col_included.append(col <= "X_%d"%num_irr)
    return col_included


@exec_if_not_cached
def _read_synth(exp_num, num_irr):
    data = pd.read_csv(os.path.join(DATA_ADDRESS, f"exp_{exp_num}__num_irr_40__recurring_False__p_drop_0.0__train.csv"))
    col_included = cols_to_include(data.columns, num_irr)
    data = data.loc[:, col_included]

    return data

@exec_if_not_cached
def _read_synth_test(exp_num, num_irr):
    data = pd.read_csv(os.path.join(DATA_ADDRESS, f"exp_{exp_num}__num_irr_40__recurring_False__p_drop_0.0__test.csv"))
    col_included = cols_to_include(data.columns, num_irr)

    data = data.loc[:, col_included]

    col_included = []
    for col in data.columns:
        if col == "t_start" or col.startswith("X_"):
            col_included.append(True)
        else:
            col_included.append(False)
    data = data.loc[:, col_included]
    data.rename(columns = {"t_start": "t"}, 
                  inplace = True)
    
    true_haz = TrueHaz(data.values, 40+exp_num)

    return true_haz, data



hyperparams = {
        "1_0" : {'max_depth':1, 'n_estimators':300},
        "1_20": {'max_depth':1, 'n_estimators':200},
        "1_40": {'max_depth':1, 'n_estimators':250},
        "2_0" : {'max_depth':1, 'n_estimators':300},
        "2_20": {'max_depth':1, 'n_estimators':250},
        "2_40": {'max_depth':1, 'n_estimators':250},
        "3_0" : {'max_depth':2, 'n_estimators':150},
        "3_20": {'max_depth':2, 'n_estimators':50},
        "3_40": {'max_depth':2, 'n_estimators':50},
        "4_0" : {'max_depth':1, 'n_estimators':300},
        "4_20": {'max_depth':1, 'n_estimators':150},
        "4_40": {'max_depth':1, 'n_estimators':200}
}



@run_as_process
def cv_synth(exp_num, num_irr, num_gpu, batch_size):
        
    #from sklearn.utils.estimator_checks import check_estimator
    #check_estimator(boxhed())
    
    param_grid = {'max_depth':    [1, 2, 3, 4, 5],
                  'n_estimators': [50, 100, 150, 200, 250, 300]}

    rslt = {'exp_num':       exp_num, 
            'num_irr':       num_irr, 
            'num_gpu':       num_gpu, 
            'batch_size':    batch_size}

    data = _read_synth(exp_num, num_irr)
    
    boxhed_ = boxhed()
    rslt['num_quantiles'] = num_quantiles
    prep_timer = timer()

    subjects, X, w, delta = boxhed_.preprocess(
            data             = data, 
            #is_cat           = [4],
            num_quantiles = num_quantiles, 
            weighted         = False, 
            nthreads         = batch_size)

    rslt["prep_time"] = prep_timer.get_dur()
    
    if use_gpu:
        gpu_list = _get_free_gpu_list(num_gpu)
    else:
        gpu_list = [-1] 

    if do_CV:
        gridsearch_timer = timer()
        #TODO: handle memory exception if model_per_gpu too large
        cv_rslts, best_params = cv(param_grid, 
                                  X, 
                                  w,
                                  delta,
                                  subjects, 
                                  5,
                                  gpu_list,
                                  batch_size)
    
        rslt["GS_time"] = gridsearch_timer.get_dur()
    else:
        best_params = hyperparams["%d_%d"%(exp_num, num_irr)]

    best_params['gpu_id'] = gpu_list[0]

    #TODO: nthread problem still not solved
    best_params['nthread'] = batch_size

    rslt.update(best_params)
     
    boxhed_.set_params (**best_params)

    fit_timer = timer()
    boxhed_.fit (X, delta, w)
    rslt["fit_time"] = fit_timer.get_dur()

    #TODO: set __all__ for the scripts

    true_haz, test_X = _read_synth_test(exp_num, num_irr) 

    pred_timer = timer()
    preds = boxhed_.predict(test_X)
    rslt["pred_time"] = pred_timer.get_dur()

    rslt["rmse"] = "%.3f"%(np.sqrt(np.square(true_haz - preds).mean(axis=None)))

    _L2 = calc_L2(preds, true_haz)
    rslt["rmse_CI"] = "%.3f (%.3f, %.3f)"%(_L2[0], _L2[1][0], _L2[1][1])
    return rslt


if __name__ == "__main__":
    
    #TODO: if nthread used, this needs to change
    def _rslt_file_name (*args):

        out_str  = curr_dat_time() 
        for arg in args:
            out_str += "__" + arg +"=" \
                    +  "".join(str(eval(arg)).split())

        out_str += ".csv"
        return out_str
    
    rslts = []
    for num_gpu in num_gpus:
        for batch_size in batch_sizes:
 
            #TODO: tqdm
            for exp_num in [1, 2, 3, 4]:
                for num_irr in [0,20,40]:

                    print ('    exp:        ', exp_num)
                    print ('    num_irr:    ', num_irr)
                    print ('    num GPU:    ', num_gpu)
                    print ('    batch size: ', batch_size)
                    print ("")

                    rslt = cv_synth(exp_num, 
                                                  num_irr,
                                                  num_gpu, 
                                                  batch_size)

                    print (rslt, "\n"*3, rslt["rmse"], "\n"*2, sep="")
                    rslts.append(rslt)

    rslt_df = pd.DataFrame(rslts)
    rslt_df_file_name = _rslt_file_name("num_quantiles", "use_gpu", "do_CV", "num_gpus", "batch_sizes")

    print (rslt_df)
    rslt_df.to_csv(os.path.join(RSLT_ADDRESS, rslt_df_file_name),
            index = None)
