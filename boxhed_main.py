import numpy as np
import pandas as pd
import os
from boxhed import boxhed
from utils import timer, curr_dat_time, run_as_process, exec_if_not_cached, _get_free_gpu_list, create_dir_if_not_exist
from grid_search import collapsed_ntree_gs
from preprocessor import preprocessor 
import math

from scipy.stats import beta # beta distribution.
from scipy.stats import norm # normal distribution.

DATA_ADDRESS = "./synth_files/"
RSLT_ADDRESS = "./results/"

for addr in [DATA_ADDRESS, RSLT_ADDRESS]:
    create_dir_if_not_exist(addr)

#TODO: get these from command line?
nom_quant   = 256
grid_search = True#False
use_gpu     = False

# when CPU hist is used, the batch size would be num_gpu * model_per_gpu
nom_gpus = [4, 6]
model_per_gpus = [8, 10]

#%%
# Author: Xiaochen Wang
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


@exec_if_not_cached
def _read_synth(ind_exp, num_irrelevant):
    datname = 'data_exp' + str(ind_exp) + '_numIrr' + str(num_irrelevant)

    long_lotraj = pd.read_csv(os.path.join(DATA_ADDRESS, 'train_long_lotraj_exp%d_numIrr_40.csv'%(ind_exp)), sep=',', header=None) 


    long_lotraj = long_lotraj.iloc[:,0:num_irrelevant+3]#.head(20)
    long_lotraj.columns = ['patient', 't_start']+["X_%i"%i for i in range(1, num_irrelevant+2)]

    long_lotraj['t_end'] = long_lotraj['t_start'].shift(-1)
    long_lotraj['delta'] = 0

    delta = pd.read_csv(os.path.join(DATA_ADDRESS, 'train_delta_exp%d_numIrr_40.csv'%(ind_exp)), sep=',', header=None).values.reshape(-1)

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
    dataname =  'exp' + str(ind_exp) + '_numIrr_%d'
    data = pd.read_csv(os.path.join(DATA_ADDRESS, 'test_random_pick_' + dataname%40 +  '.csv'), sep=',', header=None)

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
        
    #from sklearn.utils.estimator_checks import check_estimator
    #check_estimator(boxhed())
    
    param_grid = {'max_depth':    [1, 2, 3, 4, 5],
                  'n_estimators': [50, 100, 150, 200, 250, 300]}

    rslt = {'ind_exp':       ind_exp, 
            'num_irr':       num_irr, 
            'nom_gpu':       nom_gpu, 
            'model_per_gpu': model_per_gpu}

    data = _read_synth(ind_exp, num_irr)
    
    prep = preprocessor()
    rslt['nom_quant'] = nom_quant
    prep_timer = timer()
    pats, X, y = prep.preprocess(data, nom_quant, True)
    rslt["prep_time"] = prep_timer.get_dur()
    
    if use_gpu:
        gpu_list = _get_free_gpu_list(nom_gpu)
    else:
        gpu_list = [-1] 

    if grid_search:
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
    else:
        best_params = hyperparams["%d_%d"%(ind_exp, num_irr)]

    best_params['gpu_id'] = gpu_list[0]

    #TODO: nthread problem still not solved
    best_params['nthread'] = -1

    rslt.update(best_params)
     
    boxhed_ = boxhed(**best_params)

    fit_timer = timer()
    boxhed_.fit (X,y.iloc[:,0], y.iloc[:,1])
    rslt["fit_time"] = fit_timer.get_dur()

    #TODO: set __all__ for the scripts

    true_haz, test_X = _read_synth_test(ind_exp, num_irr) 

    pred_timer = timer()
    test_x = prep.fix_data_on_boundaries(test_X)
    preds = boxhed_.predict(test_X)
    rslt["pred_time"] = pred_timer.get_dur()

    rslt["rmse"] = "%.3f"%(np.sqrt(np.square(true_haz - preds)).mean(axis=None))

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
    for nom_gpu in nom_gpus:
        for model_per_gpu in model_per_gpus:
 
            #TODO: tqdm
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

                    print (rslt, "\n"*3, rslt["rmse"], "\n"*2, sep="")
                    rslts.append(rslt)

    rslt_df = pd.DataFrame(rslts)
    rslt_df_file_name = _rslt_file_name("nom_quant", "use_gpu", "grid_search", "nom_gpus", "model_per_gpus")

    print (rslt_df)
    rslt_df.to_csv(os.path.join(RSLT_ADDRESS, rslt_df_file_name),
            index = None)
