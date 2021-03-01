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
grid_search = False#True
use_gpu     = False

# when CPU hist is used, the batch size would be num_gpu * model_per_gpu
nom_gpus = [1]#[4, 6]
model_per_gpus = [20]#[8, 10]
keep_probs = [0.7, 0.8, 0.9, 1]
num_bs     = 20

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
    data.columns = ['t_start']+["X_%i"%i for i in range(1, num_irrelevant+2)]
    
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


def drop_rows(data, prob=0.5):
    data_sub = data.sample(frac=prob, random_state = np.random.RandomState(), replace=False).sort_index()
    
    patient_converter = dict(
                            zip(sorted(data_sub['patient'].unique()), 
                            range(1, 1+data_sub['patient'].nunique()))
                            )

    data_sub = data_sub.replace({'patient':patient_converter})

    return data_sub

@run_as_process
def grid_search_test_synth(ind_exp, num_irr, nom_gpu, model_per_gpu, keep_prob):
        
    #from sklearn.utils.estimator_checks import check_estimator
    #check_estimator(boxhed())
    
    param_grid = {'max_depth':    [1, 2, 3, 4, 5],
                  'n_estimators': [50, 100, 150, 200, 250, 300]}

    rslt = {'ind_exp':       ind_exp, 
            'num_irr':       num_irr, 
            'nom_gpu':       nom_gpu, 
            'model_per_gpu': model_per_gpu}

    data = _read_synth(ind_exp, num_irr)
    #data = pd.read_csv("TEST.txt")
    #print (data)

    data = drop_rows(data, keep_prob)
    rslt["keep_prob"] = keep_prob
    #nom_quant = 10
    
    prep = preprocessor()
    rslt['nom_quant'] = nom_quant
    prep_timer = timer()
    subjects, X, w, delta = prep.preprocess(
            data             = data, 
            quant_per_column = nom_quant, 
            weighted         = True, 
            nthreads         = 1)

    rslt["prep_time"] = prep_timer.get_dur()
    #raise
    
    if use_gpu:
        gpu_list = _get_free_gpu_list(nom_gpu)
    else:
        gpu_list = [-1] 

    if grid_search:
        #TODO fix y to delta,w in grid search
        gridsearch_timer = timer()
        #TODO: handle memory exception if model_per_gpu too large
        cv_rslts, best_params = collapsed_ntree_gs(boxhed(), 
                                  param_grid, 
                                  X, 
                                  w,
                                  delta,
                                  subjects, 
                                  5,
                                  gpu_list,
                                  model_per_gpu,
                                  -1)
    
        rslt["GS_time"] = gridsearch_timer.get_dur()
    else:
        best_params = hyperparams["%d_%d"%(ind_exp, num_irr)]

    best_params['gpu_id'] = gpu_list[0]

    #TODO: nthread problem still not solved
    best_params['nthread'] = nom_gpu*model_per_gpu#-1

    rslt.update(best_params)
     
    boxhed_ = boxhed(**best_params)

    fit_timer = timer()
    boxhed_.fit (X, delta, w)
    rslt["fit_time"] = fit_timer.get_dur()

    #TODO: set __all__ for the scripts

    true_haz, test_X = _read_synth_test(ind_exp, num_irr) 

    pred_timer = timer()
    test_X = prep.fix_data_on_boundaries(test_X)
    preds = boxhed_.predict(test_X)
    rslt["pred_time"] = pred_timer.get_dur()

    #print (np.array(abs(true_haz-preds)).mean(axis=None))
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
    for nom_gpu in nom_gpus:
        for model_per_gpu in model_per_gpus:
            for keep_prob in keep_probs:
 
                #TODO: tqdm
                for ind_exp in [41, 42, 43, 44]:
                    for num_irr in [0,20,40]:

                        num_bs_ = num_bs
                        if keep_prob == 1:
                            num_bs_ = 1
                        for it in range(num_bs_):
                            print ('    exp:      ', ind_exp)
                            print ('    num_irr:  ', num_irr)
                            print ('    nom GPU:  ', nom_gpu)
                            print ('    /GPU:     ', model_per_gpu)
                            print ('    keep_prob:', keep_prob)
                            print ('    bs it/all:', "%d/%d"%(it+1,num_bs_))
                            print ("")

                            rslt = grid_search_test_synth(ind_exp, 
                                                          num_irr,
                                                          nom_gpu, 
                                                          model_per_gpu,
                                                          keep_prob)

                            print (rslt, "\n"*3, rslt["rmse"], "\n"*2, sep="")
                            ####
                            #it_file_name = "_it=%d__"%(it+1)+_rslt_file_name("nom_quant", "ind_exp", "num_irr", "keep_prob")
                            #pd.DataFrame([rslt]).to_csv(os.path.join(RSLT_ADDRESS, it_file_name),index = None)
                            ####
                            rslts.append(rslt)

    rslt_df = pd.DataFrame(rslts)
    rslt_df_file_name = _rslt_file_name("nom_quant", "use_gpu", "grid_search", "nom_gpus", "model_per_gpus", "keep_probs")

    print (rslt_df)
    rslt_df.to_csv(os.path.join(RSLT_ADDRESS, rslt_df_file_name),
            index = None)
