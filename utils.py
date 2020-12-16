from datetime import datetime
from pytz import timezone
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
from joblib import Parallel, delayed
import itertools
from collections import namedtuple


import warnings
warnings.simplefilter(action='ignore', category=Warning)

import pandas as pd
from pathlib import Path
import os
import sys
sys.path.append(os.path.join(os.path.expanduser("~"), "survival_analysis/BoXHED2.0/xgboost/python-package/"))

CACHE_ADDRESS = './CACHE/'

#log("loading XGB")
import xgboost as xgb

def curr_dat_time ():
    curr_dt = datetime.now(timezone("US/Central"))
    return curr_dt.strftime("%a, %b %d, %H:%M:%S")

def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)


def exec_if_file_not_exists(file_name, saved_type, func, args):

    if   saved_type == "pickle":
        file_name += ".pkl"

    elif saved_type == "csv":
        file_name += ".csv"

    else:
        raise Exception('Saved type not properly specified!')


    file_path = os.path.join(CACHE_ADDRESS, file_name)

    if Path(file_path).is_file():
        if   saved_type == "pickle":
            with open( file_path, "rb" ) as file_handle:
                return pickle.load(file_handle)

        elif saved_type == "csv":
            return pd.read_csv(file_path)

    else:
        create_dir_if_not_exist(os.path.dirname(file_path))
        output = func(**args)

        if   saved_type == "pickle":
            with open( file_path, "wb" ) as file_handle:
                pickle.dump(output, file_handle)

        elif saved_type == "csv":
            print ("WARNING: writing CSV without header and space seperated!")
            output.to_csv(file_path, index=None, header = None, sep = " ")

        return output


def log(message):
    print (curr_dat_time(), message, "...")

def change_format(lotraj, delta):
    print (lotraj)

    lotraj['start_time'] = lotraj['t']  
    lotraj['end_time']   = -1

    lotraj['end_time'].iloc[0:-1] = lotraj['start_time'].iloc[1:].values
    lotraj = lotraj.loc[lotraj['end_time']>lotraj['start_time']]
    
    lotraj['tmp_idx'] = range(len(lotraj))
    def delta_determiner (row):

        idx = int(row['tmp_idx'])

        def get_delta_val(ID):
            return (delta['delta'].iloc[int(ID-1)])

        if (idx == len(lotraj)-1):
            return get_delta_val(row['ID'])

        if (lotraj['ID'].iloc[idx] == lotraj['ID'].iloc[idx+1]):
            return 0

        return get_delta_val(row['ID'])
                
    lotraj['delta'] = lotraj.apply(lambda row: delta_determiner(row), axis = 1)

    lotraj.drop(columns = ['tmp_idx'], inplace=True)
    lotraj.reset_index(drop=True, inplace=True)
    
    return lotraj


def read_survival_data(data_dir, ind_exp, num_irr):

    lotraj = pd.read_csv(os.path.join(data_dir, 'train_long_lotraj_exp%d_numIrr_40.csv'%(ind_exp)), sep=',', header=None, names=['ID', 't']+['X%d'%i for i in range (0, 40+1)], index_col=False)
    
    delta  = pd.read_csv(os.path.join(data_dir, 'train_delta_exp%d_numIrr_40.csv'%(ind_exp)), sep=',', header=None, names = ['delta'], index_col=False) 

    lotraj = lotraj.iloc[:,0:num_irr+3]

    data = change_format(lotraj, delta) 

    def obs_interval(row):
        lower_bound = row['end_time']
        upper_bound = row['end_time']

        if row['delta']==0:
            upper_bound = np.inf

        return pd.Series((lower_bound, upper_bound))

    data[['obs_lb', 'obs_ub']] = data.apply(lambda row: obs_interval(row), axis=1)

    return data


def create_patient_boundaries(data):

    prev_id = None
    patient_entries = []
    curr_patient = None
    for index, row in data.iterrows():
        
        is_first_entry =  prev_id != row["ID"]
        is_last_entry  =  (index == len(data)-1) or (row["ID"] != data["ID"].iloc[index+1])
        
        if index ==0:
            assert (is_first_entry)

        if is_first_entry:
            curr_patient = {"start index":index, "end index":-1}

        if is_last_entry:
            curr_patient["end index"] = index
            patient_entries.append (curr_patient)
            curr_patient = None
        
        prev_id = row["ID"]
        
    return pd.DataFrame(patient_entries)


def dataframe_to_dmatrix(data):
    dtrain = xgb.DMatrix(data.drop(columns=['ID', 'start_time', 'end_time', 'delta', 'obs_lb', 'obs_ub']))
    
    dtrain.set_float_info('label',  data['delta'])
    dtrain.set_float_info('weight', data['end_time'] - data['start_time'])

    return dtrain

def dataframe_to_dmatrix_BoXHED1(data, include_dt_delta = True):
    #'''
    #print (data['data'])
    #temp = np.copy(data['data'][:, 0])
    #data['data'][:, 0] = data['data'][:, 1]
    #data['data'][:, 1] = temp
    #print (data['data'].shape)
    #raise
    #'''

    #print ("FIX ME! utils.py:176")
    #raise
    #if not include_dt_delta:

    
    dtrain = xgb.DMatrix(data['data'])
    if include_dt_delta == False:
        return dtrain

    #print (data['dt'].mean(), data['dt'].min(), data['dt'].max())
    #raise
    #raise
    
    dtrain.set_float_info('label',  data['delta'])
    dtrain.set_float_info('weight', data['dt'])

    return dtrain



def read_dtrain(data_dir, ind_exp, num_irr, nfolds = 1):

    data = exec_if_file_not_exists("surv_train_data__exp_%d__num_irr_%d"%(ind_exp, num_irr), "pickle", read_survival_data, {'data_dir':data_dir, 'ind_exp':ind_exp, 'num_irr':num_irr})
    
    #raise
    assert nfolds >= 1

    if nfolds==1:
        return dataframe_to_dmatrix(data)

    skf = StratifiedKFold(n_splits = nfolds, random_state=666)

    return [{'data_train':data.iloc[train_idx], 'data_test':data.iloc[test_idx]} for train_idx, test_idx in skf.split(data, data['delta'])]

def getData_BoXHED1(lotraj, delta, tpart_size, xpart_size):
    import sys
    sys.path.insert(0, '/home/grads/a/a.pakbin/survival_analysis/BoXHED2.0/BoXHED1/')
    from hazardboost import getTimePartition, prepData

    tpart = getTimePartition(lotraj, tpart_size)

    trajEndPoint = np.zeros((len(lotraj), lotraj[0].shape[1]))
    i = 0
    traj = lotraj[i]
    interp, dt, trajEnd = prepData(traj, tpart)
    delta_ = np.zeros_like(dt)
    delta_[-1] = delta[i]

    trajEndPoint[i,:] = trajEnd
    mergedData = interp
    mergedDT = dt
    mergedDelta = delta_

    for i in range(1, len(lotraj)):
        traj = lotraj[i]
        interp, dt, trajEnd = prepData(traj, tpart)
        trajEndPoint[i,:] = trajEnd
        delta_ = np.zeros_like(dt)
        delta_[-1] = delta[i]

        mergedData = np.vstack((mergedData,interp))
        mergedDT = np.vstack((mergedDT, dt))
        mergedDelta = np.vstack((mergedDelta, delta_))
    
    #aa = np.sort(mergedData[:,0])
    #x = [aa[int(len(aa)*i/len(tpart))] for i in range(len(tpart))]
    #print (x)
    #raise
    print (tpart)
    TOTALTIME = 0.0
    for traj in lotraj:
        TOTALTIME += traj[-1,0] - traj[0,0]
    F0 = np.log(np.sum(delta)/TOTALTIME)

    return (F0, mergedDelta, mergedData, mergedDT)


def read_dtrain_BoXHED1 (ind_exp, num_irrelevant, tpart_size=20, xpart_size=20, nfolds=1):

    dir = '/home/grads/a/a.pakbin/survival_analysis/FlexSurv/exp' + str(ind_exp) + '/Feb_19_irrelevant_features/'
    datname = 'data_exp' + str(ind_exp) + '_numIrr' + str(num_irrelevant)

    dat = pickle.load(open(dir + datname + '.pkl', 'rb'))
    lotraj = dat[0]
    delta = dat[1]
    '''
    if ind_exp in [41, 42]:
        AA = 1
    else:
        AA = 5
    _LOTRAJ = []
    for _lotraj in lotraj:
        _lotraj[:,0]=_lotraj[:,0]/AA
        _LOTRAJ.append(_lotraj)
    lotraj = _LOTRAJ
    '''
    #import sys
    #sys.path.insert(0, '/home/grads/a/a.pakbin/survival_analysis/BoXHED2.0/BoXHED1/')
    #from hazardboost import getTimePartition
    #print ([round (x,3) for x in getTimePartition(lotraj, 20)])
    #print ([x for x in getTimePartition(lotraj, tpart_size)])
    #raise

    def write_tpart_xpart_tmp(lotraj, tpart_size, mergedData, xpart_size):
        import sys
        sys.path.insert(0, '/home/grads/a/a.pakbin/survival_analysis/BoXHED2.0/BoXHED1/')
        from hazardboost import getTimePartition, prepData

        tpart = getTimePartition(lotraj, tpart_size)
        #print (tpart)
        with open('/home/grads/a/a.pakbin/survival_analysis/BoXHED2.0/TEST_DIR/CACHE/'+'tpart.tmp', 'w') as filehandle:
            for tpart_ in tpart:
                filehandle.write('%s\n' % tpart_)

        percentiles = np.linspace(0, 100, xpart_size+1)[1:] 
        xpart = np.percentile(mergedData[:,1], percentiles)
        with open('/home/grads/a/a.pakbin/survival_analysis/BoXHED2.0/TEST_DIR/CACHE/'+'xpart.tmp', 'w') as filehandle:
            for xpart_ in xpart:
                filehandle.write('%s\n' % xpart_)
        print (tpart)
        print (len(tpart))
        #print (xpart)

    #raise


    #write_tpart_xpart_tmp(); 
    #TODO: F0 should be computed for all train test splits individually

    F0, delta, data, dt = exec_if_file_not_exists(datname+"_tpart_%d__delta_data_dt_BoXHED1"%tpart_size, "pickle", getData_BoXHED1, {'lotraj': lotraj, 'delta': delta, 'tpart_size':tpart_size, 'xpart_size':xpart_size})

    write_tpart_xpart_tmp(lotraj, tpart_size, data, xpart_size)

    if nfolds==1:
        #raise (Exception("not implemented!"))
        #print ("______nfolds_____")
        #MTC = np.sort(data[:,0])
        #tpart_size += 1
        #print (np.array([MTC[int(len(MTC)*i/tpart_size)] for i in range(tpart_size)]))

        #raise
        return {'data':data, 'delta':delta, 'dt':dt, 'f0':F0} 

    skf = StratifiedKFold(n_splits = nfolds, random_state=666)

    #print ("WARNING: data debugging in progress.")
    #return ["dat_%d"%x for x in range(nfolds)]
    return [{'data_train':{'data':data[train_idx], 'delta':delta[train_idx], 'dt':dt[train_idx], 'f0':F0}, 'data_test':{'data':data[test_idx], 'delta':delta[test_idx], 'dt':dt[test_idx]}} for train_idx, test_idx in skf.split(data, delta)]
    


def read_dtrain_BoXHED2 (ind_exp, num_irrelevant, tpart_size=20, xpart_size=20, nfolds=1):

    datname = 'data_exp' + str(ind_exp) + '_numIrr' + str(num_irrelevant)

    def _BoXHED1_to_BoXHED2(ind_exp, num_irrelevant):
        #num_irrelevant = 20
        dir = '/home/grads/a/a.pakbin/survival_analysis/FlexSurv/exp' + str(ind_exp) + '/Feb_19_irrelevant_features/'
        long_lotraj = pd.read_csv(dir + 'train_long_lotraj_exp%d_numIrr_40.csv'%(ind_exp), sep=',', header=None) 

        #print (long_lotraj.head())
        #data.columns =
        #print (['delta', 't_start', 'patient', 't_start']+["X_%i"%i for i in range(1, num_irrelevant+2)])
        
        #print (long_lotraj)
        long_lotraj = long_lotraj.iloc[:,0:num_irrelevant+3]#.head(20)
        long_lotraj.columns = ['patient', 't_start']+["X_%i"%i for i in range(1, num_irrelevant+2)]

        long_lotraj['t_end'] = long_lotraj['t_start'].shift(-1)
        long_lotraj['delta'] = 0

        delta = pd.read_csv(dir + 'train_delta_exp%d_numIrr_40.csv'%(ind_exp), sep=',', header=None).values.reshape(-1)
    
        #print (long_lotraj)
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


        #long_lotraj['_last_entry'] = long_lotraj['patient'].diff(-1).astype(bool)

        #long_lotraj['t_end'] = long_lotraj['t_start'].shift(-1)
        #delta = pd.read_csv(dir + 'train_delta_exp%d_numIrr_40.csv'%(ind_exp), sep=',', header=None).values.reshape(-1)

        #def _set_delta(_last_entry, pat_id):
        #    if not _last_entry:
        #        return 0
        #    return delta[pat_id-1]

        #long_lotraj['delta'] = long_lotraj.apply(lambda x: _set_delta(x['_last_entry'], x['patient']), axis = 1)
        #long_lotraj = long_lotraj.loc[:, [not x.startswith("_") for x in long_lotraj.columns]]
        #print (data)

        return F0, data
    

    #TODO: F0 should be computed for all train test splits individually

    F0, data = exec_if_file_not_exists(datname+"_delta_data_dt_BoXHED2", "pickle", _BoXHED1_to_BoXHED2, {'ind_exp':ind_exp, 'num_irrelevant':num_irrelevant})
    #print (data)
    data['dt']=data['t_end']-data['t_start']
    data.drop(columns = ['t_end'], inplace = True)
    #print (data)
    #raise 
    #t_start       X_1  delta        dt
    delta = data['delta']
    dt = data['dt']
    data = data.drop(columns = ['delta', 'dt'])
    '''
    raise

    from preprocessor import preprocessor
    prep = preprocessor()

    import time

    preprocess_start_time = time.time()

    preprocessed = prep.preprocess(data, tpart_size)

    pats, X, y   = preprocessed
    preprocessed = pd.concat([X,y], axis = 1)
    #print (preprocessed)
    #raise

    print (preprocessed)
    preprocessing_time = time.time() - preprocess_start_time
    print ('prep time:', preprocessing_time)


    print (preprocessed['t_start'].unique())
    
    delta = preprocessed['delta']
    dt    = preprocessed['dt']
    preprocessed.drop(columns = ['delta', 'dt'], inplace = True)
    data = preprocessed
    '''

    if nfolds==1:
        #raise
        return {'data':data, 'delta':delta, 'dt':dt, 'f0':F0} 

    skf = StratifiedKFold(n_splits = nfolds, random_state=666)

    #print ("WARNING: data debugging in progress.")
    #return ["dat_%d"%x for x in range(nfolds)]
    return [{'data_train':{'data':data[train_idx], 'delta':delta[train_idx], 'dt':dt[train_idx], 'f0':F0}, 'data_test':{'data':data[test_idx], 'delta':delta[test_idx], 'dt':dt[test_idx]}} for train_idx, test_idx in skf.split(data, delta)]
    


from py3nvml import get_free_gpus
from time import sleep

def free_gpu_list(min_needed = 4, nom_attempts = 10, idle=15, verbose = False):
    while (nom_attempts>=0):
        free_gpus = get_free_gpus()
        nom_free_gpus = free_gpus.count(True)
        if verbose:
            print ("nom GPUs avail:", nom_free_gpus)

        if nom_free_gpus < min_needed :
            nom_attempts -=1
            if nom_attempts == 0:
                raise "no GPU available"
            print ("not enough GPUs available, sleeping for %d sec..."%idle)
            sleep (idle)
            continue
        break

    free_gpu_list_ = [gpu_id for (gpu_id, gpu_free) in enumerate(free_gpus) if gpu_free == True]
    if min_needed == 1:
        return free_gpu_list_[0]

    return (free_gpu_list_)

GPU_LIST = []
def list_of_product_dicts (names, *args):
    args = list(args)[0]
    assert len(names) == len(args), "names and args' arguments should be of the same length!"
    #return pd.DataFrame([x for x in itertools.product(*args)], columns = names).to_dict("records")
    args_df =pd.DataFrame([x for x in itertools.product(*args)], columns = names)  
    
    #print ("_____VERBOSE_____")
    GPU_LIST = free_gpu_list(min_needed = 2, verbose = True)
    args_df['gpu_id'] = [GPU_LIST[int(x*len(GPU_LIST)/len(args_df))] for x in range(len(args_df))]

    return args_df.to_dict("records")

def do_parallel (func, args, njobs, prefer = "threads", parallel = True):

    if parallel:
        return Parallel( n_jobs = njobs, prefer = prefer)(delayed(func)(**_args) for _args in args)

    print ("WARNING: Running sequentially!")
    results = []
    for _args in args:
        results.append(func(**_args))
    return np.array(results)


def cv_parallel(func, args, arg_names, njobs=-1, parallel = True, prefer = "threads"):

    assert arg_names[-1] == "data", "data should be the last argument!"

    args_ = list_of_product_dicts(arg_names, args)

    if njobs == -1:
        njobs = len(args_)
        
    rslts = do_parallel (func, args_, njobs, prefer, parallel)

    nfolds = len(args[arg_names.index("data")])
    assert len(rslts)%nfolds == 0

    arg_prfmnc_list = []
    arg_prfmnc = namedtuple("arg_prfmnc", "args val")
    for idx in range(int(len(rslts)/nfolds)):
        rslt_slice_avg = np.mean(rslts[idx*nfolds:(idx+1)*nfolds])

        _args_ = args_[idx*nfolds]

        _args_.pop('data')
        _args_.pop('ind_exp')
        _args_.pop('eval_metric')

        arg_prfmnc_list.append(arg_prfmnc(args = _args_, val=rslt_slice_avg))

    sorted_arg_prfmnc_list = sorted(arg_prfmnc_list, key=lambda entry: entry.val)

    return sorted_arg_prfmnc_list[0].args, sorted_arg_prfmnc_list[-1].args


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




import time
def time_now():
    return time.time()

class timer:

    def __init__(self):
        self.t_start = time_now()

    def get_dur(self):
        return round(time_now()-self.t_start, 3)
