import pandas as pd
import numpy as np
import os
from main import TrueHaz
from utils import create_dir_if_not_exist
import itertools
from tqdm import tqdm

#integrals
from scipy.stats import norm, beta
import math

def event_time (t0, x, u, exp_num):

    if exp_num == 0:
        lambda_ = 1 if x==0 else 2
        return -math.log(u)/lambda_ + t0

    if exp_num == 1:
        return beta.ppf(
                -math.log(u)/beta.pdf(x,2,2)+beta.cdf(t0,2,2),
                2,2)

    if exp_num == 2:
        return beta.ppf(
                -math.log(u)/beta.pdf(x,4,4)+beta.cdf(t0,4,4),
                4,4)

    if exp_num == 3:
        return np.exp(x-norm.ppf(u*norm.cdf(x-math.log(t0))))

    if exp_num == 4:
        return np.power(
                -math.log(u)/(np.exp(-0.5*math.cos(2*math.pi*x)-1.5)) +
                np.power(t0, 1.5),
                2/3)
    
file_addr = "./synth_files/"
create_dir_if_not_exist(file_addr)

num_irr = 40
t_min     = 0.01
t_max = {
        1: 1,
        2: 1,
        3: 5,
        4: 5
        }
num_pcs   = 10
max_size = int(14e6)

n_sub = {
            'train':1000000,
            'test': 5000
        }

seed = {
            'train':0,
            'test': 1
        }

def set_subj_1_to_N(data):
    subject_converter = dict(
                            zip(sorted(data['subject'].unique()), 
                            range(1, 1+data['subject'].nunique()))
                            )

    data = data.replace({'subject':subject_converter})
    return data

def create_synth_data(exp_num, mode, recurring, p_dropout):
    print ("exp num:%d    recurring:%r    p_drop:%.1f    mode:%s   started ..."%(exp_num, recurring, p_dropout, mode))
    np.random.seed(seed[mode])

    synth_data = [] 
    for i in tqdm(range(n_sub[mode])):
        t_curr = t_min
        while True:
            if t_curr >= t_max[exp_num]:
                break
            len_traj = np.random.uniform(0, t_max[exp_num]/num_pcs)
            synth_data.append({
                        "subject":i+1,
                        "t_start":t_curr,
                        "t_end"  :min(t_curr+len_traj, t_max[exp_num])
                              })
            t_curr += len_traj
     
    synth_data = pd.DataFrame(synth_data)

    synth_covs = np.random.uniform(size = (synth_data.shape[0], num_irr+1))
    synth_covs = pd.DataFrame(data = synth_covs, columns = ["X_%d"%i for i in range(num_irr+1)])

    synth_data = pd.concat([synth_data, synth_covs], axis=1)
    synth_data['delta'] = 0

    synth_censored_total = np.zeros((max_size, num_irr+5))
    curr_idx = 0
    for subject, subject_data in tqdm(synth_data.groupby(['subject'])):
        synth_censored = []

        subject_data.reset_index(drop = True, inplace = True)
        for index, subject_traj in subject_data.iterrows():
            is_censored = np.random.binomial(1, p_dropout)
            if is_censored:
                continue

            t_start = subject_traj['t_start']
            t_end   = subject_traj['t_end']
            x       = subject_traj['X_0']
            t_curr = t_start
            while True:
                new_traj = subject_traj.copy(deep=True)
                new_traj['t_start'] = t_curr
                u = np.random.uniform()
                event_time_ = event_time (t_curr, x, u, exp_num)
                if math.isnan(event_time_) or event_time_ > t_end:
                    synth_censored.append(new_traj)
                    break
                new_traj['t_end'] = event_time_
                new_traj['delta'] = 1
                synth_censored.append(new_traj)
                if not recurring:
                    break
                t_curr = event_time_

            if new_traj['delta']==1 and not recurring:
                break

        subj_len = len(synth_censored)
        synth_censored_total[curr_idx:curr_idx + subj_len, :] = pd.DataFrame(synth_censored).values

        curr_idx += subj_len
    
    synth_censored_total = synth_censored_total[:curr_idx, :]
    synth_censored = pd.DataFrame(synth_censored_total)   

    synth_censored.columns = ['subject', 't_start', 't_end'] + ["X_%d"%i for i in range(num_irr+1)] + ['delta']

    synth_censored.reset_index(drop = True, inplace = True) 

    assert synth_censored['subject'].nunique() == n_sub[mode]
    
    print ("exp num:%d    recurring:%r    p_drop:%.1f    mode:%s   saving to disk ..."%(exp_num, recurring, p_dropout, mode))
    synth_censored.to_feather(
            os.path.join(file_addr, "__LARGE_exp_%d__num_irr_%d__recurring_%r__p_drop_%.1f__%s.ftr"%(exp_num, num_irr, recurring, p_dropout, mode)))

    print ("exp num:%d    recurring:%r    p_drop:%.1f    mode:%s   done ..."%(exp_num, recurring, p_dropout, mode))

from joblib import Parallel, delayed

create_synth_data(0, "train", False, 0)
create_synth_data(0, "test",  False, 0)

exit()
Parallel(n_jobs=20)(delayed(create_synth_data)(exp_num, mode, recurring, p_dropout) 
        for (exp_num, mode, recurring, p_dropout) in itertools.product([1, 2, 3, 4], ["train", "test"], [True, False], [0, 0.1, 0.2, 0.3])
        )
