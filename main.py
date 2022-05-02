import os
import pandas as pd
from boxhed import boxhed
from model_selection import cv
from utils import timer, run_as_process, TrueHaz, calc_L2, create_dir_if_not_exist

# BoXHED 2.0 (https://arxiv.org/pdf/2103.12591.pdf) is a software package
# for estimating hazard functions nonparametrically via gradient boosting. 
# It is orders of magnitude faster than BoXHED 1.0 (http://proceedings.mlr.press/v119/wang20o/wang20o.pdf).
# BoXHED 2.0 also allows for more general forms of survival data including recurrent events.

#This tutorial demonstrates how to apply BoXHED 2.0 to a synthetic dataset.

DATA_ADDRESS = "./data/"     # train/test data directory
RSLT_ADDRESS = "./results/"  # results directory

nthread_prep    = 20    # number of CPU threads used for preprocessing
nthread_train   = 20    # number of CPU threads used for training

# Create the results' directory if it does not exist
for addr in [RSLT_ADDRESS]:
    create_dir_if_not_exist(addr)



# Function: read_train_data
# Reads the synthetic training data.
# Input:
#      None
# Return: 
#      @ A pandas dataframe containing training data with the following columns:
#            * ID:      subject ID
#            * t_start: the start time of an epoch for the subject
#            * t_end:   the end time of the epoch
#            * X_i:     values of covariates between t_start and t_end
# Sample Output:
# ID	t_start         t_end           X_0             delta
#	1       0.010000	0.064333	0.152407	0.0
#	1       0.064333	0.135136	0.308475	0.0
#	1       0.194810	0.223106	0.614977	1.0
#	1       0.223106	0.248753	0.614977	0.0
#	2       0.795027	0.841729	0.196407	1.0
#	2       0.841729	0.886587	0.196407	0.0
#	2       0.886587	0.949803	0.671227	0.0
#
# For each epoch (row) we must have t_start < t_end. Also, the beginning of one epoch cannot start earlier than
# the end of the previous one, i.e. t_end_i <= t_start_{i+1}.
#
# Delta denotes whether an event (possibly recurring) occurred at the end of the epoch.
#
# For covariates with missing values, BoXHED 2.0 implements tree splits of the form:
# Left daughter node: {x<=split.point or x is missing}; Right daughter node: {x>split.point}
# or
# Left daughter node: {x<=split.point}; Right daughter node: {x>split.point or x is missing}.
# Alternatively, missing values may be manually imputed, e.g. by carrying forward the most recent value.
def read_train_data():
    return pd.read_csv(os.path.join(DATA_ADDRESS, 'training.csv'))



# Function: read_test_data
# Reads the synthetic testing data. The values of the true hazard function are also provided for accuracy comparisons.
# Input:
#      None
# Return: 
#      @ A pandas dataframe containing testing data with the following columns:
#            * t:   time
#            * X_i: covariate values at time t
#      @ A numpy array containing the values of the true hazard function for each row above
def read_test_data():
    test_data = pd.read_csv(os.path.join(DATA_ADDRESS, 'testing.csv'))
    return test_data, TrueHaz(test_data[['t', 'X_0']].values)



# Function: cv_train_BoXHED2
# Select hyperparameters by K-fold cross-validation, and then fit the BoXHED2.0 hazard estimator
# Input:
#      @ A pandas dataframe containing training data
# Return: 
#      @ The fitted BoXHED2.0 hazard estimator
#      @ A dictionary containing timings of different components.
#@run_as_process
def cv_train_BoXHED2(train_data, a, b):
    # Define the output dictionary
    train_info_dict = {}

    # Preprocess the training data. THIS ONLY NEEDS TO BE DONE ONCE.
    boxhed_ = boxhed()                            # Create an instance of BoXHED
    prep_timer = timer()                          # Initialize timer
    # boxhed.preprocess():
    # Input:
    #      @ num_quantiles: the number of candidate split points to try for time and for each covariate. 
    #                       The locations of the split points are based on the quantiles of the training data.
    #      @ is_cat:        a list of the column indexes that contain categorical data. The categorical data must be one-hot encoded.
    #                       For example, is_cat = [4,5,6] if a categorical variable with 3 factors is transformed into binary-valued columns 4,5,6
    #      @ weighted:      if set to True, the locations of the candidate split points will be based on weighted quantiles 
    #                       (see Section 3.3 of the BoXHED 2.0 paper)      
    #      @ nthreads:      number of CPU threads to use for preprocessing the data
    # Return: 
    #      @ ID:            subject ID for each row in the processed data frames X, w, and delta
    #      @ X:             each row represents an epoch of the transformed data, and contains the values of the covariates as well as
    #                       its start time
    #      @ w:             length of each epoch     
    #      @ delta:         equals one if an event occurred at the end of the epoch; zero otherwise
    ID, X, w, delta = boxhed_.preprocess(
            data          = train_data, 
            #is_cat       = [],
            num_quantiles = 256, 
            weighted      = False, 
            nthread       = nthread_prep)
    train_info_dict["prep_time"] = prep_timer.get_dur()  # calling the get_dur() function.
    
    # Perform K-fold cross-validation to select hyperparameters {tree depth, number of trees, learning rate} if do_CV = True.
    # Otherwise, users should manually specify hyperparameter values. Note that a tree of depth k has 2^k leaf nodes.
    do_CV = False         
    param_manual = {'max_depth':a, 'n_estimators':b, 'eta':0.1}
    
    # Specify the candidate values for the hyperparameters to cross-validate on (more trees and/or deeper trees may be needed for other datasets).
    param_grid = {
        'max_depth':    [1, 2, 3],#, 4, 5],
        'n_estimators': [50, 100, 150, 200, 250, 300],#[10, 20, 30, 40, 50, 60, 70, 80, 90, 100],#,
        'eta':          [0.1]
    }
    
    # Next, specify:
    #      @ gpu_list:    the list of GPU IDs to use for training. Set gpu_list = [-1] to use CPUs.
    #      @ batch_size:  the maximum number of BoXHED2.0 instances trained at any point in time. Example: Performing
    #                     10-fold cross-validation using the param_grid above requires training 5*6*10 = 300
    #                     instances in total.
    #                           * When gpu_list = [-1], batch_size specifies the number of CPU threads to be used, 
    #                             with each one training one instance at a time.
    #                           * When using GPUs, each GPU trains at most batch_size/len(gpu_list) instances at a time. Hence
    #                             if 2 GPUs are used and batch_size = 20, each GPU will train at most 10 instances at a time.
    gpu_list   = [-1]
    batch_size = 20
    num_folds  = 5
    if do_CV:
        cv_timer = timer()
        # Call the cv function to perform K-fold cross validation on the training set. 
        # This outputs the cross validation results for the different hyperparameter combinations.
        # Return: 
        #      @ cv_rslts:    mean and st.dev of the log-likelihood value for each hyperparameter combination
        #      @ best_params: The hyper-parameter combination where the mean log-likelihood value is maximized.
        #                     WE STRONGLY RECOMMEND AGAINST USING THIS COMBINATON. Instead, use the
        #                     one-standard-error rule to select the simplest model that is within st.dev/sqrt(k)
        #                     of the maximum log-likelihood value. See §7.10 in 'Elements of Statistical Learning'
        #                     by Hastie et al. (2009).
        cv_rslts, best_params = cv(param_grid, 
                                  X, 
                                  w,
                                  delta,
                                  ID, 
                                  num_folds,
                                  gpu_list,
                                  batch_size)
        print (best_params)
        
        import pickle
        #with open("./results/popo.pkl", "wb") as f:
        #    pickle.dump(cv_rslts, f)
        #exit()
        train_info_dict["CV_time"] = cv_timer.get_dur()
    else:
        best_params = param_manual

    #print (param_manual)
    best_params['gpu_id'] = gpu_list[0] # Use the first GPU in the list for training
    best_params['nthread'] = nthread_train 

    train_info_dict.update(best_params)
    boxhed_.set_params(**best_params)
    
    # Fit BoXHED to the training data
    fit_timer = timer()
    boxhed_.fit(X, delta, w)
    train_info_dict["fit_time"] = fit_timer.get_dur()

    return boxhed_, train_info_dict



# Function: testRMSE_BoXHED2
# Calculate the RMSE of the fitted hazard estimator if the true hazard function is available
# Input:
#      @ A fitted BoXHED2.0 hazard estimator
#      @ A numpy matrix containing testing data
#      @ A numpy vector containing true hazard for training data
# Return:
#      @ A dictionary containing RMSE (with 95% CIs) and timings of different components.
def testRMSE_BoXHED2(boxhed_, test_X, test_true_haz):
    # Define the output dictionary
    test_info_dict = {}

    # Use the fitted model to estimate the value of the hazard function for each row of the test set:
    pred_timer = timer()
    preds = boxhed_.predict(test_X)

    #print (preds.min(), preds.max())
    #exit()
    test_info_dict["pred_time"] = pred_timer.get_dur()

    # Compute the RMSE of the estimates, and its 95% confidence interval:
    L2 = calc_L2(preds, test_true_haz)
    test_info_dict["rmse_CI"] = f"{L2['point_estimate']:.3f} ({L2['lower_CI']:.3f}, {L2['higher_CI']:.3f})"

    return test_info_dict

import pickle
def dump_pickle(obj, addr):
    with open(addr, 'wb') as handle:
        pickle.dump(obj, handle)


def load_pickle(addr):
    with open(addr, 'rb') as handle:
        obj = pickle.load(handle)
    return obj

hypers = {
    "1_0":  (3, 50),
    "1_20": (3, 50),
    "1_40": (3, 50),
    "2_0":  (4, 50),
    "2_20": (3, 50),
    "2_40": (3, 50),
    "3_0":  (3, 50),
    "3_20": (2, 50),
    "3_40": (2, 50),
    "4_0":  (3, 50),
    "4_20": (3, 50),
    "4_40": (3, 50)
}

from os.path import join, dirname, realpath
def read_1_train_data(exp_num, num_irr):    
    return load_pickle(join(dirname(realpath(__file__)), f"./data/1data/_read_synth__{exp_num}__{num_irr}__0.pkl"))


def read_1_test_data(exp_num, num_irr):
    return load_pickle(join(dirname(realpath(__file__)), f"./data/1data/_read_synth_test__{exp_num}__{num_irr}.pkl"))[:2]


if __name__ == "__main__":
    '''
    with open("./results/popo.pkl", "rb") as f:
        cv_rslts = pickle.load(f)
    from model_selection import best_param_1se_rule
    best_param_1se_rule(cv_rslts)
    print (cv_rslts)
    exit()
    '''
    a_, b_, c_ = [], [], []
    for exp_num in [4]:#[1, 2, 3, 4]:
        for num_irr in [0]:#[0, 20, 40]:
            #Read in the training data

            train_data               = read_1_train_data(exp_num, num_irr)#read_train_data()

            #Train a BoXHED2.0 hazard estimator instance
            #a, b = hypers[f"{exp_num}_{num_irr}"]
            
            for a in [2]:#[1, 2, 3]:
                for b in [i*10 for i in range(1, 25)]:#[100]:#[i*10 for i in range(1, 25)]:#[100]:#[70+i for i in range(0, 60)]:#[10, 20, 30, 50, 100, 150, 200, 250, 300]:#[10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
                    boxhed_, train_info_dict = cv_train_BoXHED2(train_data, a, b)
                    #print (train_info_dict)

                    # Print the feature importances saved as a dictionary
                    #print ("feature importances:", boxhed_.VarImps)

                    # Load the test set and the values of the true hazard function at the test points:
                    test_X, test_true_haz = read_1_test_data(exp_num, num_irr)#read_test_data()
                    # Test the BoXHED2.0 hazard estimator on out of sample data
                    test_info_dict        = testRMSE_BoXHED2(boxhed_, test_X, test_true_haz)
                    print (a, b, '-->', test_info_dict['rmse_CI'][0:5])
            #print (test_info_dict['rmse_CI'])
            #a = test_info_dict['rmse_CI']
            #a_.append(a)
            #b_.append(hypers[f"{exp_num}_{num_irr}"])
            #a_.append(a[:5])
            #b_.append(a[7:12])
            #c_.append(a[-6:-1])
    pd.DataFrame({"a_":a_, "b_":b_}).to_csv("./results/egghed.csv", index=None)