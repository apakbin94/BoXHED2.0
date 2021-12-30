import os
import pandas as pd
from boxhed import boxhed
from model_selection import cv
from utils import timer, run_as_process, TrueHaz, calc_L2, create_dir_if_not_exist

# BoXHED 2.0 (https://arxiv.org/pdf/2103.12591.pdf) is a software package
# for estimating hazard functions nonparametrically via gradient boosting. 
# It dramatically improves upon BoXHED 1.0 (http://proceedings.mlr.press/v119/wang20o/wang20o.pdf) in speed.
# BoXHED 2.0 also allows for more general forms of survival data including recurrent events.

#This tutorial demonstrates how to apply BoXHED 2.0 to a synthetic dataset.

DATA_ADDRESS = "./data/"     # train/test data directory
RSLT_ADDRESS = "./results/"  # results directory

nthread_prep    = 20    # number of threads used for preprocessing
nthread_train   = 20    # number of threads used for training

# creating the results' directory if it does not exist
for addr in [RSLT_ADDRESS]:
    create_dir_if_not_exist(addr)

# Function: read_train_data
# reads the synthetic training data.
# Input:
#      None
# Return: 
#      @ A pandas dataframe containing training data with the following columns:
#            * ID:      subject ID
#            * t_start: the start time of an epoch for the subject
#            * t_end:   the end time of the epoch
#            * X_i:     values of covariates between t_start and t_end
# Sample Output:
# subject	t_start         t_end           X_0             delta
#	1       0.010000	0.064333	0.152407	0.0
#	1       0.064333	0.135136	0.308475	0.0
#	1       0.194810	0.223106	0.614977	1.0
#	1       0.223106	0.248753	0.614977	0.0
#	2       0.795027	0.841729	0.196407	1.0
#	2       0.841729	0.886587	0.196407	0.0
#	2       0.886587	0.949803	0.671227	0.0
#
# As can be seen above, t_start<T_end for each epoch (row). Also, the beginning of one epoch starts no earlier than
# the end of the previous one, i.e. t_end_i <= t_start_i+1 . Delta denotes whether an event (possibly recurrent) 
# occurred at the end of the epoch. For covariates with missing values, BoXHED 2.0 implements tree splits of the form:
# Left daughter node: {x<=split.point or x is missing}; Right daughter node: {x>split.point}
# or
# Left daughter node: {x<=split.point}; Right daughter node: {x>split.point or x is missing}.
# Alternatively, the user may choose to impute the missing values, for example by carrying forward the most recent observed value.
def read_train_data():
    return pd.read_csv(os.path.join(DATA_ADDRESS, 'training.csv'))


# Function: read_test_data
# reads the synthetic testing data. The values of the true hazard function are also provided for accuracy comparisons.
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


# Function: cv_train_test_BoXHED2
# cross-validate, train, and test a BoXHED2.0 instance
# Input:
#      None
# Return: 
#      @ A dictionary containing RMSE (with 95% CIs) and timings of different components.
@run_as_process
def cv_train_test_BoXHED2():
    # defining the output dictionary
    out_dict = {}

    #reading in the data
    data = read_train_data()

    # Creating an instance of BoXHED to preprocess the training data.
    boxhed_ = boxhed()

    # preprocessing. We are also timing it.
    prep_timer = timer()                          # initializing a timer.
    # boxhed.preprocess():
    # Input:
    #      @ num_quantiles: the number of candidate split points to try for time and for each covariate. 
    #                       The locations of the split points are based on the quantiles of the training data.
    #      @ is_cat:        a list of the column indexes that contain categorical data. The categorical data must be one-hot encoded.
    #                       For example, if a categorical variable with 3 factors is one hot encoded into binary-valued columns 4,5,6, 
    #                       then is_cat = [4,5,6]
    #      @ weighted:      if set to True, the locations of the candidate split points will be based on weighted quantiles 
    #                       (see Section 3.3 of the BoXHED 2.0 paper)      
    #      @ nthreads:      number of CPU threads to use for preprocessing the data
    # Return: 
    #      @ ID:            subject ID for each row in the data frames X, w, and delta
    #      @ X:             each row represents an epoch of the transformed data, and contains the values of the covariates as well as
    #                       its start time
    #      @ w:             length of each epoch     
    #      @ delta:         equals one if an event occurred at the end of the epoch; zero otherwise
    num_quantiles   = 256
    ID, X, w, delta = boxhed_.preprocess(
            data          = data, 
            #is_cat       = [],
            num_quantiles = num_quantiles, 
            weighted      = False, 
            nthread       = nthread_prep)
    out_dict["prep_time"] = prep_timer.get_dur()  # calling the get_dur() function.
    
    # perform K-fold cross-validation to choose the hyperparameters {number of boosted trees, tree depth, learning rate} if do_CV is true.
    # first, specify the candidate values for the hyperparameters to cross-validate on (more trees and/or deeper trees may be needed for other datasets).
    param_grid = {'max_depth':    [1, 2, 3, 4, 5],
              'n_estimators': [50, 100, 150, 200, 250, 300]}
    # if do_CV is false, use the following default values:
    hyperparams = {'max_depth':1, 'n_estimators':200}
    do_CV           = False # if True, cross validation is performed for best hyperparameters. Default values are used otherwise.
    # next, specify:
    #      @ gpu_list:    the list of GPU IDs to use for training (we set gpu_list to [-1] to use CPU in this tutorial).
    #      @ batch_size : training batch size, which is the maximum number of BoXHED2.0 instances trained at any point in time. 
    #                     If we perform 10-fold cross-validation using the above param_grid, we would need to train 5 * 6 * 10 = 300
    #                     instances in total
    #                           * When using GPUs, each GPU will train at most batch_size/len(gpu_list) instances at a time
    #                           * When gpu_list = [-1], batch_size is the number of CPU threads used, 
    #                             with each one training one instance at a time
    gpu_list   = [-1]
    batch_size = 6
    num_folds  = 5
    if do_CV:
        cv_timer = timer()
        # call the cv function to perform K-fold cross validation on the training set. 
        # This outputs the cross validation results for the different hyperparameter combinations.
        # Return: 
        #      @ cv_rslts:    mean and st.dev of the log-likelihood value for each hyperparameter combination
        #      @ best_params: The hyper-parameter combination where the mean log-likelihood value is maximized.
        #                     HOWEVER, we recommend using the one-standard-error rule to select the most parsimonious 
        #                     combination that is within st.dev/sqrt(k) of the maximum log-likelihood value (§7.10 of Hastie et al. (2009))
        cv_rslts, best_params = cv(param_grid, 
                                  X, 
                                  w,
                                  delta,
                                  ID, 
                                  num_folds,
                                  gpu_list,
                                  batch_size)
    
        out_dict["CV_time"] = cv_timer.get_dur()
    else:
        best_params = hyperparams
    best_params['gpu_id'] = gpu_list[0] # holding on to one GPU id to use for training
    best_params['nthread'] = nthread_train 

    out_dict.update(best_params)
    boxhed_.set_params (**best_params)

    fit_timer = timer()
    # Fit BoXHED to the training data:
    boxhed_.fit (X, delta, w)
    out_dict["fit_time"] = fit_timer.get_dur()

    # Load the test set and the values of the true hazard function at the test points:
    test_X, test_true_haz = read_test_data()

    pred_timer = timer()
    # Use the fitted model to estimate the value of the hazard function for each row of the test set:
    preds = boxhed_.predict(test_X)
    out_dict["pred_time"] = pred_timer.get_dur()

    # Compute the RMSE of the estimates, and its 95% confidence interval:
    L2 = calc_L2(preds, test_true_haz)
    out_dict["rmse_CI"] = f"{L2['point_estimate']:.3f} ({L2['lower_CI']:.3f}, {L2['higher_CI']:.3f})"

    return out_dict


if __name__ == "__main__":
    rslt = cv_train_test_BoXHED2()
    print (rslt)
