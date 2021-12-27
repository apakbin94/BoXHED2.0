import os
import pandas as pd
from boxhed import boxhed
from model_selection import cv
from utils import timer, curr_dat_time, run_as_process, TrueHaz, calc_L2, create_dir_if_not_exist


DATA_ADDRESS = "./data/"
RSLT_ADDRESS = "./results/"

num_quantiles   = 256   #number of quantiles used for each covariate
do_CV           = False
gpu_list        = [1]
nthread         = 20

param_grid = {'max_depth':    [1, 2, 3, 4, 5],
           'n_estimators': [50, 100, 150, 200, 250, 300]}

hyperparams = { 'max_depth':1, 'n_estimators':200}


for addr in [RSLT_ADDRESS]:
    create_dir_if_not_exist(addr)


def _read_train_data():
    return pd.read_csv(os.path.join(DATA_ADDRESS, 'training.csv'))


def _read_test_data():
    test_data = pd.read_csv(os.path.join(DATA_ADDRESS, 'testing.csv'))
    return test_data, TrueHaz(test_data[['t', 'X_0']].values)


@run_as_process
def cv_train_test_BoXHED2():
    out_dict = {'num_quantiles': num_quantiles}

    #reading in the data
    data = _read_train_data()

    #defining a BoXHED2.0 instance
    boxhed_ = boxhed()

    # preprocessing. We are also timing it.
    prep_timer = timer()
    subjects, X, w, delta = boxhed_.preprocess(
            data          = data, 
            #is_cat       = [4],
            num_quantiles = num_quantiles, 
            weighted      = False, 
            nthread       = nthread)
    out_dict["prep_time"] = prep_timer.get_dur()
    
    # performing cross validation if do_CV is true, otherwise, using predefined values
    if do_CV:
        gridsearch_timer = timer()
        cv_rslts, best_params = cv(param_grid, 
                                  X, 
                                  w,
                                  delta,
                                  subjects, 
                                  5,
                                  gpu_list,
                                  nthread)
    
        out_dict["CV_time"] = gridsearch_timer.get_dur()
    else:
        best_params = hyperparams
    best_params['gpu_id'] = gpu_list[0] # holding on to one GPU id to use for training
    best_params['nthread'] = nthread    # setting nthreads for

    out_dict.update(best_params)
     
    boxhed_.set_params (**best_params)

    fit_timer = timer()
    boxhed_.fit (X, delta, w)
    out_dict["fit_time"] = fit_timer.get_dur()

    test_X, test_true_haz = _read_test_data()

    pred_timer = timer()
    preds = boxhed_.predict(test_X)
    out_dict["pred_time"] = pred_timer.get_dur()

    L2 = calc_L2(preds, test_true_haz)
    out_dict["rmse_CI"] = f"{L2['point_estimate']:.3f} ({L2['lower_CI']:.3f}, {L2['higher_CI']:.3f})"

    return out_dict


if __name__ == "__main__":
    rslt = cv_train_test_BoXHED2()
    print (rslt)
