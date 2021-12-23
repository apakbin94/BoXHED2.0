import os
from ctypes import *
import numpy as np
import pandas as pd
import copy

class preprocessor:

    class c_boundary_info (Structure):
        _fields_ = [("nIDs", c_size_t), 
                    ("out_nrows", c_size_t), 
                    ("in_lbs",    c_void_p),
                    ("out_lbs",   c_void_p)]


    def __init__(self):
        self.prep_libfile = './build/lib_preprocessor.so'
        self.prep_lib = CDLL(self.prep_libfile)

        self.prep_lib.compute_quant.restype    = None
        self.prep_lib.compute_quant.argtypes   = [
                c_void_p, #data_v
                c_size_t, #nrows 
                c_size_t, #ncols
                c_void_p, #is_cat_v
                c_size_t, #t_start_idx
                c_size_t, #t_end_idx
                c_size_t, #id_idx
                c_size_t, #delta_idx
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t, #num_quantiles
                c_bool,   #weighted
                c_int     #nthreads
                ]

        self.prep_lib.get_boundaries.restype   = c_void_p
        self.prep_lib.get_boundaries.argtypes  = [
                c_void_p, #data_v
                c_size_t, #nrows
                c_size_t, #ncols
                c_size_t, #nIDs
                c_size_t, #id_col_idx
                c_size_t, #t_start_idx
                c_size_t, #t_end_idx
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t  #num_quantiles
                ]

        self.prep_lib.preprocess.restype       = None
        self.prep_lib.preprocess.argtypes      = [
                c_void_p, #data_v
                c_size_t, #nrows
                c_size_t, #ncols
                c_void_p, #is_cat_v
                c_void_p, #bndry_info
                c_void_p, #out_data_v
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t, #num_quantiles
                c_size_t, #t_start_idx 
                c_size_t, #t_end_idx
                c_size_t, #delta_idx
                c_size_t, #id_col_idx
                c_int     #nthreads 
                ]

        self.prep_lib.free_boundary_info.restype  = None
        self.prep_lib.free_boundary_info.argtypes = [
                c_void_p #bndry_info
                ]

        self.prep_lib.shift_left.restype  = None
        self.prep_lib.shift_left.argtypes = [
                c_void_p, #data_v 
                c_size_t, #nrows
                c_size_t, #ncols
                c_void_p, #quant_idx_v
                c_void_p, #quant_v
                c_size_t, #num_quantiles
                c_int     #nthreads
                ]
 

    def _set_col_indcs(self):
        self.t_start_idx = self.colnames.index('t_start')
        self.id_idx      = self.colnames.index('ID')
        self.t_end_idx   = self.colnames.index('t_end')
        self.delta_idx   = self.colnames.index('delta') #TODO: either 0 or 1

    def _contig_float(self, arr):
        return np.ascontiguousarray(arr, dtype = np.float64)

    def _contig_size_t(self, arr):
        return np.ascontiguousarray(arr, dtype = np.uintp)

    def _contig_bool(self, arr):
        return np.ascontiguousarray(arr, dtype = np.bool_)

    def __compute_quant(self, data, nrows, ncols, is_cat):
        self.prep_lib.compute_quant(
            c_void_p(data.ctypes.data), 
            c_size_t(nrows), 
            c_size_t(ncols), 
            c_void_p(is_cat.ctypes.data),
            c_size_t(self.t_start_idx), 
            c_size_t(self.t_end_idx), 
            c_size_t(self.id_idx), 
            c_size_t(self.delta_idx), 
            c_void_p(self.quant.ctypes.data), 
            c_void_p(self.quant_size.ctypes.data),
            c_size_t(self.num_quantiles),
            c_bool(self.weighted),
            c_int(self.nthreads))


    def _get_boundaries(self, data, nrows, ncols, nIDs):
        return self.c_boundary_info.from_address(self.prep_lib.get_boundaries(
            c_void_p(data.ctypes.data), 
            c_size_t(nrows), 
            c_size_t(ncols), 
            c_size_t(nIDs), 
            c_size_t(self.id_idx), 
            c_size_t(self.t_start_idx), 
            c_size_t(self.t_end_idx), 
            c_void_p(self.quant.ctypes.data), 
            c_void_p(self.quant_size.ctypes.data),
            c_size_t(self.num_quantiles)
            ))

    def _preprocess(self, data, nrows, ncols, is_cat, bndry_info):
        preprocessed = self._contig_float(np.zeros((bndry_info.out_nrows, ncols))) 

        self.prep_lib.preprocess(
                c_void_p(data.ctypes.data),
                c_size_t(nrows), 
                c_size_t(ncols), 
                c_void_p(is_cat.ctypes.data),
                byref(bndry_info), 
                c_void_p(preprocessed.ctypes.data),
                c_void_p(self.quant.ctypes.data), 
                c_void_p(self.quant_size.ctypes.data),
                c_size_t(self.num_quantiles), 
                c_size_t(self.t_start_idx), 
                c_size_t(self.t_end_idx), 
                c_size_t(self.delta_idx), 
                c_size_t(self.id_idx), 
                c_int(self.nthreads))

        return preprocessed

    def _free_boundary_info(self, bndry_info):
        self.prep_lib.free_boundary_info(byref(bndry_info))
        del bndry_info

    def _prep_output_df(self, preprocessed):
        new_col_names                  = copy.copy(self.colnames)
        new_col_names[self.t_end_idx]  = 'dt'

        preprocessed = pd.DataFrame(preprocessed, columns = new_col_names)
        id           = preprocessed['ID']
        #self.y           = self.preprocessed[['delta', 'dt']]
        w            = preprocessed['dt']
        delta        = preprocessed['delta']
        X            = preprocessed.drop(columns = ['ID', 'delta', 'dt'])

        return id, X, delta, w
    

    def _data_sanity_check(self, data):
        assert data.ndim==2,"ERROR: data needs to be 2 dimensional"
        #assert data['subject'].between(1, nIDs).all(),"ERROR: Patients need to be numbered from 1 to # subjects"

    def _setup_data(self, data):

        #making sure subject data is contiguous
        #data = data.sort_values(by=['ID', 't_start'])
        nIDs = data['ID'].nunique()

        self._data_sanity_check(data)
        data  = self._contig_float(data)

        return data, nIDs


    def _compute_quant(self, data, nrows, ncols, is_cat):
        self.quant      = self._contig_float(np.zeros((1, self.num_quantiles*(ncols))))
        self.quant_size = self._contig_size_t(np.zeros((1, ncols)))

        self.__compute_quant(data, nrows, ncols, is_cat)

    def preprocess(self, data, is_cat=[], num_quantiles=256, weighted=False, nthreads=1):
        #TODO: maye change how the data is given? pat, X, y?

        #XXX: using np.float64---c_double
        self.nthreads           = nthreads
        self.num_quantiles      = min(num_quantiles, 256)
        self.weighted           = weighted
        is_cat                  = self._contig_bool(np.zeros((1, data.shape[1])))
        for cat_col in is_cat:
            is_cat [0, cat_col] = True
        self.is_cat             = is_cat
        nrows                   = data.shape[0]
        ncols                   = data.shape[1]

        self.colnames  = list(data.columns)
        self._set_col_indcs()

        data, nIDs              = self._setup_data(data)

        self._compute_quant(data, nrows, ncols, is_cat)

        bndry_info              = self._get_boundaries(data, nrows, ncols, nIDs)
        preprocessed            = self._preprocess(data, nrows, ncols, is_cat, bndry_info)
        print (preprocessed)
        raise
        IDs, X, delta, w        = self._prep_output_df(preprocessed)
        self._free_boundary_info(bndry_info)

        return IDs, X, w, delta

    def _post_training_get_X_shape(self, X):
        assert X.ndim==2,"ERROR: data needs to be 2 dimensional"
        nrows, ncols = X.shape
        assert ncols == self.ncols-3, "ERROR: ncols in X does not match the trained data"
        return nrows, ncols

    def shift_left(self, X):

        nrows, ncols = self._post_training_get_X_shape(X)

        quant_idxs = np.ascontiguousarray(np.zeros(ncols), dtype = np.int32)

        for idx, colname in enumerate(X.columns):
            col_idx = self.colnames.index(colname)
            assert col_idx > -1, "ERROR: X and trained data colnames do not match"
            quant_idxs [idx] = col_idx


        processed = np.ascontiguousarray(X.values)

        self.prep_lib.shift_left(
            c_void_p(processed.ctypes.data),
            c_size_t(nrows),
            c_size_t(ncols),
            c_void_p(quant_idxs.ctypes.data),
            c_void_p(self.quant.ctypes.data),
            c_size_t(self.num_quantiles),
            c_int(self.nthreads))
        
        return processed

    def epoch_break_cte_hazard (self, data): # used for breaking epochs into cte hazard valued intervals
        nrows, ncols  = data.shape
        data, nIDs    = self._setup_data(data)
        bndry_info    = self._get_boundaries(data, nrows, ncols, nIDs)
        data          = self._preprocess(data, nrows, ncols, self.is_cat, bndry_info)
        
        processed       = pd.DataFrame(data, columns = self.colnames)
        processed.rename(columns={"t_end": "dt"}, inplace=True)

        self._free_boundary_info(bndry_info)
        
        return processed