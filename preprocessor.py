import os
from ctypes import *
import numpy as np
import pandas as pd

class preprocessor:

    class c_boundary_info (Structure):
        _fields_ = [("npatients", c_size_t), 
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
                c_size_t, #pat_idx
                c_size_t, #delta_idx
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t, #quant_per_column
                c_bool,   #weighted
                c_int     #nthreads
                ]

        self.prep_lib.get_boundaries.restype   = c_void_p
        self.prep_lib.get_boundaries.argtypes  = [
                c_void_p, #data_v
                c_size_t, #nrows
                c_size_t, #ncols
                c_size_t, #npatients
                c_size_t, #pat_col_idx
                c_size_t, #t_start_idx
                c_size_t, #t_end_idx
                c_void_p, #quant_v
                c_void_p, #quant_size_v
                c_size_t  #quant_per_column
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
                c_size_t, #quant_per_column
                c_size_t, #t_start_idx 
                c_size_t, #t_end_idx
                c_size_t, #delta_idx
                c_size_t, #pat_col_idx
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
                c_size_t, #quant_per_column
                c_int     #nthreads
                ]
 

    def _get_col_indcs(self):
        self.t_start_idx = self.colnames.index('t_start')
        self.pat_idx     = self.colnames.index('patient')
        self.t_end_idx   = self.colnames.index('t_end')
        self.delta_idx   = self.colnames.index('delta') #TODO: either 0 or 1


    def _cnvrt_colnames(self):
        self.colnames[self.t_end_idx] = 'dt'


    def _contig_float(self, arr):
        return np.ascontiguousarray(arr, dtype = np.float64)

    def _contig_size_t(self, arr):
        return np.ascontiguousarray(arr, dtype = np.uintp)

    def _contig_bool(self, arr):
        return np.ascontiguousarray(arr, dtype = np.bool_)

    def __compute_quant(self):
        self.prep_lib.compute_quant(
            c_void_p(self.data.ctypes.data), 
            c_size_t(self.nrows), 
            c_size_t(self.ncols), 
            c_void_p(self.is_cat.ctypes.data),
            c_size_t(self.t_start_idx), 
            c_size_t(self.t_end_idx), 
            c_size_t(self.pat_idx), 
            c_size_t(self.delta_idx), 
            c_void_p(self.quant.ctypes.data), 
            c_void_p(self.quant_size.ctypes.data),
            c_size_t(self.quant_per_column),
            c_bool(self.weighted),
            c_int(self.nthreads))


    def _get_boundaries(self):
        return self.c_boundary_info.from_address(self.prep_lib.get_boundaries(
            c_void_p(self.data.ctypes.data), 
            c_size_t(self.nrows), 
            c_size_t(self.ncols), 
            c_size_t(self.npatients), 
            c_size_t(self.pat_idx), 
            c_size_t(self.t_start_idx), 
            c_size_t(self.t_end_idx), 
            c_void_p(self.quant.ctypes.data), 
            c_void_p(self.quant_size.ctypes.data),
            c_size_t(self.quant_per_column)
            ))

    def _preprocess(self):
        self.preprocessed = self._contig_float(np.zeros((self.bndry_info.out_nrows, self.ncols))) 

        self.prep_lib.preprocess(
                c_void_p(self.data.ctypes.data),
                c_size_t(self.nrows), 
                c_size_t(self.ncols), 
                c_void_p(self.is_cat.ctypes.data),
                byref(self.bndry_info), 
                c_void_p(self.preprocessed.ctypes.data),
                c_void_p(self.quant.ctypes.data), 
                c_void_p(self.quant_size.ctypes.data),
                c_size_t(self.quant_per_column), 
                c_size_t(self.t_start_idx), 
                c_size_t(self.t_end_idx), 
                c_size_t(self.delta_idx), 
                c_size_t(self.pat_idx), 
                c_int(self.nthreads))

    def _free_boundary_info(self):
        self.prep_lib.free_boundary_info(byref(self.bndry_info))
        del self.bndry_info

    def _prep_output_df(self):
        self._cnvrt_colnames(); 

        self.preprocessed = pd.DataFrame(self.preprocessed, columns = self.colnames)
        self.subjects     = self.preprocessed['patient']
        #self.y           = self.preprocessed[['delta', 'dt']]
        self.w            = self.preprocessed['dt']
        self.delta        = self.preprocessed['delta']
        self.X            = self.preprocessed.drop(columns = ['patient', 'delta', 'dt'])
        

    def _set_lbs_ubs_ptrs(self):
        self.in_lbs   = (c_size_t * (self.bndry_info.npatients+1)).from_address(self.bndry_info.in_lbs)
        self.out_lbs  = (c_size_t * (self.bndry_info.npatients+1)).from_address(self.bndry_info.out_lbs)
    

    def _data_sanity_check(self):
        assert self.data.ndim==2,"ERROR: data needs to be 2 dimensional"
        assert self.data['patient'].between(1, self.npatients).all(),"ERROR: Patients need to be numbered from 1 to # patients"

    def _setup_data(self):

        #making sure patient data is contiguous
        self.data.sort_values(by=['patient', 't_start'], inplace = True)

        self.colnames  = list(self.data.columns)
        self.npatients = self.data['patient'].nunique()

        self._data_sanity_check()
        self._get_col_indcs()

        self.data  = self._contig_float(self.data)


    def _compute_quant(self):
        self.tpart      = self._contig_float(np.zeros((1, self.quant_per_column)))
        self.quant      = self._contig_float(np.zeros((1, self.quant_per_column*(self.ncols))))
        self.quant_size = self._contig_size_t(np.zeros((1, self.ncols)))

        self.__compute_quant()

    def preprocess(self, data, is_cat=[], quant_per_column=20, weighted=False, nthreads=-1):
        #TODO: maye change how the data is given? pat, X, y?

        #XXX: using np.float64---c_double
        self.nthreads       = nthreads
        self.quant_per_column   = min(quant_per_column, 256)
        self.weighted       = weighted
        self.data           = data
        self.is_cat         = self._contig_bool(np.zeros((1, data.shape[1])))
        for cat_col in is_cat:
            self.is_cat [0, cat_col] = True
        self.nrows          = data.shape[0]
        self.ncols          = data.shape[1]

        self._setup_data()

        self._compute_quant()

        self.bndry_info = self._get_boundaries()
        self._set_lbs_ubs_ptrs()
        self._preprocess()
        self._prep_output_df()
        self._free_boundary_info()

        return self.subjects, self.X, self.w, self.delta

    def shift_left(self, X, nthreads=-1):

        assert X.ndim==2,"ERROR: data needs to be 2 dimensional"
        nrows, ncols = X.shape
        assert ncols == self.ncols-3, "ERROR: ncols in X does not match the trained data"

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
            c_size_t(self.quant_per_column),
            c_int(nthreads))
        
        return processed
