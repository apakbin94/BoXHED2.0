#ifndef PREPROCESSOR_H
#define PREPROCESSOR_H

#include <string>
#include <fstream>
#include <iostream>
#include <sstream>
#include <algorithm>
#include <vector>
#include <iomanip>

template<class T>
class df;

typedef struct boundary_info_{
    boundary_info_(size_t nsubjects_, size_t out_nrows_, size_t* in_lbs_, size_t* out_lbs_):
        nsubjects(nsubjects_),
        out_nrows(out_nrows_),
        in_lbs(in_lbs_),
        out_lbs(out_lbs_)
        {}

    size_t  nsubjects;
    size_t  out_nrows;
    size_t* in_lbs;
    size_t* out_lbs;

    ~boundary_info_(){
        delete []in_lbs;
        delete []out_lbs;
    }
} boundary_info;


#ifdef __cplusplus
extern "C" 
{
#endif
void free_boundary_info(boundary_info* bndry_info);
void preprocess(
                    const void*          data_v, 
                    const size_t         nrows, 
                    const size_t         ncols, 
                    const void*          is_cat_v,
                    const boundary_info* bndry_info, 
                          void*          out_data_v, 
                    const void*          quant_v, 
                    const void*          quant_size_v,
                    const size_t         quant_per_column, 
                    const size_t         t_start_idx, 
                    const size_t         t_end_idx, 
                    const size_t         delta_idx, 
                    const size_t         pat_col_idx, 
                    const int            nthreads
                    );

boundary_info* get_boundaries(
        const void* data_v, 
        size_t nrows, 
        size_t ncols, 
        size_t nsubjects, 
        size_t pat_col_idx, 
        size_t t_start_idx, 
        size_t t_end_idx, 
        const void* quant_v, 
        const void* quant_size_v, 
        size_t quant_per_column
        );


void compute_quant(
        const void* data_v, 
        size_t nrows, 
        size_t ncols, 
        void* is_cat_v, 
        size_t t_start_idx, 
        size_t t_end_idx, 
        size_t pat_idx, 
        size_t delta_idx, 
        void* quant_v, 
        void* quant_size_v, 
        size_t quant_per_column, 
        bool weighted, 
        int nthreads
        );


void shift_left(
        void* data_v, 
        size_t nrows, 
        size_t ncols, 
        const void* quant_idx_v, 
        const void* quant_v, 
        size_t quant_per_column, 
        int nthreads
        );
#ifdef __cplusplus
}
#endif



#endif /* PREPROCESSOR_H */
