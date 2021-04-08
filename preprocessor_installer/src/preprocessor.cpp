#include "preprocessor.h"
#include <stdexcept>
#include <unistd.h>
#include <stdlib.h>
#include <numeric> 
#include <cstdlib>
#include <cmath> 

#include <omp.h>
#include <string>
#include <exception>
#include <sstream>

#define EPSILON 1e-4

#define PARALLEL

std::stringstream err;

template <class T>
using vec_iter = typename std::vector<T>::const_iterator;


template <class T>
inline bool _approx_equal (T val1, T val2){
    return std::abs(val1 - val2) < EPSILON;
}

template <class T>
inline size_t _tquant_distance(const std::vector<T>& tquant, int tquant_i, int tquant_j, T t_min,  T t_max){

    return std::max(tquant_j-tquant_i
            -static_cast<int>(_approx_equal(tquant[std::min(tquant_i, static_cast<int>(tquant.size()-1))],t_min))
            -static_cast<int>(_approx_equal(tquant[std::min(tquant_j, static_cast<int>(tquant.size()-1))],t_max)),
            0);
}


template <class T>
inline std::pair<int,int> _get_range_sorted(const std::vector<T> &vec, const T min, const T max){//, const int start_idx=-1, const int end_idx=-1){
    
    //TODO: make it work with start_idx and end_idx
    auto start_iter = vec.begin();
    auto end_iter   = vec.end();
    
    vec_iter<T> min_idx = std::lower_bound (start_iter, end_iter, min);
    vec_iter<T> max_idx = std::lower_bound (min_idx, end_iter, max);

    return std::make_pair(min_idx-vec.begin(), max_idx-vec.begin());
}


template <class T>
class preprocessor{

    public :

        preprocessor(const T* data_, const size_t nrows_, const size_t ncols_, const bool* is_cat_, const boundary_info* bndry_info_, T* out_data_, const T* quant_arr, const size_t* quant_size_arr, const size_t quant_per_column_, const size_t t_start_idx_, const size_t t_end_idx_, const size_t delta_idx_, const size_t pat_col_idx_):
            data(data_),
            nrows(nrows_),
            ncols(ncols_),
            is_cat(is_cat_),
            bndry_info(bndry_info_),
            npatients(bndry_info->npatients),
            out_data(out_data_),
            out_nrows(bndry_info->out_nrows),
            tquant(std::vector<T>(
                        quant_arr+t_start_idx_*quant_per_column_, 
                        quant_arr+t_start_idx_*quant_per_column_
                        +quant_size_arr[t_start_idx_])),
            quant(quant_arr),
            quant_size(quant_size_arr),
            quant_per_column(quant_per_column_),
            t_start_idx(t_start_idx_),
            t_end_idx(t_end_idx_),
            dt_idx(t_end_idx),
            delta_idx(delta_idx_),
            pat_col_idx(pat_col_idx_)
            {}


        ~preprocessor(){
            tquant.clear();
            delete[] temp_quantized_data;
        }


        inline void quantize_column(size_t col_idx){
            auto column_quant = std::vector<T>(
                        quant + col_idx* quant_per_column, 
                        quant + col_idx* quant_per_column
                              + quant_size[col_idx]);

            for (size_t row_idx = 0; row_idx < nrows; ++row_idx){

                T val = data[row_idx*ncols + col_idx];
                T quantized_val;

                if (is_cat[col_idx]         ||
                    std::isnan(val)         ||
                    col_idx == t_start_idx  ||
                    col_idx == t_end_idx    || 
                    col_idx == pat_col_idx  || 
                    col_idx == delta_idx
                    ){

                    quantized_val = val;
                }
                else{
                    auto quant_val_iter = std::lower_bound (
                            column_quant.begin(), 
                            column_quant.end(), 
                            val);

                    quant_val_iter = max(
                            --quant_val_iter, 
                            column_quant.begin());

                    quantized_val = *quant_val_iter;
                }

                temp_quantized_data[row_idx * ncols + col_idx] = quantized_val;
            }
            
        }

        void quantize_non_time_columns(){

            temp_quantized_data  = new T [nrows*ncols]; 

            //TODO: this can be further broken down into parallel parts by having blocks of the same column
            #pragma omp parallel for schedule(static)
            for (size_t col_idx=0; col_idx < ncols; ++col_idx){
                quantize_column(col_idx);
            }
            
        }

        void preprocess(){

            quantize_non_time_columns();

            #pragma omp parallel for schedule(static)
            for (size_t pat_idx=0; pat_idx<npatients; ++pat_idx){
            
                try {
                    _preprocess_one_patient(pat_idx);
                } catch (std::invalid_argument& e){
                    err<<e.what()<<std::endl;
                    throw;
                }
            }  
        }

    private:
        /*
        maybe for posterity if we want to not include patient number
        */
        /*
        inline int _cnvrt_out_col(int col_idx){
            if (col_idx == pat_col_idx)
                throw std::invalid_argument("ERROR: col_indx cannot be equal to pat_col_idx");
            return (col_idx<pat_col_idx)?col_idx:col_idx-1;
        }
        */

        inline void _preprocess_one_patient (size_t pat_idx){
            
            //const pat_lb_ub* _pat_lb_ub  = &pat_lb_ubs[pat_idx];
            const size_t in_lb  = bndry_info->in_lbs[pat_idx];
            const size_t in_ub  = bndry_info->in_lbs[pat_idx+1]-1;
            const size_t out_lb = bndry_info->out_lbs[pat_idx];
            const size_t out_ub = bndry_info->out_lbs[pat_idx+1]-1;

            size_t out_row = out_lb;
            for (size_t row = in_lb; row<=in_ub; ++row){

                //T t_start = data[row*ncols+t_start_idx];
                //T t_end   = data[row*ncols+t_end_idx];
                T t_start = temp_quantized_data[row*ncols+t_start_idx];
                T t_end   = temp_quantized_data[row*ncols+t_end_idx];


                if (t_end<=t_start)
                {
                    std::stringstream err_str;
                    err_str << "ERROR: t_end should be > t_start in input row"<<" "<<row;
                    throw std::invalid_argument(err_str.str());
                }

                //TODO: lower/upper bounds can be optimized
                auto tquant_i_j = _get_range_sorted<T>(tquant, t_start, t_end);
                int tquant_i = tquant_i_j.first;
                int tquant_j = tquant_i_j.second;

                size_t out_len = 1 + _tquant_distance<T>(tquant, tquant_i, tquant_j, t_start, t_end);
                size_t curr_row_ub  = out_row + out_len;
                
                auto tquant_iter  = std::next(tquant.begin(), tquant_i);
                //XXX: -- can get out of bounds. does not happen now probably because minimum is in tquant so it does not go past it. And when we ++ a few lines down, it works now because we have precomputed them
                if (t_start != tquant[tquant_i])
                    --tquant_iter;

                bool first_row_to_fill = true;
                for (;out_row < curr_row_ub; ++out_row){
                    for (size_t col=0; col<ncols; ++col){
                        //if (col==pat_col_idx || col==t_start_idx || col==t_end_idx)
                        if (col==t_start_idx || col==t_end_idx)
                            continue;
                        //out_data[out_row*ncols + col] = data[row*ncols+col];
                        out_data[out_row*ncols + col] = temp_quantized_data[row*ncols+col];

                    }

                    out_data[out_row*ncols + t_start_idx] = *tquant_iter;


                    if (out_len<=1){
                        out_data[out_row*ncols + dt_idx] = t_end - t_start;
                        continue;
                    }
                   
                    // DT
                    int rows_to_fill = curr_row_ub-out_row;
                    if (rows_to_fill > 1){
                        T dt_ = 0.0;
                        if (first_row_to_fill){
                            //dt_ = *(std::next(tquant_iter)) - data[row*ncols + t_start_idx];
                            dt_ = *(std::next(tquant_iter)) - temp_quantized_data[row*ncols + t_start_idx];

                            first_row_to_fill = false;
                        } else {
                            dt_ = *(std::next(tquant_iter)) - *tquant_iter;
                        }
                        out_data [out_row*ncols + dt_idx]    = dt_;
                        out_data [out_row*ncols + delta_idx] = static_cast<T>(0);
                    }else{
                        out_data [out_row*ncols + dt_idx]    = t_end - *tquant_iter;
                    }
                    ++tquant_iter;
                }
            }
            if (out_row-1 != out_ub){
                std::stringstream err_str;
                err_str << "ERROR: loop reached its end for patient index"<<" "<<pat_idx<<"."<<" Check the corresponding patient data.";
                throw std::invalid_argument(err_str.str());
            }
        }

        const T* data;
        const size_t nrows;
        const size_t ncols;
        const bool* is_cat;
        const boundary_info* bndry_info;

        const size_t npatients;

        T* out_data;
        const size_t out_nrows;
        std::vector<T> tquant;
        const T* quant;
        const size_t* quant_size;
        const size_t quant_per_column;
        T* temp_quantized_data;

        const size_t t_start_idx;
        const size_t t_end_idx;
        const size_t dt_idx;
        const size_t delta_idx;
        const size_t pat_col_idx;

};



template <class T>
class pat_lb_ub_calculator{

    public:

        pat_lb_ub_calculator(const T* data_, const size_t nrows_, const size_t ncols_, const size_t npatients_, const T* quant_arr, const size_t* quant_size, const size_t quant_per_column, const size_t t_start_idx_, const size_t pat_col_idx_, size_t t_end_idx_):
            data(data_),
            nrows(nrows_),
            ncols(ncols_),
            npatients(npatients_),
            tquant(std::vector<T>(
                        quant_arr+t_start_idx_*quant_per_column, 
                        quant_arr+t_start_idx_*quant_per_column
                        + quant_size[t_start_idx_])),
            t_start_idx(t_start_idx_),
            pat_col_idx(pat_col_idx_),
            t_end_idx(t_end_idx_),
            in_lbs(new size_t[npatients+1]),
            out_lbs(new size_t[npatients+1])
            {}


        ~pat_lb_ub_calculator(){
            tquant.clear();
        }

        boundary_info* get_boundaries(){
            _get_boundaries();
            boundary_info* bndry_info = new boundary_info(npatients, out_nrows, in_lbs, out_lbs);
            return bndry_info;
        }

    private:

         void _get_boundaries(){
            //XXX: assuming data of patients in chronological order, and contiguous
            //XXX: now assuming patient ids from one to N

            int last_patient=1, curr_patient = 1;
            size_t in_lb  = 0;
            size_t out_lb = 0;

            for (size_t row=0; row<nrows; ++row){
                curr_patient = data[row*ncols+pat_col_idx];

                if (curr_patient == last_patient)
                    continue;

                size_t out_len = _out_len(in_lb, row-1);

                //_pat_lb_ubs[last_patient-1].set(last_patient, in_lb, row-1, out_lb, out_lb+out_len-1);
                in_lbs[last_patient-1]=in_lb;
                out_lbs[last_patient-1]=out_lb;

                in_lb = row;
                out_lb += out_len;
                last_patient = curr_patient;
            }

            size_t out_len = _out_len(in_lb, nrows-1);
            size_t last_ub = out_lb + out_len;

            //_pat_lb_ubs[last_patient-1].set(last_patient, in_lb, nrows-1, out_lb, last_ub);
            in_lbs[last_patient-1]=in_lb;
            out_lbs [last_patient-1]=out_lb;

            in_lbs[last_patient]=nrows;
            out_lbs[last_patient]=last_ub;

            out_nrows  = last_ub;
            //pat_lb_ubs = _pat_lb_ubs;

        }  

        inline size_t _out_len(size_t lb, size_t ub){
            size_t extra_len = 0;
            //TODO: this loop can still be optimized. tpart_i can be better approximated by the tpart_j of the previous iteration, assuming the intervals to be non-overlapping
            for (size_t in_row = lb; in_row <= ub; ++in_row){
                T t_start = data[in_row*ncols+t_start_idx];
                T t_end   = data[in_row*ncols+t_end_idx];

                auto tquant_i_j = _get_range_sorted<T>(tquant, t_start, t_end);
                int tquant_i = tquant_i_j.first;
                int tquant_j = tquant_i_j.second;

                extra_len += _tquant_distance<T>(tquant, tquant_i, tquant_j, t_start, t_end);
            }
            
            return (ub-lb+1) + extra_len;
        }


        const T* data;
        const size_t nrows;
        const size_t ncols;
        const size_t npatients;
        std::vector<T> tquant;

        size_t t_start_idx;
        size_t pat_col_idx;
        size_t t_end_idx;   

        size_t out_nrows;

        size_t* in_lbs;
        size_t* out_lbs;

};



template <class T>
inline void _rmv_dupl_srtd(T* arr, const size_t arr_size, size_t * out_size){

    size_t idx = 0;
    T last_val = arr[0];
    for (size_t i=1; i<arr_size; ++i){
        if (arr[i]!=last_val){
            arr[++idx] = arr[i];
            last_val   = arr[i];
        }
    }
    /*
    *out_size = idx+1; 
    */
    *out_size =  std::isnan(arr[idx]) ? idx : idx+1;
}


template <class T>
inline void _rmv_dupl_srtd(const std::vector<std::pair<T, size_t>> &vals, T* out, size_t * out_size){

    size_t idx = 0;
    T last_val = vals[0].first;
    out [0] = last_val; 
 
    for (size_t i=1; i<vals.size(); ++i){
        if (vals[i].first!=last_val){
            out[++idx] = vals[i].first;
            last_val   = vals[i].first;
        }
    }
    /*
    *out_size = idx+1; 
    */
    *out_size =  std::isnan(out[idx]) ? idx : idx+1;
}


template <class T>
inline void _copy_col2arr(const T* src, size_t nrows, size_t ncols,
                      size_t col_idx, T* dst){
    for (size_t row_idx = 0; row_idx < nrows; ++row_idx){
        dst[row_idx] = src[row_idx*ncols+col_idx];
    }
}


template <class T>
inline void _compute_quant(const T* data, size_t nrows, size_t ncols, const bool* is_cat, size_t t_start_idx, size_t t_end_idx, size_t pat_idx, size_t delta_idx, T* quant, size_t* quant_size, size_t quant_per_column){
 
    //#pragma omp parallel for schedule(dynamic)
    for (size_t col_idx = 0; col_idx<ncols; ++col_idx){
        std::cout << col_idx << std::endl;
        if (is_cat[col_idx] || col_idx == t_end_idx || col_idx == pat_idx || col_idx == delta_idx){
            continue;
        }
        size_t vals_size = (col_idx==t_start_idx) ? 2*nrows : nrows;

        /*
        T vals [vals_size];
        */
        T *vals = new T[vals_size];

        _copy_col2arr(data, nrows, ncols, col_idx, vals);
        if (col_idx == t_start_idx){
            _copy_col2arr(data, nrows, ncols, t_end_idx, vals + nrows);
        }

        std::stable_sort(vals, vals+vals_size,
               [](const T a, 
                  const T b)
                 {return std::isnan(b) || a < b;}
                );
        
        size_t num_unique;
        _rmv_dupl_srtd<T> (vals, vals_size, &num_unique);

        size_t num_quants = std::min(num_unique, quant_per_column);
        quant_size [col_idx] = num_quants;

        for (size_t i=0; i<num_quants; ++i){
            quant[col_idx*quant_per_column+i] = vals[static_cast<int>(num_unique*i/num_quants)];
        }
        delete [] vals;
                
    }

}

template <class T>
inline void _fill_time_hist(const T* unique_arr, const size_t unique_arr_size, 
                            const T* data, size_t nrows, size_t ncols,
                            size_t t_start_idx, size_t t_end_idx, //size_t pat_idx,
                            size_t* hist){
    //TODO: maybe pat_idx for searching optimization??

    std::vector<T> unique_arr_vec (unique_arr, unique_arr + unique_arr_size);

    #pragma omp parallel for schedule(dynamic) 
    for (size_t i=0; i < nrows; ++i){
        const auto t_start = data [i*ncols + t_start_idx];
        const auto t_end   = data [i*ncols + t_end_idx];
        //TODO: should I make sure they are not the same?
        auto iter_from     = std::lower_bound(unique_arr_vec.begin(), unique_arr_vec.end(), t_start);
        size_t idx_from    = static_cast<size_t>(iter_from - unique_arr_vec.begin());
 
        for (size_t i = idx_from; ; ++i){
            if (_approx_equal(unique_arr_vec[i],t_end)){
                break;
            }
            
            #pragma omp atomic
            hist [i] += 1;
        }
    }
}


template <class T>
inline void _fill_non_time_acc_weight(const std::vector<std::pair<T, size_t>> &srtd_val_idx,
                                       const T* data, size_t nrows, size_t ncols,
                                       size_t t_start_idx, size_t t_end_idx,// size_t col_idx,
                                       T* acc_weight){

    const size_t first_data_idx = srtd_val_idx[0].second;
    const T      first_val      = srtd_val_idx[0].first;
    const T      first_dt       = data [first_data_idx*ncols + t_end_idx] - data [first_data_idx*ncols + t_start_idx];

    acc_weight [0] = 0;
    acc_weight [1] = first_dt;
    size_t val_idx = 1;
    T last_val = first_val;

    for (size_t i = 1; i < nrows; ++i){
        const size_t data_idx = srtd_val_idx[i].second;
        const T      val      = srtd_val_idx[i].first;
        const T      dt       = data [data_idx*ncols + t_end_idx] - data [data_idx*ncols + t_start_idx];
        
        if (val != last_val){
            ++val_idx;
            last_val = val;
            acc_weight[val_idx] = acc_weight[val_idx-1] + dt;
            continue;
        }
        acc_weight [val_idx] += dt;
    }
}


template <class T>
void _fill_quants (T* quant, size_t quant_per_column, size_t num_quants, size_t col_idx,
                          T* unique, T* acc_weight, size_t num_unique){

    std::cout << "-1" << std::endl;
    std::cout << col_idx << std::endl;
    if (col_idx == 1){
        std::cout << "returning because it's time" << std::endl;
        return;
    }
    
    /*
    std::vector<T> acc_weight_vec (acc_weight, acc_weight + num_unique);
    */
    
    std::vector<T> *acc_weight_vec = new std::vector<T>;
    
    /*
    std::cout << "max val: " << acc_weight_vec->max_size() << ", actual: " << num_unique << std::endl;

    std::cout << "0" << std::endl;
    */
    acc_weight_vec -> resize(500);
    std::cout << "resized ! "<< std::endl;
    /*
    acc_weight_vec -> resize(num_unique);
    std::cout << "-0" << std::endl;

    for (size_t i = 0; i<num_unique; ++i){
        (*acc_weight_vec)[i] = acc_weight[i];
    }
    std::cout << "-2" << std::endl;
    */

    if (num_unique <= quant_per_column){
        for (size_t i = 0; i < num_unique; ++i){
            quant[col_idx*quant_per_column + i] = unique [i];
        }
        return;
    }

    std::cout << "-3" << std::endl;

    //TODO: loop can be optimized by providing lower bound
    for (size_t i = 0; i<num_quants; ++i){
        const T quant_to_select = static_cast<T>(i)/num_quants;
        /*
        auto iter  = std::lower_bound(acc_weight_vec.begin(), acc_weight_vec.end(), quant_to_select);
        size_t idx = static_cast<size_t>(max(--iter, acc_weight_vec.begin()) - acc_weight_vec.begin());
        */
        
        auto iter  = std::lower_bound(acc_weight_vec -> begin(), acc_weight_vec -> end(), quant_to_select);
        size_t idx = static_cast<size_t>(max(--iter, acc_weight_vec->begin()) - acc_weight_vec->begin());
        

        
        T val   = unique[idx];
        T val_n = (idx < num_unique-1) ? unique[idx+1] : 
                                         unique[idx]+1;
        /* 
        T w     = acc_weight_vec[idx];
        T w_n   = (idx < num_unique-1) ? acc_weight_vec[idx+1] : 
                                         acc_weight_vec[idx]+1; 
        */
        
        T w     = (*acc_weight_vec)[idx];
        T w_n   = (idx < num_unique-1) ? (*acc_weight_vec)[idx+1] : 
                                         (*acc_weight_vec)[idx]+1;
        

        T q;

        if (_approx_equal(w, quant_to_select)){
            q = val;
        } else if (_approx_equal(w_n, quant_to_select)){
            q = val_n;
        } else {
            q = ((val_n-val)/(w_n-w))*(quant_to_select-w)+val;
        }

        quant[col_idx*quant_per_column + i] = q;

        /*
        if ((idx < acc_weight_vec.size()-1) 
         && (std::abs(acc_weight_vec[idx]-quant_to_select) > std::abs(acc_weight_vec[idx+1]-quant_to_select))){
            idx += 1;
         }

        quant[col_idx*quant_per_column + i] = unique[idx];
        */
        if ((i>0) && (quant[col_idx*quant_per_column + i] == quant[col_idx*quant_per_column + i - 1]))
            {
            /*
            throw std::invalid_argument("ERROR: An error has occured. Consider decreasing quant_per_column.");        
            */
            std::stringstream err_str;
            err_str << "ERROR: An error has occured in column"<<" "<<col_idx<<" while extracting quantiles.";
            throw std::invalid_argument(err_str.str());
            }
    }
    acc_weight_vec -> clear();
    //delete acc_weight_vec;
    //acc_weight_vec.clear();
}


//TODO: many of these loops can be optimized using OMP
template <class T>
inline T _compute_total_t(const T* data, size_t nrows, size_t ncols, size_t t_start_idx, size_t t_end_idx){
    
    T total_t = 0;
    for (size_t i =0; i<nrows; ++i){
        total_t += data[i*ncols + t_end_idx] - data[i*ncols + t_start_idx];
    }
    return total_t;
}


template <class T>
inline void normalize (T* arr, size_t size, T norm_factor){
    for (size_t i = 0; i<size; ++i)
        arr[i] = arr[i]/norm_factor;
}

template <class T>
void _compute_quant_weighted(const T* data, size_t nrows, size_t ncols, const bool* is_cat, size_t t_start_idx, size_t t_end_idx, size_t pat_idx, size_t delta_idx, T* quant, size_t* quant_size, size_t quant_per_column){

    throw std::invalid_argument("NOT IMPLEMENTED: Use the non-weighted version for now please.");


    T total_t = _compute_total_t(data, nrows, ncols, t_start_idx, t_end_idx);
    
    #pragma omp parallel for schedule(dynamic)
    for (size_t col_idx = 0; col_idx<ncols; ++col_idx){
        std::cout<<"col:"<<col_idx<<std::endl;
        if (is_cat[col_idx] || col_idx == t_end_idx || col_idx == pat_idx || col_idx == delta_idx){
            continue;
        }
        size_t vals_size = (col_idx==t_start_idx) ? 2*nrows : nrows;
        /*
        T vals [vals_size];
        */
        T *vals = new T[vals_size];

        std::vector<std::pair<T, size_t>> srtd_val_idx (vals_size);

        _copy_col2arr(data, nrows, ncols, col_idx, vals);
        if (col_idx == t_start_idx){
            _copy_col2arr(data, nrows, ncols, t_end_idx, vals + nrows);
        }

        for (size_t i=0; i<vals_size; ++i){
            srtd_val_idx [i] = std::make_pair(vals[i], i);
        }

        delete [] vals;

        std::sort(srtd_val_idx.begin(), srtd_val_idx.end(), 
               [](const std::pair<T, size_t> a, 
                  const std::pair<T, size_t> b)
                 { 
                 if (std::isnan(a.first) && std::isnan(b.first)) return false; 
                   return std::isnan(b.first) || a.first < b.first; }
                 );
        
        /*
        T unique [vals_size];
        */
        T *unique = new T [vals_size];
        size_t num_unique;
        _rmv_dupl_srtd(srtd_val_idx, unique, &num_unique);
        
        size_t num_quants = std::min(num_unique, quant_per_column);
        quant_size [col_idx] = num_quants;

        /*
        size_t vals_hist [num_unique];
        std::fill_n(vals_hist, num_unique, 0);
        */

        std::cout<<"num uniq: "<<num_unique<<std::endl;
        /*
        T acc_weight [num_unique];
        */
        T *acc_weight = new T[num_unique];

        if (col_idx == t_start_idx){ 
            size_t *vals_hist = new size_t [num_unique];
            std::fill_n(vals_hist, num_unique, 0);

            _fill_time_hist(unique, num_unique, 
                            data, nrows, ncols,
                            t_start_idx, t_end_idx, //pat_idx,
                            vals_hist);
            /*
            T time_diff [num_unique];
            */
            T *time_diff = new T[num_unique];

            std::adjacent_difference (unique, unique+num_unique, time_diff);

            acc_weight[0] = 0;  
            for (size_t i = 1; i<num_unique; ++i){ //multiplying hist count by the duration
                acc_weight [i] = acc_weight[i-1] + time_diff[i] * vals_hist [i-1];
            }
                        
            delete [] time_diff;
        }
        else{
            _fill_non_time_acc_weight(srtd_val_idx, 
                            data, nrows, ncols,
                            t_start_idx, t_end_idx,// col_idx,
                            acc_weight);
            }

        normalize (acc_weight, num_unique, total_t);

        _fill_quants (quant, quant_per_column, num_quants, col_idx,
                          unique, acc_weight, num_unique);        
        delete [] unique;
        delete [] acc_weight;
        srtd_val_idx.clear();
    }
}


void free_boundary_info(boundary_info* bndry_info){
    delete bndry_info; 
}



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
                    ){

    const double* data        = (double *) data_v;
    double* out_data          = (double *) out_data_v;
    bool* is_cat              = (bool   *) is_cat_v;
    const double* quant_arr   = (double *) quant_v;
    const size_t* quant_size  = (size_t *) quant_size_v;

#if defined(_OPENMP)
        omp_set_num_threads(nthreads);
#endif

    preprocessor<double> preprocessor_ (data, nrows, ncols, is_cat, bndry_info, out_data, quant_arr, quant_size, quant_per_column, t_start_idx, t_end_idx, delta_idx, pat_col_idx);
    
    try {
        preprocessor_.preprocess();
    } catch (std::invalid_argument& e){
        std::cout<<err.str();
        //throw;
    }

}


boundary_info* get_boundaries(
        const void* data_v, 
        size_t nrows, 
        size_t ncols, 
        size_t npatients, 
        size_t pat_col_idx, 
        size_t t_start_idx, 
        size_t t_end_idx, 
        const void* quant_v, 
        const void* quant_size_v, 
        size_t quant_per_column
        ){

    const double* data       = (double *) data_v;
    const double* quant_arr  = (double *) quant_v;
    const size_t* quant_size = (size_t *) quant_size_v;
       
    pat_lb_ub_calculator<double> pat_lb_ub_calculator_ = pat_lb_ub_calculator<double>(data, nrows, ncols, npatients, quant_arr, quant_size, quant_per_column, t_start_idx, pat_col_idx, t_end_idx);

    return pat_lb_ub_calculator_.get_boundaries();

}



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
        ){

    const double* data = (double *) data_v;
    const bool* is_cat = (bool *)   is_cat_v;
    double* quant      = (double *) quant_v;
    size_t* quant_size = (size_t *) quant_size_v;

#if defined(_OPENMP)
        omp_set_num_threads(nthreads);
#endif

    if (weighted){
        _compute_quant_weighted<double> (data, nrows, ncols, is_cat, t_start_idx, t_end_idx, pat_idx, delta_idx, quant, quant_size, quant_per_column);
    }
    else{
        _compute_quant<double> (data, nrows, ncols, is_cat, t_start_idx, t_end_idx, pat_idx, delta_idx, quant, quant_size, quant_per_column);
    }
}

void shift_left(
        void* data_v, 
        size_t nrows, 
        size_t ncols, 
        const void* quant_idx_v, 
        const void* quant_v, 
        size_t quant_per_column, 
        int nthreads
        ){

    double* data         = (double *) data_v;
    const double* quant  = (double *) quant_v;
    const int* quant_idx = (int *)    quant_idx_v;

    typedef double T;

#if defined(_OPENMP)
        omp_set_num_threads(nthreads);
#endif

    #pragma omp parallel for schedule(static)
    for (size_t col_idx = 0; col_idx < ncols; ++col_idx){
        size_t quant_idx_ = quant_idx [col_idx];

        auto column_quant = std::vector<T>(
            quant + quant_idx_*     quant_per_column, 
            quant + (quant_idx_+1)* quant_per_column);

        for (size_t row_idx = 0; row_idx < nrows; ++row_idx){
            T val = data [row_idx*ncols + col_idx];

            auto quant_val_iter = min (std::lower_bound (
                    column_quant.begin(), 
                    column_quant.end(), 
                    val), 
                    column_quant.end()-1);

            if (_approx_equal(val, *quant_val_iter)){
                quant_val_iter = max(
                    --quant_val_iter, 
                    column_quant.begin());
                data [row_idx*ncols + col_idx] = *quant_val_iter;
            }            
        }
    }
}

