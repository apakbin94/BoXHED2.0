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
    boundary_info_(size_t npatients_, size_t out_nrows_, size_t* in_lbs_, size_t* out_lbs_):
        npatients(npatients_),
        out_nrows(out_nrows_),
        in_lbs(in_lbs_),
        out_lbs(out_lbs_)
        {}

    size_t  npatients;
    size_t  out_nrows;
    size_t* in_lbs;
    size_t* out_lbs;

    ~boundary_info_(){
        delete []in_lbs;
        delete []out_lbs;
    }
} boundary_info;
