file(REMOVE_RECURSE
  "../lib/libxgboost.pdb"
  "../lib/libxgboost.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/xgboost.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
