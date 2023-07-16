#pragma once
#if defined (NO_SSE3)
#include <emmintrin.h>
#else
#include <pmmintrin.h>
#endif

//template <class type_info> type_info* aligned_vector(int length);
__m128d* aligned_vector_m128d(int length);

template <class type_info> inline type_info** aligned_matrix(int rows, int colums);
template <class type_info> inline void aligned_deallocate_matrix(type_info** p_x, int rows);

void deallocate_matrix_3(double*** p_x, int n_1, int n_2);