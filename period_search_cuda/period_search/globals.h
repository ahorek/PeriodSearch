#pragma once

#include "constants.h"

extern int l_max, m_max, n_iter, last_call,
           n_coef, num_fac, n_ph_par,
		   deallocate, max_l_points;

extern double o_chi_square, chi_square, a_lambda, a_lamda_incr, a_lamda_start, phi_0, scale,
              area[MAX_N_FAC+1], d_area[MAX_N_FAC+1],
              f_c[MAX_N_FAC+1][MAX_LM+1], f_s[MAX_N_FAC+1][MAX_LM+1],
			  t_c[MAX_N_FAC+1][MAX_LM+1], t_s[MAX_N_FAC+1][MAX_LM+1],
			  d_sphere[MAX_N_FAC+1][MAX_N_PAR+1], d_g[MAX_N_FAC+1][MAX_N_PAR+1],
              normal[MAX_N_FAC+1][3], bl_matrix[4][4],
              pleg[MAX_N_FAC+1][MAX_LM+1][MAX_LM+1],
              d_bl_matrix[3][4][4]; 