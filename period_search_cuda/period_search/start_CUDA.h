#pragma once
#include <vector>

int CUDAPrepare(int cudadev, double* beta_pole, double* lambda_pole, double* par, double cl, double Alamda_start, double Alamda_incr, 
    std::vector<std::vector<double>>& ee, std::vector<std::vector<double>>& ee0, std::vector<double>& tim,
    double Phi_0, int checkex, int ndata, struct globals& gl);

int CUDAPrecalc(int cudadev, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double* conw_r,
    int ndata, std::vector<int>& ia, int* ia_par, int* new_conw, std::vector<double>& cg_first, std::vector<double>& sig, int Numfac, 
    std::vector<double>& brightness, struct globals& gl);

int CUDAStart(int cudadev, int n_start_from, double freq_start, double freq_end, double freq_step, double stop_condition, int n_iter_min, double conw_r,
    int ndata, std::vector<int>& ia, int* ia_par, std::vector<double>& cg_first, MFILE& mf, double escl, std::vector<double>& sig, int Numfac,
    std::vector<double>& brightness, struct globals& gl);

int DoCheckpoint(MFILE& mf, int nlines, int newConw, double conwr);

void CUDAFree(void);

void GetCUDAOccupancy(const int cudaDevice);
