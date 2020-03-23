#pragma once

__kernel void CudaCalculatePrepare(int n_start, int n_max, double freq_start, double freq_step);

void CudaCalculatePreparePole(int m);

void CudaCalculateIter1Begin(void);

void CudaCalculateIter1Mrqmin1End(void);

void CudaCalculateIter1Mrqmin2End(void);

void CudaCalculateIter1Mrqcof1Start(void);

void CudaCalculateIter1Mrqcof1Matrix(int lpoints);

void CudaCalculateIter1Mrqcof1Curve1(int inrel, int lpoints);

void CudaCalculateIter1Mrqcof1Curve1Last(int inrel, int lpoints);

void CudaCalculateIter1Mrqcof1End(void);

void CudaCalculateIter1Mrqcof2Start(void);

void CudaCalculateIter1Mrqcof2Matrix(int lpoints);

void CudaCalculateIter1Mrqcof2Curve1(int inrel, int lpoints);

void CudaCalculateIter1Mrqcof2Curve1Last(int inrel, int lpoints);

void CudaCalculateIter1Mrqcof2End(void);

void CudaCalculateIter2(void);

void CudaCalculateFinishPole(void);

void CudaCalculateFinish(void);
