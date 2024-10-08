#pragma once

void printArray(int array[], int iMax, char msg[]);
void printArray(double array[], int iMax, char msg[]);
void printArray(double **array, int iMax, int jMax, char msg[]);
void printArray(double ***array, int iMax, int jMax, int kMax, char msg[]);

struct globals
{
	int Lcurves;
	int maxLcPoints;	// replaces MAX_LC_POINTS
	int maxDataPoints;	// replaces MAX_N_OBS
	int dytemp_sizeY;
	int* Lpoints;
	int* Inrel;

	double* ytemp;
	double* Weight;
	double** dytemp;
};

