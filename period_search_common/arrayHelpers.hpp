#pragma once
#include <iostream>

#include "constants.h"
#include <vector>
#include <memory>

// Template function to initialize a vector in any structure
template <typename T>
void init_vector(std::vector<T>& vector, const int size, T init_value = T{})
{
    vector.resize(size + 1, init_value); // // Resize and initialize the vector with the specified value
}

// Template function to initialize a 2D vector (matrix) in any structure
template <typename T>
void init_matrix(std::vector<std::vector<T>>& matrix, const int rows, const int cols, T init_value = T{})
{
    matrix.resize(rows + 1); // Resize the outer vector

    for (int i = 0; i < rows + 1; ++i) {
        matrix[i].resize(cols + 1, init_value); // Resize and initialize each inner vector with the specified value
    }
}

template <typename T>
bool compareVectors(const std::vector<std::vector<T>>& matrix, const std::vector<T>& flattened)
{
    int rows = matrix.size();
    int cols = (rows > 0)
        ? matrix[0].size()
        : 0;

    // Check if the total number of elements match
    if (flattened.size() != rows * cols)
    {
        return false;
    }

    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            if (matrix[i][j] != flattened[i * cols + j])
            {
                return false;
            }
        }
    }
    return true;
}

// Flatten the 2D vector to a 1D vector
template <typename T>
std::vector<T> flatten2Dvector(const std::vector<std::vector<T>>& matrix)
{
    const size_t rows = matrix.size();
    const size_t cols = rows > 0
        ? matrix[0].size()
        : 0;

    std::vector<T> flattened_vec;
    flattened_vec.reserve(rows * cols); // Pre-allocate space for efficiency

    for (const auto& row : matrix)
    {
        flattened_vec.insert(flattened_vec.end(), row.begin(), row.end());
    }

    //if (compareVectors(matrix, flattened_vec))
    //{
    //    std::cout << "The vectors match!" << std::endl;
    //}
    //else
    //{
    //    std::cout << "The vectors do not match!" << std::endl;
    //}

    return flattened_vec;
}


void init2Darray(std::vector<std::unique_ptr<double[]>>& matrix, int xSize, int ySize);
void init2Darray(double**& matrix, int dytemp_siszeX, int dytemp_sizeY);
void delete2Darray(double**& ary, int sizeY);
void printArray(int array[], int iMax, char msg[]);
void printArray(double array[], int iMax, char msg[]);
void printArray(double** array, int iMax, int jMax, char msg[]);
void printArray(double*** array, int iMax, int jMax, int kMax, char msg[]);

struct globals
{
#ifdef __GNUC__
    double Nor[3][MAX_N_FAC + 8] __attribute__((aligned(64))),
        Area[MAX_N_FAC + 8] __attribute__((aligned(64))),
        Darea[MAX_N_FAC + 8] __attribute__((aligned(64))),
        Dg[MAX_N_FAC + 16][MAX_N_PAR + 8] __attribute__((aligned(64)));
    double dyda[MAX_N_PAR + 16] __attribute__((aligned(64)));
#else
    // NOTE: About MSVC - https://learn.microsoft.com/en-us/cpp/cpp/alignment-cpp-declarations?view=msvc-170
    alignas(64) double Nor[3][MAX_N_FAC + 8];
    alignas(64) double Area[MAX_N_FAC + 8];
    alignas(64) double Darea[MAX_N_FAC + 8];
    alignas(64) double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8];
    alignas(64) double dyda[MAX_N_PAR + 16];
#endif

    int Lcurves;
    int maxLcPoints;	// replaces MAX_LC_POINTS
    int maxDataPoints;	// replaces MAX_N_OBS
    int dytemp_sizeX;
    int dytemp_sizeY;
    // points in this lightcurve 
    std::vector<int> Lpoints;	// int*
    std::vector<int> Inrel;		// int*

    double ymod;
    double wt;
    double sig2i;
    double dy;
    double coef;
    double wght;
    double ave;
    double xx1[4];
    double xx2[4];
    double dave[MAX_N_PAR + 1 + 4];
    std::vector<double> ytemp;
    std::vector<double> Weight;
    std::vector<std::vector<double>> dytemp;
};
