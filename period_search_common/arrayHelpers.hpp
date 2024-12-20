#pragma once
//#include <iostream>

#include "constants.h"
#include <vector>
#include <memory>

/**
 * @brief Initializes a vector with a specified size and initial value.
 *
 * This template function resizes the provided vector to the specified size and
 * initializes all elements with the given initial value.
 *
 * @tparam T The type of the elements in the vector.
 * @param vector A reference to the vector to be initialized.
 * @param size An integer specifying the new size of the vector.
 * @param init_value An optional initial value for the elements of the vector.
 *                   Defaults to a value-initialized T object if not specified.
 */
#if defined _MSC_VER & _MSC_VER < 1900 // Visual Studio 2013 or older
template <typename T>
void init_vector(std::vector<T>& vector, const int size, T init_value = T())
{
    vector.resize(size, init_value); //Resize and initialize the vector with the specified value
}
#else 
template <typename T>
void init_vector(std::vector<T>& vector, const int size, T init_value = T{})
{
    vector.resize(size, init_value); //Resize and initialize the vector with the specified value
}
#endif

/**
 * @brief Initializes a 2D vector (matrix) with specified dimensions and initial value.
 *
 * This template function resizes the provided 2D vector (matrix) to the specified number
 * of rows and columns, and initializes all elements with the given initial value.
 *
 * @tparam T The type of the elements in the matrix.
 * @param matrix A reference to the 2D vector (matrix) to be initialized.
 * @param rows An integer specifying the number of rows in the matrix.
 * @param cols An integer specifying the number of columns in the matrix.
 * @param init_value An optional initial value for the elements of the matrix.
 *                   Defaults to a value-initialized T object if not specified.
 */
#if defined (_MSC_VER) & (_MSC_VER < 1900) // Visual Studio 2013 or older
template <typename T>
void init_matrix(std::vector<std::vector<T>>& matrix, const int rows, const int cols, T init_value = T())
{
    matrix.resize(rows); // Resize the outer vector

    for (int i = 0; i < rows; ++i) {
        matrix[i].resize(cols, init_value); // Resize and initialize each inner vector with the specified value
    }
}
#else
template <typename T>
void init_matrix(std::vector<std::vector<T>>& matrix, const int rows, const int cols, T init_value = T{})
{
    matrix.resize(rows); // Resize the outer vector

    for (int i = 0; i < rows; ++i) {
        matrix[i].resize(cols, init_value); // Resize and initialize each inner vector with the specified value
    }
}
#endif

/**
 * @brief Compares a 2D vector (matrix) with a flattened 1D vector for equality.
 *
 * This template function checks whether a given 2D vector (matrix) is equal to a
 * provided flattened 1D vector. The comparison is done element-wise, and the function
 * ensures that the total number of elements in both vectors match.
 *
 * @tparam T The type of the elements in the vectors.
 * @param matrix A constant reference to the 2D vector (matrix) to be compared.
 * @param flattened A constant reference to the 1D flattened vector to be compared.
 * @return A boolean value indicating whether the matrix and flattened vector are equal.
 *         Returns true if they are equal, otherwise false.
 */
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

/**
 * @brief Flattens a 2D vector (matrix) into a 1D vector.
 *
 * This template function takes a 2D vector (matrix) and converts it into a flattened
 * 1D vector. The function pre-allocates space for the flattened vector for efficiency.
 *
 * @tparam T The type of the elements in the vectors.
 * @param matrix A constant reference to the 2D vector (matrix) to be flattened.
 * @return A 1D vector containing all the elements of the 2D matrix in row-major order.
 */
template <typename T>
std::vector<T> flatten2Dvector(const std::vector<std::vector<T>>& matrix)
{
    const std::size_t rows = matrix.size();
    const std::size_t cols = rows > 0
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

double dot_product(const double a[], const double b[]);
double optimized_dot_product(const double a[], const double b[]);

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
#if _MSC_VER >= 1900 // Visual Studio 2015 or later
    // NOTE: About MSVC - https://learn.microsoft.com/en-us/cpp/cpp/alignment-cpp-declarations?view=msvc-170
    alignas(64) double Nor[3][MAX_N_FAC + 8];
    alignas(64) double Area[MAX_N_FAC + 8];
    alignas(64) double Darea[MAX_N_FAC + 8];
    alignas(64) double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8];
    alignas(64) double dyda[MAX_N_PAR + 16];
#else
    __declspec(align(64)) double Nor[3][MAX_N_FAC + 8];
    __declspec(align(64)) double Area[MAX_N_FAC + 8];
    __declspec(align(64)) double Darea[MAX_N_FAC + 8];
    __declspec(align(64)) double Dg[MAX_N_FAC + 16][MAX_N_PAR + 8];
    __declspec(align(64)) double dyda[MAX_N_PAR + 16];
#endif
#endif

    int Lcurves;
    int maxLcPoints;	// replaces macro MAX_LC_POINTS
    int maxDataPoints;	// replaces macro MAX_N_OBS
    int dytemp_sizeX;
    int dytemp_sizeY;

    // points in every lightcurve 
    std::vector<int> Lpoints;
    std::vector<int> Inrel;

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
