//#include "pch.h"
//#include "stdafx.h"
#include <numeric> // For std::inner_product double
#include "arrayHelpers.hpp"

#if defined __GNUC__
#pragma GCC optimize ("O2")
#pragma GCC target ("avx2")
#else
#pragma optimize( "gt", on ) // Enable global optimization and speed optimization
//#pragma clang attribute push (__attribute__((target("avx2"))), apply_to=function) // TODO: Needs to be updated against MSVC compiler
#endif

/**
 * @brief Computes the dot product of two 1-based indexed arrays using std::inner_product.
 *
 * This function computes the dot product of two 1-based indexed arrays using the
 * standard library function std::inner_product, which may be optimized by the compiler.
 * It does not use the zero elements of the arrays.
 * @param a An array of doubles with at least 4 elements, where only a[1], a[2], and a[3] are used.
 * @param b An array of doubles with at least 4 elements, where only b[1], b[2], and b[3] are used.
 * @return The dot product of the arrays.
 */
double optimized_dot_product(const double a[4], const double b[4])
{
    // Adjust pointers to 1-based indexing
    const double c = std::inner_product(a + 1, a + 4, b + 1, 0.0);

    return c;
}

#if defined __GNUC__
#pragma GCC reset_options
#else
//#pragma clang attribute pop
#endif

/**
 * @brief Computes the dot product of two 1-based indexed arrays.
 *
 * This function computes the dot product of two 1-based indexed arrays.
 * It does not use the zero elements of the arrays.
 *
 * @param a An array of doubles with at least 4 elements, where only a[1], a[2], and a[3] are used.
 * @param b An array of doubles with at least 4 elements, where only b[1], b[2], and b[3] are used.
 * @return The dot product of the arrays.
 */
double dot_product(const double a[4], const double b[4])
{
    const double c = a[1] * b[1] + a[2] * b[2] + a[3] * b[3];

    return c;
}

void dot_product_new(double a[4], double b[4], double &c)
{
	c = a[1] * b[1];
	c += (a[2] * b[2]);
	c += (a[3] * b[3]);
}

