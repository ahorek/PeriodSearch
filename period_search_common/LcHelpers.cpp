#include "pch.h"
#include <string>
#include <fstream>
#include <numeric>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <memory>
#include <vector>

#if defined __GNU__
#include <bits/stdc++.h>
#endif

#include "constants.h"
#include "arrayHelpers.hpp"

void processLine15(struct globals& gl, const char* line, int& err)
{
    err = sscanf(line, "%d", &gl.Lcurves);
    if (err != 1) {
        err = -1;
        return;
    }
    
    gl.Lpoints.resize(gl.Lcurves + 2, 0);
}

void processLine16(struct globals& gl, const char* line, int& err, int& offset, int& i, int& i_temp, int lineNumber)
{
    err = sscanf(line, "%d %d", &gl.Lpoints[0], &i_temp);
    if (err != 2) {
        err = -1;
        return;
    }

    offset = lineNumber;
    i++;
}

void processSubsequentLines(struct globals& gl, const char* line, int& err, int& offset, int& i, int& i_temp, int lineNumber) {
    err = sscanf(line, "%d %d", &gl.Lpoints[i], &i_temp);
    if (err != 2) {
        err = -1;
        return;
    }
    offset = lineNumber;
    i++;
}

//// Template function to initialize a 2D vector (matrix) in any structure
//template <typename T>
//void init_matrix(std::vector<std::vector<T>>& matrix, int rows, int cols, T init_value = T{})
//{
//    matrix.resize(rows + 1); // Resize the outer vector
//
//    for (int i = 0; i < rows + 1; ++i) {
//        matrix[i].resize(cols + 1, init_value); // Resize and initialize each inner vector with the specified value
//    }
//}

/// Convexity regularization: make one last 'lightcurve' that
/// consists of the three comps.of the residual non-convex vectors
/// that should all be zero
/// @param gl struct of globals
void MakeConvexityRegularization(struct globals& gl)
{
    gl.Lcurves = gl.Lcurves + 1;
    gl.Lpoints[gl.Lcurves] = 3;
    gl.Inrel[gl.Lcurves] = 0;

    // keep it '+ 1' instead of ' + 2' as the gl.Lcurves has been incremented by 1 already!7
    gl.maxDataPoints = std::accumulate(gl.Lpoints.begin(), gl.Lpoints.end(), 0); 

    //for (auto q = 0; q <= gl.Lcurves; q++)
    //    fprintf(stderr, "Lpoints[%d] %d\n", q, gl.Lpoints[q]);
}

///< summary>
/// Performs the first loop over lightcurves to find all data points (replacing MAX_LC_POINTS, MAX_N_OBS, etc.)
///</summary>
///< param name="gl"></param>
///< param name="filename"></param>
int PrepareLcData(struct globals& gl, const char* filename)
{
    int i_temp;
    int err = 0;

    std::ifstream file(filename);
    std::string lineStr;
    gl.Lcurves = 0;

    if (!file.is_open())
    {
        return 2;
    }

    int lineNumber = 0;
    int offset = 0;
    int i = 0;

    std::unordered_map<int, std::function<void(const char*, int&)>> actions;
    actions[15] = [&](const char* line, int& err) { processLine15(gl, line, err); };
    actions[16] = [&](const char* line, int& err) { processLine16(gl, line, err, offset, i, i_temp, lineNumber); };

    while (std::getline(file, lineStr))
    {
        char line[2000];
        std::strcpy(line, lineStr.c_str());
        lineNumber++;

        if (actions.find(lineNumber) != actions.end()) {
            actions[lineNumber](line, err);
            if (err <= 0) {
                file.close();
                return err;
            }
        }

        if (lineNumber <= 16)
        {
            continue;
        }

        if (lineNumber == offset + 1 + gl.Lpoints[i - 1])
        {
            processSubsequentLines(gl, line, err, offset, i, i_temp, lineNumber);
            if (err <= 0) {
                file.close();
                return err;
            }
            if (i == gl.Lcurves)
            {
                break;
            }
        }
    }

    file.close();

    gl.Inrel.resize(gl.Lcurves + 2, 0);
    gl.maxLcPoints = *(std::max_element(gl.Lpoints.begin(), gl.Lpoints.end()));

    gl.ytemp.resize(gl.maxLcPoints + 2, 0.0);        // Not used in CUDA

    gl.dytemp_sizeY = MAX_N_PAR + 1 + 4;
    gl.dytemp_sizeX = gl.maxLcPoints + 2;
    init_matrix(gl.dytemp, gl.dytemp_sizeX, gl.dytemp_sizeY, 0.0);  // Not used in CUDA

    gl.maxDataPoints = std::accumulate(gl.Lpoints.begin(), gl.Lpoints.end(), 0);   // OK

    gl.Weight.resize(gl.maxDataPoints + 1 + 4, 0.0);
    gl.ave = 0.0;

    return 1;
}