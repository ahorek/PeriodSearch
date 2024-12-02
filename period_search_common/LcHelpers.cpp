#include "pch.h"
#include <string>
#include <fstream>
#include <numeric>
#include <cstdio>
#include <algorithm>
#include <unordered_map>
#include <functional>
#include <memory>

#if defined __GNU__
#include <bits/stdc++.h>
#endif

#include "constants.h"
#include "arrayHelpers.hpp"

void processLine15(struct globals& gl, const char* line, int& err) {
    err = sscanf(line, "%d", &gl.Lcurves);
    if (err != 1) {
        err = -1;
        return;
    }
    gl.Lpoints = std::make_unique<int[]>(gl.Lcurves + 2); // +2 instead of +1+1
    std::fill_n(gl.Lpoints.get(), gl.Lcurves + 2, 0);
}

void processLine16(struct globals& gl, const char* line, int& err, int& offset, int& i, int& i_temp, int lineNumber) {
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

    gl.maxLcPoints = *(std::max_element(gl.Lpoints.get(), gl.Lpoints.get() + gl.Lcurves + 1)) + 1;
    gl.ytemp = std::make_unique<double[]>(gl.maxLcPoints + 1);

    gl.dytemp_sizeY = MAX_N_PAR + 1 + 4;
    gl.dytemp_sizeX = gl.maxLcPoints + 1;
    init2Darray(gl.dytemp, gl.dytemp_sizeX, gl.dytemp_sizeY);
    gl.maxDataPoints = std::accumulate(gl.Lpoints.get(), gl.Lpoints.get() + gl.Lcurves, 0);
    gl.Weight = std::make_unique<double[]>(gl.maxDataPoints + 1 + gl.Lcurves);
    gl.Inrel = std::make_unique<int[]>(gl.Lcurves + 1 + gl.Lcurves);
    gl.ave = 0.0;

    return 1;
}