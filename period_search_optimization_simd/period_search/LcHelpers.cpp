#include <string>
#include <fstream>
#include <numeric>
#include <cstdio>
// #include <stdio.h>
#include <algorithm>

#if defined __GNU__
#include <bits/stdc++.h>
#endif

#include "constants.h"
#include "declarations.h"
#include "arrayHelpers.hpp"

///< summary>
/// Performes the first loop over lightcurves to find all data points (replacing MAX_LC_POINTS, MAX_N_OBS, etc.)
///</summary>
///< param name="gl"></param>
///< param name="filename"></param>
void prepareLcData(struct globals &gl, const char *filename)
{
	int err = 0;
	int i_temp;

	std::ifstream file(filename);
	std::string lineStr;
	char line[MAX_LINE_LENGTH];
	gl.Lcurves = 0;

	if (file.is_open())
	{
		int lineNumber = 0;
		int offset = 0;
		int i = 0;

		// char *a;
		// char *b;
		// char *x;

		while (std::getline(file, lineStr))
		{
			std::strcpy(line, lineStr.c_str());
			lineNumber++;
			switch (lineNumber)
			{
				case 15:
					err = sscanf(line, "%d", &gl.Lcurves);
					// err = sscanf(line.c_str(), "%s", x);
					// fprintf(stderr, "%s Line 15 was red", x);
					// gl.Lcurves = std::stoi(x);
					gl.Lpoints = new int[gl.Lcurves + 1 + 1];
					std::fill_n(gl.Lpoints, gl.Lcurves + 1 + 1, 0);
					// continue;
					break;
				case 16:
					err = sscanf(line, "%d %d", &gl.Lpoints[0], &i_temp);
					// err = sscanf(line.c_str(), "%s %s", a, b);
					// gl.Lpoints[0] = std::stoi(a);
					// i_temp = std::stoi(b);
					offset = lineNumber;
					i++;
					// continue;
					break;
			}

			if (lineNumber <= 16)
			{
				continue;
			}

			if (lineNumber == offset + 1 + gl.Lpoints[i - 1])
			{
				err = sscanf(line, "%d %d", &gl.Lpoints[i], &i_temp);
				// err = sscanf(line.c_str(), "%s %s", a, b);

				// gl.Lpoints[0] = std::stoi(a);
				// i_temp = std::stoi(b);
				offset = lineNumber;
				i++;
				if (i == gl.Lcurves)
					break;
			}
		}

		file.close();

		gl.maxLcPoints = *(std::max_element(gl.Lpoints, gl.Lpoints + gl.Lcurves + 1)) + 1;
		gl.ytemp = new double[gl.maxLcPoints + 1];

		gl.dytemp_sizeY = MAX_N_PAR + 1 + 4;
		gl.dytemp_sizeX = gl.maxLcPoints + 1;
		init2Darray(gl.dytemp, gl.dytemp_sizeX, gl.dytemp_sizeY);

		gl.maxDataPoints = std::accumulate(gl.Lpoints, gl.Lpoints + gl.Lcurves, 0);
		gl.Weight = new double[gl.maxDataPoints + 1 + gl.Lcurves];
		gl.Inrel = new int[gl.Lcurves + 1 + gl.Lcurves];
		gl.ave = 0.0;
	}
	else
	{
		fprintf(stderr, "\nCouldn't find input file, resolved name %s.\n", filename);
		fflush(stderr);

		exit(2);
	}
}