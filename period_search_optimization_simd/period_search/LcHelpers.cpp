#include <string>
#include <fstream>
#include <numeric>
#include <cstdio>

#include "constants.h"
#include "declarations.h"
#include "arrayHelpers.hpp"

///<summary>
///Performes the first loop over lightcurves to find all data points (replacing MAX_N_OBS)
///</summary>
///<param name="gl"></param>
///<param name="filename"></param>
void prepareLcData(struct globals &gl, const char *filename)
{
	int err = 0;
	int i_temp;

	std::ifstream file(filename);
	std::string line;
	gl.Lcurves = 0;

	if (file.is_open())
	{
		int lineNumber = 0;
		int offset = 0;
		int i = 0;

		while (std::getline(file, line))
		{
			lineNumber++;
			switch (lineNumber)
			{
				case 15:
					err = sscanf_s(line.c_str(), "%d", &gl.Lcurves);
					gl.Lpoints = new int[gl.Lcurves + 1 + 1];
					//continue;
					break;
				case 16:
					sscanf_s(line.c_str(), "%d %d", &gl.Lpoints[0], &i_temp);
					offset = lineNumber;
					i++;
					//continue;
					break;
			}

			if (lineNumber <= 16)
			{
				continue;
			}

			if (lineNumber == offset + 1 + gl.Lpoints[i - 1])
			{
				sscanf_s(line.c_str(), "%d %d", &gl.Lpoints[i], &i_temp);
				offset = lineNumber;
				i++;
				if (i == gl.Lcurves) break;
			}
		}

		file.close();

		gl.maxLcPoints = *(std::max_element(gl.Lpoints, gl.Lpoints + gl.Lcurves + 1)) + 1;
		gl.ytemp = new double[gl.maxLcPoints + 1];

		gl.dytemp_sizeY = MAX_N_PAR + 1 + 4;
		int dytemp_siszeX = gl.maxLcPoints + 1;
		init2Darray(gl.dytemp, dytemp_siszeX, gl.dytemp_sizeY);

		//gl.maxDataPoints = std::accumulate(gl.Lpoints, gl.Lpoints + (gl.maxLcPoints + 1 + gl.Lcurves + 1), 0);
		gl.maxDataPoints = std::accumulate(gl.Lpoints, gl.Lpoints + gl.Lcurves, 0);
		gl.Weight = new double[gl.maxDataPoints + 1 + gl.Lcurves];
		gl.Inrel = new int[gl.Lcurves + 1 + gl.Lcurves];
	}
	else
	{
		fprintf(stderr, "\nCouldn't find input file, resolved name %s.\n", filename);
		fflush(stderr);

		exit(2);
	}
}