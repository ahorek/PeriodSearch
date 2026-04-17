/* This program take the input lightcurves, scans over the
   given period range and finds the best period+pole+shape+scattering
   solution. Shape is forgotten. The period, rms residual
   of the fit, and pole solution (lamdda, beta) are given to the output.
   Is starts from six initial poles and selects the best period.
   Reports also pole solution.

   syntax:
   period_search_BOINC

   output: period [hr], rms deviation, chi^2, dark facet [%] lambda_best beta_best

   8.11.2006

   new version of lightcurve files (new input lcs format)
   testing the dark facet, finding the optimal value for convexity weight: 0.1, 0.2, 0.4, 0.8, ... <10.0
   first line of output: fourth column is the optimized conw (not dark facet), all other lines include dark facet

   16.4.2012

   version for BOINC

*/

// This file is part of BOINC.
// http://boinc.berkeley.edu
// Copyright (C) 2008 University of California
//
// BOINC is free software; you can redistribute it and/or modify it
// under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation,
// either version 3 of the License, or (at your option) any later version.
//
// BOINC is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with BOINC.  If not, see <http://www.gnu.org/licenses/>.

// ReSharper disable CppClangTidyCertErr33C
// ReSharper disable CppClangTidyPerformanceAvoidEndl
// ReSharper disable CppClangTidyConcurrencyMtUnsafe
// ReSharper disable CppClangTidyCertErr34C
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#if defined _DEBUG
// #include <ctime>
#include <time.h>
#endif

#include "declarations.h"
#include "constants.h"
#include "globals.h"

#ifdef _WIN32
#include "boinc_win.h"
#include <shlwapi.h>
#include "winbase.h"
#else
#include "config.h"
#include <cstdio>
#include <cctype>
// #include <ctime>
#include <cstring>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <iostream>
#endif

#ifdef __GNUC__
#include <filesystem>
#endif

#include "str_util.h"
#include "util.h"
#include "filesys.h"
#include "boinc_api.h"
#include "mfile.h"
#include "arrayHelpers.hpp"
#include "systeminfo.h"
#include "Enums.h"
#include "CalcStrategy.hpp"
#include "CalcStrategyNone.hpp"
#include "LcHelpers.hpp"
#include "SIMDHelpers.h"

#ifdef APP_GRAPHICS
#include "graphics2.h"
#include "uc2.h"
UC_SHMEM* shmem;
#endif

#if !defined _WIN32
#include <stdarg.h>

int fscanf_s(FILE* file, const char* format, ...) {
    va_list args;
    va_start(args, format);
    int result = vfscanf(file, format, args);
    va_end(args);

    if (result == EOF) {
        fprintf(stderr, "\nError: reading input\n"); fflush(stderr); std::exit(2);
    }
    else if (result == 0) {
        fprintf(stderr, "\nError: input format mismatch\n"); fflush(stderr); std::exit(2);
    }
    return result;
}
#endif

CalcContext calcCtx(std::allocate_shared<CalcStrategyNone>(AlignedAllocator<CalcStrategyNone>(64)));
SIMDSupport CPUopt;

constexpr auto checkpoint_file = "period_search_state";
constexpr auto input_filename = "period_search_in";
constexpr auto output_filename = "period_search_out";

int DoCheckpoint(MFILE& mf, const int nlines, const int newconw, const double conwr, const double sumdarkfacet, const int testperiods)
{
    std::string resolvedName;

    const auto file = fopen("temp", "w");
    if (!file) return 1;
	fprintf(file, "%d %d %.17g %.17g %d", nlines, newconw, conwr, sumdarkfacet, testperiods);
	fclose(file);

	auto retval = mf.flush();
	if (retval) return retval;
    boinc_resolve_filename_s(checkpoint_file, resolvedName);
	retval = boinc_rename("temp", resolvedName.c_str());
	if (retval) return retval;

    return 0;
}

#ifdef APP_GRAPHICS
void update_shmem() {
    if (!shmem) return;

    // always do this; otherwise a graphics app will immediately
    // assume we're not alive
    shmem->update_time = dtime();

    // Check whether a graphics app is running,
    // and don't bother updating shmem if so.
    // This doesn't matter here,
    // but may be worth doing if updating shmem is expensive.
    //
    if (shmem->countdown > 0) {
        // the graphics app sets this to 5 every time it renders a frame
        shmem->countdown--;
    }
    else {
        return;
    }
    shmem->fraction_done = boinc_get_fraction_done();
    shmem->cpu_time = boinc_worker_thread_cpu_time();;
    boinc_get_status(&shmem->status);
}
#endif

//#if defined __GNUC__
//// Helper function to allocate aligned memory
//void* allocate_aligned_memory(std::size_t alignment, std::size_t size) {
//    void* ptr = nullptr;
//    if (posix_memalign(&ptr, alignment, size) != 0) {
//        throw std::bad_alloc();
//    }
//    return ptr;
//}
//
//// Wrapper function to create an aligned std::vector
//std::vector<double> create_aligned_vector(std::size_t size, std::size_t alignment = 64)
//{
//    double* aligned_memory = static_cast<double*>(allocate_aligned_memory(alignment, size * sizeof(double)));
//    return std::vector<double>(aligned_memory, aligned_memory + size);
//}
//#endif


/* global parameters */
int Lmax, Mmax, Niter, Lastcall,
Ncoef, Numfac, Nphpar,
Deallocate, n_iter;

double Ochisq, Chisq, Alamda, Alamda_incr, Alamda_start, Phi_0, Scale,

Fc[MAX_N_FAC + 1][MAX_LM + 1], Fs[MAX_N_FAC + 1][MAX_LM + 1],
Tc[MAX_N_FAC + 1][MAX_LM + 1], Ts[MAX_N_FAC + 1][MAX_LM + 1],
Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1],
Blmat[4][4],
Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1],
Dblm[3][4][4];

std::vector<double> atry;
std::vector<double> beta;
std::vector<double> da;

// NOTE: RPi related:
//void blinkLed(int count) {
//	for (int i = 0; i < count; i++) {
//		digitalWrite(LED, HIGH);  // On
//		delay(150); // ms
//		digitalWrite(LED, LOW);	  // Off
//		delay(150);
//	}
//}

int main(int argc, char** argv)
{
    printf("start");
    exit(0);

}

#ifdef _WIN32

int WINAPI WinMain(_In_ HINSTANCE hInst, _In_opt_ HINSTANCE hPrevInst, _In_ LPSTR Args, _In_ int WinMode)
{
	LPSTR command_line;
	char* argv[100];
	int argc;

	command_line = GetCommandLine();
	argc = parse_command_line(command_line, argv);
	return main(argc, argv);
}

#endif
