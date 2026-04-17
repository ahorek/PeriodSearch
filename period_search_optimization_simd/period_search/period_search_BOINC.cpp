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
//#include "boinc_win.h"
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
//#include "boinc_api.h"
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
    /*
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
    */

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


#include <cstring>
#include <iostream>

#define BOINC_SUCCESS		0
#define EXIT_CHILD_FAILED	1

struct BOINC_OPTIONS
{
	int normal_thread_priority;
	int main_program;
	int check_heartbeat;
	int handle_process_control;
	int send_status_msgs;
	int direct_process_action;
	int multi_thread;
	int multi_process;
};

inline void boinc_options_defaults(BOINC_OPTIONS &) {}
inline int boinc_init_options(BOINC_OPTIONS *) { std::cout << "boinc_init()" << std::endl; return 0; }
inline int boinc_finish(const int status) { std::cout << "boinc_finish(" << status << ")" << std::endl; exit(status); /* never reached */ return 0; }

inline int boinc_resolve_filename(const char * const virtual_name, char * const physical_name, const int len)
{
	strncpy(physical_name, virtual_name, size_t(len - 1));
	return 0;
}

inline FILE * boinc_fopen(const char * const path, const char * const mode)
{
	return std::fopen(path, mode);
}

inline int boinc_is_standalone() { return 1; }

inline int boinc_time_to_checkpoint() { static int cnt = 0; if (++cnt == 20) { cnt = 0; return 1; } return 0; }
inline int boinc_checkpoint_completed() { return 0; }

inline int boinc_fraction_done(const double f) { std::cout << "boinc_fraction_done(" << f << ")" << std::endl; return 0; }

struct BOINC_STATUS { int no_heartbeat, suspended, quit_request, abort_request; };

inline int boinc_get_status(BOINC_STATUS * const status)
{
	// std::cout << "boinc_get_status" << std::endl;
	status->no_heartbeat = status->suspended = status->quit_request = status->abort_request = 0;
	static int cnt = 0;
	if ((++cnt >= 10) && (cnt < 20)) status->suspended = 1;
	if (cnt >= 1000) status->suspended = status->abort_request = 1;
	return 0;
}


int main(int argc, char** argv)
{
    printf("start");

    int nlines = 0, ntestperiods, checkpoint_exists, n_start_from;
    char input_path[512], output_path[512], chkpt_path[512], buf[256];
    MFILE out;

    int i, j, l, m, k, n = 0, nrows, ndir, i_temp,
        n_iter_max, n_iter_min,
        ia_prd, ia_par[4]{}, ia_cl,
        lc_number,
        new_conw, max_test_periods,
        ma = 0;

    double per_start, per_step_coef, per_end,
        freq, freq_start, freq_step, freq_end,
        dev_old, dev_new, iter_diff, iter_diff_max, stop_condition,
        totarea, sum, dark, dev_best, per_best, dark_best, la_tmp, be_tmp, la_best, be_best, fraction_done,
        sum_dark_facet = 0.0, ave_dark_facet;

    double jd_00, conw, conw_r, a0 = 1.05, b0 = 1.00, c0 = 0.95,
        prd, cl, e0len, elen, cos_alpha,
        dth, dph, rfit, escl,
        chck[4]{},
        par[4]{}, rchisq;

    auto* str_temp = static_cast<char*>(malloc(MAX_LINE_LENGTH));

    double lambda_pole[N_POLES + 1] = { 0.0, 0.0, 90.0, 180.0, 270.0, 60.0, 180.0, 300.0, 60.0, 180.0, 300.0 };
    double beta_pole[N_POLES + 1] = { 0.0, 0.0, 0.0, 0.0, 0.0, 60.0, 60.0, 60.0, -60.0, -60.0, -60.0 };

    int ia_lambda_pole = 1;
    int ia_beta_pole = 1;

    //wiringPiSetupSys();
    //pinMode(LED, OUTPUT);

    printf("preboinc");

    int retval = 0;//boinc_init();
    //if (retval)
    //{
    //    fprintf(stderr, "%s boinc_init returned %d\n", boinc_msg_prefix(buf, sizeof(buf)), retval);
    //    std::exit(retval);
    //}

    printf("after boinc");

    // resolve logical name first
    boinc_resolve_filename(input_filename, input_path, sizeof(input_path));

    auto gl = globals();
    auto res = PrepareLcData(gl, input_path);
    if (res <= 0)
    {
        fprintf(stderr, "\nCouldn't find input file, resolved name %s.\n", input_path);
        fflush(stderr);
    }

    printf("after lc");

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
