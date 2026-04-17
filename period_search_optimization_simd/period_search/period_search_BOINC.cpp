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

    /* Time in JD*/
    std::vector<double> tim(gl.maxDataPoints + 4 + 1, 0.0);
    /* Brightness*/
    std::vector<double> brightness(gl.maxDataPoints + 4 + 1);
    /* Solar phase angle */
    std::vector<double> al(gl.Lcurves + 1, 0.0);
    /* Weights...*/
    std::vector<double> weight_lc(gl.Lcurves + 1, 0.0);
    /* Ecliptic astronomical tempo-centric coordinates of the Sun in AU*/
    double e0[4]{};
    /* Ecliptic astronomical centric coordinates of the Earth in AU*/
    double e[4]{};
    /* Normalization of distance vectors*/
    std::vector<std::vector<double>> ee;
    init_matrix(ee, gl.maxDataPoints + 4 + 1, 3 + 1, 0.0);
    std::vector<std::vector<double>> ee0;
    init_matrix(ee0, gl.maxDataPoints + 4 + 1, 3 + 1, 0.0);

    std::vector<double> sig(gl.maxDataPoints + 4 + 1, 0.0);
    std::vector<double> cg_first(MAX_N_PAR + 1, 0.0);
    std::vector<double> cg(MAX_N_PAR + 1, 0.0);

    std::vector<double> t(MAX_N_FAC + 1, 0.0);
    std::vector<double> f(MAX_N_FAC + 1, 0.0);
    std::vector<double> at(MAX_N_FAC + 1, 0.0);
    std::vector<double> af(MAX_N_FAC + 1, 0.0);
    std::vector<int> ia(MAX_N_PAR + 1, 0);
    std::vector<std::vector<int>> ifp;
    init_matrix(ifp, MAX_N_FAC + 1, 4 + 1, 0);

#if defined __GNUC__
    gl.initializeVectors(MAX_N_PAR + 1, MAX_N_PAR + 8 + 1);
#else
    init_matrix(gl.covar, MAX_N_PAR + 1, MAX_N_PAR + 1, 0.0);
    init_matrix(gl.alpha, MAX_N_PAR + 1, MAX_N_PAR + 8 + 1, 0.0);
#endif

    // open the input file
    FILE* infile = boinc_fopen(input_path, "r");
    if (!infile) {
        printf("error infile");
        //fprintf(stderr,
        //    "%s Couldn't find input file, resolved name %s.\n",
        //    boinc_msg_prefix(buf, sizeof(buf)), input_path
        //);
        std::exit(-1);
    }

    // output file
    boinc_resolve_filename(output_filename, output_path, sizeof(output_path));
    //    out.open(output_path, "w");

        // See if there's a valid checkpoint file.
        // If so seek input file and truncate output file
        //
    boinc_resolve_filename(checkpoint_file, chkpt_path, sizeof(chkpt_path));
    FILE* state = boinc_fopen(chkpt_path, "r");
    if (state) {
        n = fscanf(state, "%d %d %lf %lf %d", &nlines, &new_conw, &conw_r, &sum_dark_facet, &ntestperiods);
        fclose(state);
    }
    if (state && n == 5) {
        checkpoint_exists = 1;
        n_start_from = nlines + 1;
        retval = out.open(output_path, "a");
    }
    else {
        checkpoint_exists = 0;
        n_start_from = 1;
        retval = out.open(output_path, "w");
    }
    if (retval) {
        /*
        fprintf(stderr, "%s APP: period_search output open failed:\n",
            boinc_msg_prefix(buf, sizeof(buf))
        );
        fprintf(stderr, "%s resolved name %s, retval %d\n",
            boinc_msg_prefix(buf, sizeof(buf)), output_path, retval
        );
        */
        printf("error open");
        perror("open");
        std::exit(1);
    }

    /*
#ifdef APP_GRAPHICS
    // create shared mem segment for graphics, and arrange to update it
    //
    shmem = (UC_SHMEM*)boinc_graphics_make_shmem("uppercase", sizeof(UC_SHMEM));
    if (!shmem) {
        fprintf(stderr, "%s failed to create shared mem segment\n",
            boinc_msg_prefix(buf, sizeof(buf))
        );
    }
    update_shmem();
    boinc_register_timer_callback(update_shmem);
#endif
*/

printf("read file");

    int err = 0;

    /* Period interval (hrs) fixed or free */
    err = fscanf_s(infile, "%lf %lf %lf %d", &per_start, &per_step_coef, &per_end, &ia_prd);	fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Epoch of zero time t0 */
    err = fscanf_s(infile, "%lf", &jd_00);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Initial fixed rotation angle fi0 */
    err = fscanf_s(infile, "%lf", &Phi_0);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* The weight factor for conv. reg. */
    err = fscanf_s(infile, "%lf", &conw);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Degree and order of the Laplace series */
    err = fscanf_s(infile, "%d %d", &Lmax, &Mmax);                        fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Number of triangulation rows per octant */
    err = fscanf_s(infile, "%d", &nrows);                                 fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Initial guesses for phase funct. params. */
    err = fscanf_s(infile, "%lf %d", &par[1], &ia_par[1]);                fgets(str_temp, MAX_LINE_LENGTH, infile);
    err = fscanf_s(infile, "%lf %d", &par[2], &ia_par[2]);                fgets(str_temp, MAX_LINE_LENGTH, infile);
    err = fscanf_s(infile, "%lf %d", &par[3], &ia_par[3]);                fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Initial Lambert coeff. (L-S=1) */
    err = fscanf_s(infile, "%lf %d", &cl, &ia_cl);                        fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Maximum number of iterations (when > 1) or
       minimum difference in dev to stop (when < 1) */
    err = fscanf_s(infile, "%lf", &stop_condition);                       fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Minimum number of iterations when stop_condition < 1 */
    err = fscanf_s(infile, "%d", &n_iter_min);                            fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Multiplicative factor for Alamda */
    err = fscanf_s(infile, "%lf", &Alamda_incr);                          fgets(str_temp, MAX_LINE_LENGTH, infile);

    /* Alamda initial value*/
    err = fscanf_s(infile, "%lf", &Alamda_start);                         fgets(str_temp, MAX_LINE_LENGTH, infile);

    if (boinc_is_standalone())
    {
        printf("\n%g  %g  %g  period start/step/stop (%d)\n", per_start, per_step_coef, per_end, ia_prd);
        printf("%g epoch of zero time t0\n", jd_00);
        printf("%g  initial fixed rotation angle fi0\n", Phi_0);
        printf("%g  the weight factor for conv. reg.\n", conw);
        printf("%d %d  degree and order of the Laplace series\n", Lmax, Mmax);
        printf("%d  nr. of triangulation rows per octant\n", nrows);
        printf("%g %g %g  initial guesses for phase funct. params. (%d,%d,%d)\n", par[1], par[2], par[3], ia_par[1], ia_par[2], ia_par[3]);
        printf("%g  initial Lambert coeff. (L-S=1) (%d)\n", cl, ia_cl);
        printf("%g  stop condition\n", stop_condition);
        printf("%d  minimum number of iterations\n", n_iter_min);
        printf("%g  Alamda multiplicative factor\n", Alamda_incr);
        printf("%g  initial Alamda \n\n", Alamda_start);
    }

    /* lightcurves + geometry file */
    /* number of lightcurves and the first realtive one */
    err = fscanf_s(infile, "%d", &gl.Lcurves);

    int ndata = 0;			/* total number of data */
    int k2 = 0;				/* index */
    double al0 = PI;		/* the smallest solar phase angle */
    double al0_abs = PI;
    int ial0 = -1;			/* initialization, index of al0 */
    int ial0_abs = -1;
    double jdMin = 1e20;	/* initial minimum JD (Julian date)*/
    double jdMax = -1e40;	/* initial maximum JD (Julian date)*/
    int onlyrel = 1;
    double jd_0 = jd_00;
    double a = a0;
    double b = b0;
    double c_axis = c0;

    /* Loop over lightcurves */
    for (i = 1; i <= gl.Lcurves; i++)
    {
        double average = 0; /* average */
        err = fscanf_s(infile, "%d %d", &gl.Lpoints[i], &i_temp); /* points in this lightcurve */
        fgets(str_temp, MAX_LINE_LENGTH, infile);

        gl.Inrel[i] = 1 - i_temp;
        if (gl.Inrel[i] == 0)
            onlyrel = 0;

        /* loop over one lightcurve */
        for (j = 1; j <= gl.Lpoints[i]; j++)
        {
            ndata++;

            err = fscanf_s(infile, "%lf %lf", &tim[ndata], &brightness[ndata]); /* JD, brightness */
            err = fscanf_s(infile, "%lf %lf %lf", &e0[1], &e0[2], &e0[3]); /* ecliptic astr_tempocentric coord. of the Sun in AU */
            err = fscanf_s(infile, "%lf %lf %lf", &e[1], &e[2], &e[3]); /* ecliptic astrocentric coord. of the Earth in AU */

            /* selects the minimum and maximum JD */
            if (tim[ndata] < jdMin) jdMin = tim[ndata];
            if (tim[ndata] > jdMax) jdMax = tim[ndata];

            /* normals of distance vectors */
            e0len = sqrt(e0[1] * e0[1] + e0[2] * e0[2] + e0[3] * e0[3]);
            elen = sqrt(e[1] * e[1] + e[2] * e[2] + e[3] * e[3]);

            average += brightness[ndata];

            /* normalization of distance vectors */
            for (k = 1; k <= 3; k++)
            {
                ee[ndata][k] = e[k] / elen;
                ee0[ndata][k] = e0[k] / e0len;
            }

            if (j == 1)
            {
                cos_alpha = dot_product(e, e0) / (elen * e0len);
                al[i] = acos(cos_alpha); /* solar phase angle */
                /* Find the smallest solar phase al0 (not important, just for info) */
                if (al[i] < al0)
                {
                    al0 = al[i];
                    ial0 = ndata;
                }
                if ((al[i] < al0_abs) && (gl.Inrel[i] == 0))
                {
                    al0_abs = al[i];
                    ial0_abs = ndata;
                }
            }
        } /* j, one lightcurve */

        // For Unit test reference only
        /*printArray(ee, ndata, 3, "ee");
        printArray(ee0, ndata, 3, "ee0");*/

        average /= gl.Lpoints[i];
        // For unit test reference only
        //printf("gl.ave: %.30f\n", gl.ave);

        /* Mean brightness of lcurve
           Use the mean brightness as 'sigma' to renormalize the
           mean of each lightcurve to unity */

        for (j = 1; j <= gl.Lpoints[i]; j++)
        {
            k2++;
            sig[k2] = average;
        }

    } /* i, all lightcurves */

    /* initiation of weights */
    for (i = 1; i <= gl.Lcurves; i++)
        weight_lc[i] = -1;

    /* reads weights */
    auto scanResult = 0;
    while (true)
    {
        scanResult = fscanf(infile, "%d", &lc_number);
        if (scanResult <= 0) break;
        scanResult = fscanf(infile, "%lf", &weight_lc[lc_number]);
        if (scanResult <= 0) break;
        if (boinc_is_standalone())
            printf("weights %d %g\n", lc_number, weight_lc[lc_number]);

        if (feof(infile)) break;
    }

    /* If input jd_0 <= 0 then the jd_0 is set to the day before the lowest JD in the data */
    if (jd_0 <= 0)
    {
        jd_0 = static_cast<int>(jdMin);
        if (boinc_is_standalone())
            printf("\nNew epoch of zero time  %f\n", jd_0);
    }

    /* loop over data - subtraction of jd_0 */
    for (i = 1; i <= ndata; i++)
        tim[i] = tim[i] - jd_0;

    // For Unit test reference only
    //printArray(tim, ndata, "tim");

    Phi_0 = Phi_0 * DEG2RAD;

    k = 0;
    for (i = 1; i <= gl.Lcurves; i++)
        for (j = 1; j <= gl.Lpoints[i]; j++)
        {
            k++;
            if (weight_lc[i] == -1)
                gl.Weight[k] = 1;
            else
                gl.Weight[k] = weight_lc[i];
        }

    for (i = 1; i <= 3; i++)
        gl.Weight[k + i] = 1;

    // For Unit tests reference only
    //printArray(Weight, 122, "Weight");

    /* use calibrated data if possible */
    if (onlyrel == 0)
    {
        al0 = al0_abs;
        ial0 = ial0_abs;
    }

    // For unit test reference only
    //printf("al0: %.30f\tial0 %d\n", al0, ial0);

    /* Initial shape guess */
    rfit = sqrt(2 * sig[ial0] / (0.5 * PI * (1 + cos(al0))));
    escl = rfit / sqrt((a * b + b * c_axis + a * c_axis) / 3);
    if (onlyrel == 0)
        escl *= 0.8;
    a = a * escl;
    b = b * escl;
    c_axis = c_axis * escl;
    if (boinc_is_standalone())
    {
        printf("\nWild guess for initial sphere size is %g\n", rfit);
        printf("Suggested scaled a,b,c: %g %g %g\n\n", a, b, c_axis);
    }

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
