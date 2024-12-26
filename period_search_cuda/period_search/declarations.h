#pragma once
#include <vector>

#if _MSC_VER >= 1900 // Visual Studio 2015 or later
#define DEPRECATED(msg) [[deprecated(msg)]]
#elif _MSC_VER >= 1300 // Visual Studio 2012
#define DEPRECATED(msg) __declspec(deprecated(msg))
#else
#define DEPRECATED(msg)
#endif


void trifac(int nrows, std::vector<std::vector<int>>& ifp);

void areanorm(const std::vector<double>& t, const std::vector<double>& f, const int ndir, const int nfac,
	const std::vector<std::vector<int>>& ifp, std::vector<double>& at, std::vector<double>& af);

void sphfunc(const int ndir, const std::vector<double>& at, const std::vector<double>& af);

void ellfit(std::vector<double>& cg, const double a, const double b, const double c, const int ndir, const int ncoef,
	const std::vector<double>& at, const std::vector<double>& af);


void ludcmp(std::vector<std::vector<double>>& a, const int n, std::vector<int>& indx, std::vector<double>& d);

void lubksb(const std::vector<std::vector<double>>& a, const int n, const std::vector<int>& indx, std::vector<double>& b);

int mrqmin(double **x1, double **x2, double x3[], double y[],
			double sig[], double a[], int ia[], int ma,
		double **covar, double **alpha);

double mrqcof(double **x1, double **x2, double x3[], double y[],
			  double sig[], double a[], int ia[], int ma,
		  double **alpha, double beta[], int mfit, int lastone, int lastma);
//void curv(double cg[]);

void blmatrix(double bet, double lam);
double conv(int nc, double dres[], int ma);
void gauss_1(double **aa, int n, double b[]);
void covsrt(double **covar, int ma, int ia[], int mfit);
void phasec(double dcdp[], double alpha, double p[]);
void matrix(double omg, double t, double tmat[][4], double dtm[][4][4]);
double bright(double ee[], double ee0[], double t, double cg[],
			double dyda[], int ncoef);
void shell(int n, double a[], int index[]);

//DEPRECATED("Use std::vector<T> and init_vector() instead")
//double *vector_double(int length);
//
//DEPRECATED("Use std::vector<T> and init_vector() instead")
//int *vector_int(int length);
//
//DEPRECATED("Use std::vector<std::vector<T>> and init_matrix() instead")
//double **matrix_double(int rows, int columns);
//
//DEPRECATED("Use std::vector<std::vector<T>> and init_matrix() instead")
//int **matrix_int(int rows, int columns);
//
//DEPRECATED("")
//double ***matrix_3_double(int n_1, int n_2, int n_3);
//
//DEPRECATED("")
//void deallocate_vector(void *p_x);
//
//DEPRECATED("")
//void deallocate_matrix_double(double **p_x, int rows);
//
//DEPRECATED("")
//void deallocate_matrix_int(int **p_x, int rows);
//
//DEPRECATED("")
//void deallocate_matrix_3(void ***p_x, int n_1, int n_2);

double hapke(double mi0, double mi, double alfa, double sc_param[]);

void sph2cart(double *vektor);
void rotation(double vector[], char *axis, double angle, char *direction);
void inverze(double **a, int n);
void cross_product(double a[], double b[], double c[]);
double norm(double a[]);
double ran1(long *idum);
double gasdev(long *idum);

double raytracing(double sl[], double poz[], int n_fac,
				  double **d, double **e, double **f, double **o,
		  double **normal, double ds[],
			  int n_over_horiz[], int **fac_list, char *sc_law,
		  double sc_param[]);
void precomp(int n_fac, double *x, double *y, double *z, int **fac,
			 double **d, double **e, double **f, double **o,
		 double **normal, double *ds,
		 int *n_over_horiz, int **fac_list);

void matrix_YORP(double omg, double yorp, double t, double tmat[][4], double dtm[][4][4]);
double bright_YORP(double ee[], double ee0[], double t, double cg[],
			double dyda[], int ncoef);

int mrqmin_ell(double **x1, double **x2, double x3[], double y[],
			double sig[], double a[], int ia[], int ma,
		double **covar, double **alpha, double (*funcs)());
double mrqcof_ell(double **x1, double **x2, double x3[], double y[],
			  double sig[], double a[], int ia[], int ma,
		  double **alpha, double beta[], double (*funcs)());

void matrix_ell(double omg, double fi0, double t, double tmat[][4], double dtm[][4][4]);
double bright_ell(double ee[], double ee0[], double t, double cg[],
			double dyda[], int ncoef);

double bright_ell_YORP(double ee[], double ee0[], double t, double cg[],
			double dyda[], int ncoef);
void matrix_ell_YORP(double omg, double fi0, double yorp, double t, double tmat[][4], double dtm[][4][4]);
void ErrorFunction(const char* buffer, int no_conversions);