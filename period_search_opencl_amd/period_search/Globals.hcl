struct freq_context2
{
	double Area[MAX_N_FAC + 1];
	double Dg[(MAX_N_FAC + 1) * (MAX_N_PAR + 1)];
	double covar[MAX_N_PAR+1][MAX_N_PAR+1];
	double freq;
	double Ochisq, Chisq, Alamda;
	double Blmat[4][4];
	double Dblm[3][4][4];
	double iter_diff, rchisq, dev_old, dev_new;
	double trial_chisq, ave;
	int Niter;
	int isInvalid, isAlamda, isNiter;
	int np, np1, np2;
	double e_1[POINTS_MAX + 1], e_2[POINTS_MAX + 1], e_3[POINTS_MAX + 1], e0_1[POINTS_MAX + 1], e0_2[POINTS_MAX + 1], e0_3[POINTS_MAX + 1];
	double de[POINTS_MAX + 1][4][4], de0[POINTS_MAX + 1][4][4];
	double jp_Scale[POINTS_MAX + 1];
	double jp_dphp_1[POINTS_MAX + 1], jp_dphp_2[POINTS_MAX + 1], jp_dphp_3[POINTS_MAX + 1];
	int indxc[MAX_N_PAR + 1], indxr[MAX_N_PAR + 1], ipiv[MAX_N_PAR + 1];
	double cg[MAX_N_PAR + 1];
	double dytemp[(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	double ytemp[POINTS_MAX + 1];
	double dyda[MAX_N_PAR + 1], dave[MAX_N_PAR + 1];
	double alpha[MAX_N_PAR + 1];
	double atry[MAX_N_PAR + 1], beta[MAX_N_PAR + 1], da[MAX_N_PAR + 1];
};

struct freq_result
{
	int isReported, isInvalid;
	double dark_best, per_best, dev_best, la_best, be_best, freq;
};

struct FuncArrays
{
	int Mmax, Lmax;
	int ma, Nphpar;
	int Ncoef, Ncoef0;
	int Numfac, Numfac1;
	int Lmfit, Lmfit1;
	int Dg_block;
	double Phi_0;
	int ia[MAX_N_PAR + 1];
	double tim[MAX_N_OBS + 1];
	double Darea[MAX_N_FAC + 1];
	double ee[MAX_N_OBS + 1][3];
	double Nor[MAX_N_FAC + 1][3];
	double ee0[MAX_N_OBS + 1][3];
	double Fc[MAX_N_FAC + 1][MAX_LM + 1];
	double Fs[MAX_N_FAC + 1][MAX_LM + 1];
	double Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	double Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
};

typedef struct FuncArrays varholder;
