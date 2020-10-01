struct freq_context2
{
	double chck[4];
	double Area[MAX_N_FAC + 1];
	double Dg[(MAX_N_FAC + 1) * (MAX_N_PAR + 1)];
	double Sig[MAX_N_OBS + 1], Weight[MAX_N_OBS + 1], Brightness[MAX_N_OBS + 1];
	double freq;
	double Ochisq, Chisq, Alamda;
	double Blmat[4][4];
	double Dblm[3][4][4];
	double iter_diff, rchisq, dev_old, dev_new;
	double trial_chisq, ave;
	int Niter;
	int isInvalid, isAlamda, isNiter;
	int n, np, np1, np2;
	int indxc[MAX_N_PAR + 1];
	int indxr[MAX_N_PAR + 1];
	int ipiv[MAX_N_PAR + 1];
	double e_1[POINTS_MAX + 1];
	double e_2[POINTS_MAX + 1];
	double e_3[POINTS_MAX + 1];
	double e0_1[POINTS_MAX + 1];
	double e0_2[POINTS_MAX + 1];
	double e0_3[POINTS_MAX + 1];
	double de[POINTS_MAX + 1][4][4];
	double de0[POINTS_MAX + 1][4][4];
	double jp_Scale[POINTS_MAX + 1];
	double jp_dphp_1[POINTS_MAX + 1];
	double jp_dphp_2[POINTS_MAX + 1];
	double jp_dphp_3[POINTS_MAX + 1];
	double cg[MAX_N_PAR + 1];
	double dytemp[(POINTS_MAX + 1) * (MAX_N_PAR + 1)];
	double ytemp[POINTS_MAX + 1];
	double dyda[MAX_N_PAR + 1];
	double dave[MAX_N_PAR + 1];
	double alpha[MAX_N_PAR + 1];
	double atry[MAX_N_PAR + 1];
	double beta[MAX_N_PAR + 1];
	double da[MAX_N_PAR + 1];
	double covar[MAX_N_PAR + 1];
};

struct freq_result
{
	int isReported, isInvalid;
	double dark_best, per_best, dev_best, la_best, be_best, freq;
};

typedef struct FuncArrays
{
	int Mmax, Lmax;
	int ma, Nphpar;
	int Ncoef, Ncoef0;
	int Numfac;
	int Numfac1;
	int Lmfit, Lmfit1;
	int Dg_block;
	int lastone;
	int lastma;
	int Ndata;
	double Conw_r;
	double Phi_0;
	double Alamda_incr;
	double logCl;
	int ia[MAX_N_PAR + 1];
	//int* ia;
	double par[4];
	//double* lambda_pole;
	//double* beta_pole;
	////double cgFirst[MAX_N_PAR + 1];
	//double* cgFirst;
	double tim[MAX_N_OBS + 1];
	double ee[MAX_N_OBS + 1][3];
	double ee0[MAX_N_OBS + 1][3];
	double Darea[MAX_N_FAC + 1];
	double Nor[MAX_N_FAC + 1][3];
	double Fc[MAX_N_FAC + 1][MAX_LM + 1];
	double Fs[MAX_N_FAC + 1][MAX_LM + 1];
	double Dsph[MAX_N_FAC + 1][MAX_N_PAR + 1];
	double Pleg[MAX_N_FAC + 1][MAX_LM + 1][MAX_LM + 1];
	double cg[MAX_N_PAR + 1];
	double Sig[MAX_N_OBS + 1];
} varholder;
