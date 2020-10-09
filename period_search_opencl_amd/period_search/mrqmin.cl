/* N.B. The foll. L-M routines are modified versions of Press et al.
   converted from Mikko's fortran code

   8.11.2006
*/

void mrqmin_1_end(__global struct freq_context2* CUDA_LCC,
    __global varholder* Fa,
    int isAlamda, 
    int3 threadIdx, 
    int3 blockIdx) 
{
    int ma = Fa->ma;
    int j;
    //precalc thread boundaries
    int tmph, tmpl;
    tmph = ma / BLOCK_DIM;
    if (ma % BLOCK_DIM) tmph++;
    tmpl = threadIdx.x * tmph;
    tmph = tmpl + tmph;
    if (tmph > ma) tmph = ma;
    //if (blockIdx.x == 0)
    //	printf("[%d][%d] tmph: %d\n", blockIdx.x, threadIdx.x, tmph);
    tmpl++;
    //
    int mfit = Fa->Lmfit;
    int mfit1 = Fa->Lmfit1;

    int brtmph, brtmpl;
    brtmph = mfit / BLOCK_DIM;
    if (mfit % BLOCK_DIM) brtmph++;
    brtmpl = threadIdx.x * brtmph;
    brtmph = brtmpl + brtmph;
    if (brtmph > mfit) brtmph = mfit;
    brtmpl++;

    //if (threadIdx.x == 9)
    //	printf("[%d][%d] isAlamda: %d, tmpl: %d, tmph: %d, brtmpl: %d, brtmph: %d\n", blockIdx.x, threadIdx.x, isAlamda, tmpl, tmph, brtmpl, brtmph);

    if (isAlamda)
    {
        for (j = tmpl; j <= tmph; j++)
        {
            (*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];

            //if (blockIdx.x == 2)
            //	printf("[%d][%d] atry[%d]: % .9f\n", blockIdx.x, threadIdx.x, j, (*CUDA_LCC).atry[j]);
        }

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    for (j = brtmpl; j <= brtmph; j++)
    {
        int ixx = j * mfit1 + 1;
        for (int k = 1; k <= mfit; k++, ixx++)
        {
            (*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];

            //if (blockIdx.x == 2 && threadIdx.x == 5)
            //	printf("[%d][%d] covar[%d]: % .9f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).covar[ixx]);
        }

        (*CUDA_LCC).covar[j * mfit1 + j] = (*CUDA_LCC).alpha[j * mfit1 + j] * (1 + (*CUDA_LCC).Alamda);
        (*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];
    }
    
    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    if (threadIdx.x == 0)
    {
        for (j = 1; j <= Fa->Lmfit; j++)
        {
            (*CUDA_LCC).ipiv[j] = 0;
        }
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //int err_code = 
    gauss_errc(CUDA_LCC, Fa, brtmpl, brtmph, threadIdx, blockIdx);
    
    //if (err_code) return;
    //err_code = gauss_errc(CUDA_LCC, CUDA_mfit, (*CUDA_LCC).da);

    //     __syncthreads(); inside gauss

    if (threadIdx.x == 0)
    {

        //		if (err_code != 0) return(err_code); bacha na sync threads

        j = 0;
        for (int l = 1; l <= ma; l++)
            if (Fa->ia[l])
            {
                j++;
                (*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
            }
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
}

void mrqmin_1_end_old(__global struct freq_context2* CUDA_LCC,
    __global varholder* Fa,
    int isAlamda)
{
    int3 blockIdx, threadIdx;
    threadIdx.x = get_local_id(0);
    blockIdx.x = get_group_id(0);

    __private int i, j, k, l, ll, err_code;

    /*precalc thread boundaries*/
    __private int tmph, tmpl;
    tmph = Fa->ma / BLOCK_DIM;
    if (Fa->ma % BLOCK_DIM) tmph++;     

    tmpl = threadIdx.x * tmph;
    tmph = tmpl + tmph;
    if (tmph > Fa->ma) tmph = Fa->ma;
    tmpl++;
    
    int brtmph, brtmpl;
    brtmph = Fa->Lmfit / BLOCK_DIM;
    if (Fa->Lmfit % BLOCK_DIM) brtmph++;

    brtmpl = threadIdx.x * brtmph;
    brtmph = brtmpl + brtmph;
    if (brtmph > Fa->Lmfit) brtmph = Fa->Lmfit;
    brtmpl++;

    //if(threadIdx.x == 9)
    //    printf("[%d][%d] isAlamda: %d, tmpl: %d, tmph: %d, brtmpl: %d, brtmph: %d\n", blockIdx.x, threadIdx.x, isAlamda, tmpl, tmph, brtmpl, brtmph);

    if (isAlamda)
    {
        for (j = tmpl; j <= tmph; j++)
        {
            (*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];

            //if (blockIdx.x == 2)
             //  printf("[%d][%d] atry[%d]: % .9f\n", blockIdx.x, threadIdx.x, j, (*CUDA_LCC).atry[j]);
        }
        
        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    for (j = brtmpl; j <= brtmph; j++)
    {
        int ixx = j * Fa->Lmfit1 + 1;
        for (k = 1; k <= Fa->Lmfit; k++, ixx++)
        {
            (*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];

            //if (blockIdx.x == 2 && threadIdx.x == 5)
            //  printf("[%d][%d] covar[%d]: % .9f\n", blockIdx.x, threadIdx.x, ixx, (*CUDA_LCC).covar[ixx]);
        }

        int idx = j * Fa->Lmfit1 + j;
        (*CUDA_LCC).covar[idx] = (*CUDA_LCC).alpha[idx] * (1 + (*CUDA_LCC).Alamda);
        (*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    //<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    if (threadIdx.x == 0)
    {
        for (j = 1; j <= Fa->Lmfit; j++)
        {
            (*CUDA_LCC).ipiv[j] = 0;

            //if (blockIdx.x == 0)
            //    printf("[%d][%d] ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, j, (*CUDA_LCC).ipiv[j]);
        }
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //if (blockIdx.x == 2)
    //    printf("[%d][%d] brtmpl: %d, brtmph: %d, Lmfit: %d\n", blockIdx.x, threadIdx.x, brtmpl, brtmph, Fa->Lmfit);

    double big;
    int irow, licol;
    __local int icol;
    __local double pivinv;
    __local int sh_icol[BLOCK_DIM];
    __local int sh_irow[BLOCK_DIM];
    __local double sh_big[BLOCK_DIM];
    
    for (i = 1; i <= Fa->Lmfit; i++)
    {
        //if(blockIdx.x == 2)
        //    printf("[%d][%d] i: %d\n", blockIdx.x, threadIdx.x, i);
        // TODO: Fix this ->
        //err_code = 
        //gauss_errc(CUDA_LCC, Fa, (*CUDA_LCC).ipiv, brtmpl, brtmph, i); // , i);

        //     __syncthreads(); inside gauss

        big = 0.0f;
        irow = 0;
        licol = 0;
        for (j = brtmpl; j <= brtmph; j++)
        {
            //if (blockIdx.x == 2 && i == 1)
            //    printf("[%d][%d] j[%d], i[%d] brtmpl: %d, brtmph: %d\n", blockIdx.x, threadIdx.x, j, i, brtmpl, brtmph);


            if ((*CUDA_LCC).ipiv[j] == 1) continue;

            //if (blockIdx.x == 2 & i == 1)
            //    printf("[%d][%d] i[%d], j[%d], ipiv[%d]: %d\n", blockIdx.x, threadIdx.x, i, j, j, (*CUDA_LCC).ipiv[j]);
            //int x = blockIdx.x;
            err_code = gauss_errc_begin(CUDA_LCC, Fa, &big, &irow, &licol, brtmpl, brtmph, i, j);
            //if (err_code) goto end;
        }

        sh_big[threadIdx.x] = big;
        sh_irow[threadIdx.x] = irow;
        sh_icol[threadIdx.x] = licol;

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        //if (blockIdx.x == 2 && threadIdx.x == 2)
        //    printf("[%d][%d] i[%d] big: % .9f, irow: %d, icol: %d\n", blockIdx.x, threadIdx.x, i, big, irow, licol);


        if (threadIdx.x == 0)
        {
            big = sh_big[0];
            icol = sh_icol[0];
            irow = sh_irow[0];
            for (j = 1; j < BLOCK_DIM; j++)
            {
                if (sh_big[j] >= big)
                {
                    big = sh_big[j];
                    irow = sh_irow[j];
                    icol = sh_icol[j];
                }
            }

            err_code = gauss_errc_mid(CUDA_LCC, Fa, &big, &irow, &icol, &pivinv, i);
            //if (err_code) goto end;
        }

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        for (l = brtmpl; l <= brtmph; l++)
        {
            (*CUDA_LCC).covar[icol * Fa->Lmfit1 + l] *= pivinv;
        }

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

        int ixx, jxx;
        double dum;

        for (ll = brtmpl; ll <= brtmph; ll++)
        {
            //if (threadIdx.x == 0)
            //	printf("[%d][%d][%d] brtmph: %d, ll: %d, icol: %d\n", blockIdx.x, threadIdx.x, i, brtmph, ll, icol);

            gauss_errc_end(CUDA_LCC, Fa, icol, ll, i);
        }

        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    return;

    if (threadIdx.x == 0)
    {
        for (l = Fa->Lmfit; l >= 1; l--)
        {
            int indxr = (*CUDA_LCC).indxr[l];
            int indxc = (*CUDA_LCC).indxc[l];
            if (indxr != indxc)
            {
                for (k = 1; k <= Fa->Lmfit; k++)
                {
                    int a = k * Fa->Lmfit1 + indxr;
                    int b = k * Fa->Lmfit1 + indxc;

                    swap(&((*CUDA_LCC).covar[a]), &((*CUDA_LCC).covar[b]));
                    //SWAP((*CUDA_LCC).covar[a], (*CUDA_LCC).covar[b]);
                }
            }
        }
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // >>>>>>>>>>>>>>>>>>>>>>>>>>>>
 end:
    if (threadIdx.x == 0)
    {

        //		if (err_code != 0) return(err_code); bacha na sync threads

        j = 0;
        for (l = 1; l <= Fa->ma; l++)
        {
            if (Fa ->ia[l])
            {
                j++;
                (*CUDA_LCC).atry[l] = (*CUDA_LCC).cg[l] + (*CUDA_LCC).da[j];
            }
        }
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    //return(err_code);
}

void mrqmin_2_end(
    __global struct freq_context2* CUDA_LCC, 
    __global varholder* Fa)
{
    int j, k, l;

    if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
    {
        (*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / Fa->Alamda_incr;
        for (j = 1; j <= Fa->Lmfit; j++)
        {
            for (k = 1; k <= Fa->Lmfit; k++)
            {
                (*CUDA_LCC).alpha[j * Fa->Lmfit1 + k] = (*CUDA_LCC).covar[j * Fa->Lmfit1 + k];
            }

            (*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
        }
        for (l = 1; l <= Fa->ma; l++)
        {
            (*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
        }
    }
    else
    {
        (*CUDA_LCC).Alamda = Fa->Alamda_incr * (*CUDA_LCC).Alamda;
        (*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
    }
}

