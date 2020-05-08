/* N.B. The foll. L-M routines are modified versions of Press et al.
   converted from Mikko's fortran code

   8.11.2006
*/

//#include <cuda.h>
//#include "globals_CUDA.h"
//#include "declarations_CUDA.h"

void mrqmin_1_end(
    __global struct freq_context2* CUDA_LCC,
    __global varholder* Fa)
{
    int j, k, l, err_code;
    int3 threadIdx;
    threadIdx.x = get_local_id(0);
    //precalc thread boundaries
    int tmph, tmpl;
    tmph = Fa->ma / BLOCK_DIM;
    if (Fa->ma % BLOCK_DIM) tmph++;
    tmpl = threadIdx.x * tmph;
    tmph = tmpl + tmph;
    if (tmph > Fa->ma) tmph = Fa->ma;
    tmpl++;
    //
    int brtmph, brtmpl;
    brtmph = Fa->Lmfit / BLOCK_DIM;
    if (Fa->Lmfit % BLOCK_DIM) brtmph++;
    brtmpl = threadIdx.x * brtmph;
    brtmph = brtmpl + brtmph;
    if (brtmph > Fa->Lmfit) brtmph = Fa->Lmfit;
    brtmpl++;

    if ((*CUDA_LCC).isAlamda)
    {
        for (j = tmpl; j <= tmph; j++)
        {
            (*CUDA_LCC).atry[j] = (*CUDA_LCC).cg[j];
        }
        
        //__syncthreads();
        barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
    }

    for (j = brtmpl; j <= brtmph; j++)
    {
        int ixx = j * Fa->Lmfit1 + 1;
        for (k = 1; k <= Fa->Lmfit; k++, ixx++)
        {
            *(*CUDA_LCC).covar[ixx] = (*CUDA_LCC).alpha[ixx];
        }

        int idx = j * Fa->Lmfit1 + j;
        *(*CUDA_LCC).covar[idx] = (*CUDA_LCC).alpha[idx] * (1 + (*CUDA_LCC).Alamda);
        (*CUDA_LCC).da[j] = (*CUDA_LCC).beta[j];
    }

    //__syncthreads();
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);

    // TODO: Fix this ->
    err_code = gauss_errc(CUDA_LCC, Fa);

    //     __syncthreads(); inside gauss

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

//void mrqmin_2_end(freq_context* CUDA_LCC, int ia[], int ma)
//{
//    int j, k, l;
//
//    if ((*CUDA_LCC).Chisq < (*CUDA_LCC).Ochisq)
//    {
//        (*CUDA_LCC).Alamda = (*CUDA_LCC).Alamda / CUDA_Alamda_incr;
//        for (j = 1; j <= CUDA_mfit; j++)
//        {
//            for (k = 1; k <= CUDA_mfit; k++)
//                (*CUDA_LCC).alpha[j * CUDA_mfit1 + k] = (*CUDA_LCC).covar[j * CUDA_mfit1 + k];
//            (*CUDA_LCC).beta[j] = (*CUDA_LCC).da[j];
//        }
//        for (l = 1; l <= ma; l++)
//            (*CUDA_LCC).cg[l] = (*CUDA_LCC).atry[l];
//    }
//    else
//    {
//        (*CUDA_LCC).Alamda = CUDA_Alamda_incr * (*CUDA_LCC).Alamda;
//        (*CUDA_LCC).Chisq = (*CUDA_LCC).Ochisq;
//    }
//
//    return;
//}

