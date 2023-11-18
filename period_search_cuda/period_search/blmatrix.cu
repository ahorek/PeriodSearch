/* beta, lambda rotation matrix and its derivatives

   8.11.2006
*/

#include <math.h>
#include "globals_CUDA.h"

__device__ void __forceinline__ blmatrix(double bet, double lam, int tid)
{
   double cb, sb, cl, sl, cbcl, cbsl, sbcl, sbsl, nsb, ncb, nsl, ncl;
   //__builtin_assume(bet > (-2.0 * PI) && bet < (2.0 * PI));
   sincos(bet, &sb, &cb);

   //__builtin_assume(lam > (-2.0 * PI) && lam < (2.0 * PI));
   sincos(lam, &sl, &cl);

   nsb = -sb;
   ncb = -cb;
   cbcl = cb * cl;
   cbsl = cb * sl;
   nsl = -sl;

   Blmat[1][1][tid] = cl;
   Blmat[2][2][tid] = cb;
   Blmat[0][0][tid] = cbcl;
   Dblm[0][2][0][tid] = cbcl;
   Dblm[1][0][1][tid] = cbcl;
   Blmat[0][1][tid] = cbsl;
   Dblm[0][2][1][tid] = cbsl;
   Dblm[1][0][0][tid] = -cbsl;
   Blmat[0][2][tid] = nsb;
   Dblm[0][2][2][tid] = nsb;
   Blmat[1][0][tid] = nsl;
   Dblm[1][1][1][tid] = nsl;
   Blmat[1][2][tid] = 0;

   sbcl = sb * cl;
   sbsl = sb * sl;
   ncl = -cl;
   double nsbcl = -sbcl;
   double nsbsl = -sbsl;

   Blmat[2][0][tid] = sbcl;
   Dblm[1][2][1][tid] = sbcl;
   Dblm[0][0][0][tid] = nsbcl;
   Blmat[2][1][tid] = sbsl;
   Dblm[0][0][1][tid] = nsbsl;
   Dblm[1][2][0][tid] = nsbsl;
   Dblm[1][1][0][tid] = ncl;
   Dblm[0][0][2][tid] = ncb;

   // Ders. of Blmat w.r.t. bet
   Dblm[0][1][0][tid] = 0;
   Dblm[0][1][1][tid] = 0;
   Dblm[0][1][2][tid] = 0;

   // Ders. w.r.t. lam
   Dblm[1][0][2][tid] = 0;
   Dblm[1][1][2][tid] = 0;
   Dblm[1][2][2][tid] = 0;
}
