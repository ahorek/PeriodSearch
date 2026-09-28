/* Real.cl - the floating-point type of the model computation.

   Every double-precision quantity of the kernels is a `real`, and every
   operation on one is written with the R_* macros below.

   Default build (the device supports cl_khr_fp64):
     real == double and each macro expands textually to the plain C
     expression it replaces (R_ADD(a, b) -> ((a) + (b)), R_DIV -> ddiv,
     R_EXP -> exp, ...), so the compiler sees exactly the code it always
     compiled.

   -D PS_FP32 (set by the host when the device has no FP64 support):
     real holds the IEEE double's bit pattern and every macro calls the
     software binary64 implementation in SoftFP64.cl: correctly rounded
     add/sub/mul/fma/div/sqrt and bit-exact ports of the AMD device-library
     exp/log/acos/sincos/fmod. The results are identical, bit for bit, to
     the FP64 build on AMD hardware. A real still occupies 8 bytes with
     8-byte alignment - it IS the double - so structs, buffers and the host
     side are unchanged. `real` is a struct on purpose: a plain operator or
     literal applied to it is a compile error instead of a silent
     integer/float operation.

   Fused multiply-add: under FP_CONTRACT ON clang fuses an add/sub with a
   multiply that is its direct operand (left operand checked first) into
   one fma - a single rounding. Those spots are written with the explicit
   R_MADD/R_ADDM/R_MSUB/R_SUBM/R_ADDTOM/R_SUBFROMM forms: in the FP64 build
   they are the same expressions (so the compiler fuses them as before),
   in the PS_FP32 build they are software fma calls. Keep them in sync
   when editing: an R_ADD with an R_MUL operand would be fused by the FP64
   compiler but rounded twice by the emulation.

   Rules for kernel code:
     - arithmetic, comparisons and math functions on reals only through the
       R_* macros;
     - literals through R_C() (a float-exact floating literal, e.g.
       R_C(1.0), R_C(90.0)); other constants have named macros (R_PI,
       R_DEG2RAD, ...);
     - double kernel arguments are real_arg, read with R_ARG(). */

#ifndef PS_FP32

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

typedef double real;
typedef double real_arg;

#define R_ARG(x)            (x)
#define R_C(x)              (x)
#define R_FROM_INT(i)       ((double)(i))

#define R_ADD(a, b)         ((a) + (b))
#define R_SUB(a, b)         ((a) - (b))
#define R_MUL(a, b)         ((a) * (b))
#define R_DIV(a, b)         ddiv((a), (b))
#define R_NEG(a)            (-(a))
#define R_ADDTO(a, b)       ((a) += (b))
#define R_SUBFROM(a, b)     ((a) -= (b))

/* the contracted (fma) forms, see above */
#define R_MADD(a, b, c)     (((a) * (b)) + (c))
#define R_ADDM(c, a, b)     ((c) + ((a) * (b)))
#define R_MSUB(a, b, c)     (((a) * (b)) - (c))
#define R_SUBM(c, a, b)     ((c) - ((a) * (b)))
#define R_ADDTOM(x, a, b)   ((x) += ((a) * (b)))
#define R_SUBFROMM(x, a, b) ((x) -= ((a) * (b)))

#define R_LT(a, b)          ((a) < (b))
#define R_GT(a, b)          ((a) > (b))
#define R_LE(a, b)          ((a) <= (b))
#define R_GE(a, b)          ((a) >= (b))
#define R_EQ(a, b)          ((a) == (b))
#define R_NE(a, b)          ((a) != (b))

#define R_FABS(x)           fabs(x)
#define R_SQRT(x)           sqrt(x)
#define R_EXP(x)            exp(x)
#define R_LOG(x)            log(x)
#define R_ACOS(x)           acos(x)
#define R_SINCOS(x, pc)     sincos((x), (pc))
#define R_FMOD_2PI(x)       fmod((x), (2 * PI))
#define R_CLAMP(x, lo, hi)  clamp((x), (lo), (hi))
#define R_POW2(x)           pow((x), 2.0)
#define R_ISNAN(x)          isnan(x)

#define R_PI                PI
#define R_2PI               (2 * PI)
#define R_48PI              (24.0 * 2.0 * PI)
#define R_DEG2RAD           DEG2RAD
#define R_RAD2DEG           RAD2DEG
#define R_TINY              TINY
#define R_1E_10             1e-10
#define R_1E30              1e30
#define R_1E40              1e40

#else /* PS_FP32 */

typedef struct { ulong u; } real;
typedef ulong real_arg;

real sf_r(ulong u)
{
	real r;
	r.u = u;
	return r;
}

/* sincos with the OpenCL signature: returns sin, stores cos */
real sf_sincos_r(real x, real* c)
{
	ulong cu;
	ulong s = sf_ocml_sincos(x.u, &cu);
	c->u = cu;
	return sf_r(s);
}

#define R_ARG(v)            sf_r(v)
#define R_C_(x)             sf_r(sf_from_f32(x##f))
#define R_C(x)              R_C_(x)
#define R_FROM_INT(i)       sf_r(sf_from_i32(i))

#define R_ADD(a, b)         sf_r(sf_add((a).u, (b).u))
#define R_SUB(a, b)         sf_r(sf_sub((a).u, (b).u))
#define R_MUL(a, b)         sf_r(sf_mul((a).u, (b).u))
#define R_DIV(a, b)         sf_r(sf_div((a).u, (b).u))
#define R_NEG(a)            sf_r(sf_neg((a).u))
#define R_ADDTO(a, b)       ((a) = R_ADD((a), (b)))
#define R_SUBFROM(a, b)     ((a) = R_SUB((a), (b)))

#define R_MADD(a, b, c)     sf_r(sf_fma((a).u, (b).u, (c).u))
#define R_ADDM(c, a, b)     sf_r(sf_fma((a).u, (b).u, (c).u))
#define R_MSUB(a, b, c)     sf_r(sf_fma((a).u, (b).u, sf_neg((c).u)))
#define R_SUBM(c, a, b)     sf_r(sf_fma(sf_neg((a).u), (b).u, (c).u))
#define R_ADDTOM(x, a, b)   ((x) = R_ADDM((x), (a), (b)))
#define R_SUBFROMM(x, a, b) ((x) = R_SUBM((x), (a), (b)))

#define R_LT(a, b)          sf_lt((a).u, (b).u)
#define R_GT(a, b)          sf_gt((a).u, (b).u)
#define R_LE(a, b)          sf_le((a).u, (b).u)
#define R_GE(a, b)          sf_ge((a).u, (b).u)
#define R_EQ(a, b)          sf_eq((a).u, (b).u)
#define R_NE(a, b)          sf_ne((a).u, (b).u)

#define R_FABS(x)           sf_r(sf_abs((x).u))
#define R_SQRT(x)           sf_r(sf_sqrt((x).u))
#define R_EXP(x)            sf_r(sf_ocml_exp((x).u))
#define R_LOG(x)            sf_r(sf_ocml_log((x).u))
#define R_ACOS(x)           sf_r(sf_ocml_acos((x).u))
#define R_SINCOS(x, pc)     sf_sincos_r((x), (pc))
#define R_FMOD_2PI(x)       sf_r(sf_ocml_fmod((x).u, 0x401921FB54442D18UL))
#define R_CLAMP(x, lo, hi)  sf_r(sf_minnum(sf_maxnum((x).u, (lo).u), (hi).u))
#define R_POW2(x)           sf_r(sf_mul((x).u, (x).u))	/* the compiler turns pow(x, 2.0) into x * x */
#define R_ISNAN(x)          sf_isnan((x).u)

/* the bit patterns of the FP64 build's (constant-folded) values */
#define R_PI                sf_r(0x400921FB54442D18UL)
#define R_2PI               sf_r(0x401921FB54442D18UL)
#define R_48PI              sf_r(0x4062D97C7F3321D2UL)
#define R_DEG2RAD           sf_r(0x3F91DF46A2529D39UL)
#define R_RAD2DEG           sf_r(0x404CA5DC1A63C1F8UL)
#define R_TINY              sf_r(0x3E45798EE2308C3AUL)
#define R_1E_10             sf_r(0x3DDB7CDFD9D7BDBBUL)
#define R_1E30              sf_r(0x46293E5939A08CEAUL)
#define R_1E40              sf_r(0x483D6329F1C35CA5UL)

#endif /* PS_FP32 */
