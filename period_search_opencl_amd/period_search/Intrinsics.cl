/* double-only helpers: the software FP64 build has its own division in
   Real.cl and no double type */
#ifndef PS_SWFP64

/* WORKAROUND(rusticl / aco): runtime f64 '/' returns results with ~3*2^-29
   relative error (verified by [DIVTEST]); fma() and '*' are exact.
   Markstein sequence: two Newton steps refine the reciprocal, the final
   fused correction restores correct rounding.
   NATIVE_DIV_OK=1 (set by the host after the startup probe) replaces the
   helper with plain '/' on drivers whose division is correctly rounded;
   both paths then produce identical bits, so determinism is preserved. */
#ifndef NATIVE_DIV_OK
#define NATIVE_DIV_OK 0
#endif
#if NATIVE_DIV_OK
#define ddiv(a, b) ((a) / (b))
#else
inline double ddiv(double a, double b)
{
    double r = 1.0 / b;
    double e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    e = fma(-b, r, 1.0);
    r = fma(r, e, r);
    double q = a * r;
    return fma(fma(-b, q, a), r, q);
}
#endif

#endif /* !PS_SWFP64 */
