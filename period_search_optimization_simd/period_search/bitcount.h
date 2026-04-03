#ifdef _MSC_VER
    #include <intrin.h>
    #pragma intrinsic(_BitScanForward)
    static inline int ctz(unsigned int mask) {
        unsigned long idx;
        _BitScanForward(&idx, mask);
        return (int)idx;
    }
#else
    #define ctz(mask) __builtin_ctz(mask)
#endif