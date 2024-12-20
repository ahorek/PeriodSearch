#pragma once

#if defined(__GNUC__) || defined(__clang__)
#define OBSOLETE __attribute__((deprecated))
#elif defined(_MSC_VER)
#define OBSOLETE __declspec(deprecated)
#else
#define OBSOLETE
#endif
