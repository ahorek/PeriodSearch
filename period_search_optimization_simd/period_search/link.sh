#!/bin/bash

#g++ -o "/home/paganini/source/pssimd2_Linux/bin/x64/Debug/pssimd2_Linux.out" -Wl,--no-undefined -Wl,-rpath-link=/home/paganini/source/boinc_src/api -Wl,-rpath-link=/home/paganini/source/boinc_src/lib -Wl,-L/home/paganini/source/boinc_src/api -Wl,-L/home/paganini/source/boinc_src/lib -Wl,-z,relro -Wl,-z,now -Wl,-z,noexecstack /home/paganini/source/pssimd2_Linux/obj/x64/Debug/areanorm.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/arrayHelpers.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/blmatrix.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/bright.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/bright_avx.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/bright_avx512.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/bright_fma.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/bright_sse2.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/bright_sse3.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/conv.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/conv_avx.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/conv_avx512.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/conv_fma.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/conv_sse2.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/conv_sse3.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/CpuInfo.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/curv.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/curv_avx.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/curv_avx512.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/curv_fma.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/curv_sse2.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/curv_sse3.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/dot_product.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/ellfit.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/EnumHelpers.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/gauss_errc.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/gauss_errc_avx.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/gauss_errc_avx512.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/gauss_errc_fma.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/gauss_errc_sse2.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/gauss_errc_sse3.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/LcHelpers.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/lubksb.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/ludcmp.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/matrix.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/mrqcof.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/mrqcof_avx.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/mrqcof_avx512.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/mrqcof_fma.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/mrqcof_sse2.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/mrqcof_sse3.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/mrqmin.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/pch.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/period_search_BOINC.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/phasec.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/sphfunc.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/systeminfo.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/trifac.o /home/paganini/source/pssimd2_Linux/obj/x64/Debug/VersionInfo.o -lboinc -lboinc_api

# cd /home/paganini/source/pssimd2_Linux/; \
# g++ -march=x86-64 -std=c++2a -g -D_DEBUG -o "/home/paganini/source/pssimd2_Linux/bin/x64/Debug/pssimd2_Linux.out" \
# -Wl,--no-undefined \
# -Wl,-rpath-link=/home/paganini/source/boinc_src/api \
# -Wl,-rpath-link=/home/paganini/source/boinc_src/lib \
# -Wl,-L/home/paganini/source/boinc_src/api \
# -Wl,-L/home/paganini/source/boinc_src/lib \
# -Wl,-L../../period_search_common/build \
# -Wl,-L/usr/X11R6/lib \
# -Wl,-z,relro \
# -Wl,-z,now \
# -Wl,-z,noexecstack \
# /home/paganini/source/pssimd2_Linux/obj/x64/Debug/*.o \
# -lps_commond \
# -lboinc \
# -lboinc_api \
# -lstdc++ \
# -pthread \
# -lm

# cd /home/paganini/source/pssimd2_Linux/; \
g++ -march=x86-64 -std=c++2a -g -D_DEBUG -o "/home/paganini/source/pssimd2_Linux/bin/x64/Debug/pssimd2_Linux.out" \
-Wl,--no-undefined \
-Wl,-I../../boinc_src \
-Wl,-I../../boinc_src/lib \
-Wl,-I../../boinc_src/api \
-Wl,-L/home/paganini/source/boinc_src/api \
-Wl,-L/home/paganini/source/boinc_src/lib \
-Wl,-L/usr/X11R6/lib \
-Wl,-z,relro \
-Wl,-z,now \
-Wl,-z,noexecstack \
/home/paganini/source/pssimd2_Linux/period_search/*.o \
-lstdc++ -pthread \
../../boinc_src/api/libboinc_api.a \
../../boinc_src/lib/libboinc.a \
-lm

# -Wl,-rpath-link=/home/paganini/source/boinc_src/api \
# -Wl,-rpath-link=/home/paganini/source/boinc_src/lib \
# -Wl,-z,relro \
# -Wl,-z,now \
# -Wl,-z,noexecstack \

# -Wl,-L../../period_search_common/build \
# -lps_commond \
# /home/paganini/source/pssimd2_Linux/obj/x64/Debug/*.o \
# -Wl,--no-undefined \

# g++ -march=x86-64 -std=c++2a -g -o -D_DEBUG -I../../boinc_src -I../../boinc_src/lib -I../../boinc_src/api -L /usr/X11R6/lib -L. -o period_search_BOINC_linux_10222_x64_universal_Debug dot_product.o areanorm.o blmatrix.o bright.o bright_avx.o bright_avx512.o bright_fma.o bright_sse2.o bright_sse3.o conv.o conv_avx.o conv_avx512.o conv_fma.o conv_sse2.o conv_sse3.o CpuInfo.o curv.o curv_sse3.o curv_avx.o curv_fma.o curv_avx512.o curv_sse2.o ellfit.o EnumHelpers.o gauss_errc.o gauss_errc_avx.o gauss_errc_fma.o gauss_errc_avx512.o gauss_errc_sse2.o gauss_errc_sse3.o lubksb.o ludcmp.o matrix.o mrqcof.o mrqcof_avx.o mrqcof_avx512.o mrqcof_fma.o mrqcof_sse2.o mrqcof_sse3.o mrqmin.o phasec.o sphfunc.o systeminfo.o trifac.o VersionInfo.o LcHelpers.o period_search_BOINC.o libstdc++.a -pthread \
# ../../boinc_src/api/libboinc_api.a \
# ../../boinc_src/lib/libboinc.a \
# -lm

# cd /home/paganini/source/pssimd2_Linux/
# ln -s `g++ -print-file-name=libstdc++.a`
