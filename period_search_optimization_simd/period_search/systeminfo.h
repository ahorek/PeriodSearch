#pragma once

#include <string>

#if defined(__arm__) || defined(__aarch64__) || defined(_M_ARM) || defined(_M_ARM64)
void getSystemInfo();
void getCpuInfoByArch(std::ifstream &cpuinfo);
#endif
#ifdef _WIN32
std::string getTotalSystemMemory();
#else
float getTotalSystemMemory();
#endif
