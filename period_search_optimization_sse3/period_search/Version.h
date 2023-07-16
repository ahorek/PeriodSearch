#pragma once
#include <Windows.h>

bool GetVersionInfo(
    std::string filename,
    int& major,
    int& minor,
    int& build,
    int& revision);
