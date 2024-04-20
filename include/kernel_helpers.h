#pragma once

#include <curand_kernel.h>

__declspec(dllexport) void dumpFbToPng(float *fb, int fbWidth, int fbHeight);

__declspec(dllexport) void dumpAccumFbToOpenExr(float *hostAccumFb, int fbWidth, int fbHeight);

__declspec(dllexport) void initRandState(int maxX, int maxY, curandState *randState);
