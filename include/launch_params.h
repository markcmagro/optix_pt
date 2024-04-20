#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <optix.h>
#include "light.h"

enum class RayType
{
    Radiance = 0,
    Shadow,
    Count
};

struct LaunchParams
{
    float *frameBuffer;
    float *accumFrameBuffer;
    uint32_t w; // frame buffer width
    uint32_t h; // frame buffer height

    uint32_t sample;
    uint32_t seed;

    // Camera
    float3 eye;
    float3 U;
    float3 V;
    float3 W;

    OptixTraversableHandle traversable;

    Light *lights;
    uint32_t numLights;

    uint32_t numEmissiveTriangles;

    curandState *randState;

    int vis;

    bool toneMapping;
    int toneMappingImpl;

    bool gammaCorrection;
    float gammaTerm;
};
