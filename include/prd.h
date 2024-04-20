#pragma once

#include <sutil/vec_math.h>
#include "random_number_generators.h"

struct PRD
{
    PRNG random;
    float3 radiance;
    float3 throughput;
    float3 origin; // ray origin
    float3 dir;    // ray direction

    int depth;
    bool done;
    bool specularHit;

    bool debug;
    bool error;
};

struct ShadowRayData
{
    float3 lightVisibility;
    float xi;
};
