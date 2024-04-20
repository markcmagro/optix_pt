#pragma once

struct BsdfSample
{
    float3 newRayDir;
    float3 bsdf;
    float pdf;
};
