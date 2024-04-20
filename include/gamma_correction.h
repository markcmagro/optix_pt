#pragma once

inline __host__ __device__ float4 linearToSrgb(const float4 &c, float gammaTerm)
{
    float kInvGamma = 1.0f / gammaTerm;
    return make_float4(powf(c.x, kInvGamma), powf(c.y, kInvGamma), powf(c.z, kInvGamma), c.w);
}
