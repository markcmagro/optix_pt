# pragma once

#include <sutil/vec_math.h>

// Calculate the percentage of light reflected.
inline __device__ float fresnelPowerTerm(float vDotH)
{
    return powf(1.0f - fabs(vDotH), 5.0f);
}

// The glTF 2.0 specs recommend using Schlick's Fresnel approximation for metals
// and the full Fresnel equations for dielectrics.
inline __device__ float fresnelSchlick(const float f0, float vDotH)
{
    return f0 + (1.0f - f0) * fresnelPowerTerm(vDotH);
}

// The glTF 2.0 specs recommend using Schlick's Fresnel approximation for metals
// and the full Fresnel equations for dielectrics.
inline __device__ float3 fresnelSchlick(const float3 f0, float vDotH)
{
    return f0 + (1.0f - f0) * fresnelPowerTerm(vDotH);
}

inline __device__ float schlickG(float cosTheta, float alpha)
{
    float a = alpha * 0.5f;
    return cosTheta / (cosTheta * (1.0f - a) + a);
}
