#pragma once

#include <sutil/vec_math.h>
#include "material_flags.h"
#include "schlick.h"

inline __device__ float calcF0(const float ior)
{
    float tmp = (1.0f - ior) / (1.0f + ior);
    return tmp * tmp;
}

inline __device__ float3 calcF0(const float ior, const float3 f0Colour)
{
    float tmp = (1.0f - ior) / (1.0f + ior);
    return tmp * tmp * f0Colour;
}

inline __device__ float calcIorFromSpecular(float specular)
{
    return 2.0f / (1.0f - sqrtf(specular * 0.08)) - 1.0f;
}

inline __device__ float3 conductorFresnel(const float3 f0, float3 bsdf, float vDotH)
{
    return bsdf * fresnelSchlick(f0, vDotH);

    //cuComplex eta = make_cuComplex(1.5f, 0.0f);
    //float3 F = f0 + (1.0f - f0) * fresnelMetal(vDotH, eta);
    //return bsdf * F;
}

inline __device__ float3 fresnelMix(const float ior, const float vDotH, const float3 base, const float3 layer)
{
    float f0 = calcF0(ior);
    //float F = fresnelSchlick(f0, vDotH);
    float F = f0 + (1.0f - f0) * fresnelDielectric(vDotH, ior);
    return lerp(base, layer, F);
}

// KHR_materials_specular
inline __device__ float3 fresnelMixArtistic(const float ior, const float3 f0Colour, float weight, const float vDotH,
    const float3 base, const float3 layer)
{
    // Note: This may break energy conservation.

    float3 f0 = calcF0(ior, f0Colour);
    f0 = fminf(f0, make_float3(1.0f));
    //float3 F = fresnelSchlick(f0, vDotH);
    float3 F = f0 + (1.0f - f0) * fresnelDielectric(vDotH, ior);
    return (1.0f - weight * maxValue(F)) * base + weight * F * layer;
}

inline __device__ float heaviside(float x)
{
    return x > 0.0f ? 1.0f : 0.0f;
}

inline __device__ void printMaterialFlags(unsigned int flags)
{
    printf("materialFlags=0x%08X\n", flags);
    if (flags & MaterialFlags::Metal)
        printf("  Metal\n");
    if (flags & MaterialFlags::Dielectric)
        printf("  Dielectric\n");
    if (flags & MaterialFlags::Emission)
        printf("  Emission\n");
    if (flags & MaterialFlags::Opaque)
        printf("  Opaque\n");
    if (flags & MaterialFlags::Transmission)
        printf("  Transmission\n");
    if (flags & MaterialFlags::Reflection)
        printf("  Reflection\n");
    if (flags & MaterialFlags::Diffuse)
        printf("  Diffuse\n");
    if (flags & MaterialFlags::Subsurface)
        printf("  Subsurface\n");
    if (flags & MaterialFlags::Coat)
        printf("  Coat\n");
    if (flags & MaterialFlags::Sheen)
        printf("  Sheen\n");
    if (flags & MaterialFlags::Lambertian)
        printf("  Lambertian\n");
    if (flags & MaterialFlags::Specular)
        printf("  Specular\n");
}
