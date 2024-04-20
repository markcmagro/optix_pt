#pragma once

#include <cuComplex.h>
#include <sutil/vec_math.h>
#include "cuda_math.h"

inline __device__ float fresnelDielectric(float cosThetaI, float eta)
{
    if (cosThetaI < 0.0f) { // exiting material
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
    }

    // Compute cosThetaT using Snell's law.
    float sin2ThetaI = 1.0f - (cosThetaI * cosThetaI);
    float sin2ThetaT = sin2ThetaI / (eta * eta);

    // Total internal reflection.
    if (sin2ThetaT >= 1.0f)
        return 1.0f;

    float cosThetaT = fmax(0.0f, sqrtf(1.0f - sin2ThetaT));

    float para = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    float perp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);

    return (para * para + perp * perp) / 2.0f;
}

inline __host__ __device__ float cuCnorm(cuComplex x)
{
    float a = cuCrealf(x);
    float b = cuCimagf(x);
    return sqrtf(a * a + b * b);
}

inline __device__ float fresnelMetal(float cosThetaI, cuComplex eta)
{
    // Compute cosThetaT using Snell's law.
    float sin2ThetaI = 1.0f - (cosThetaI * cosThetaI);
    cuComplex sin2ThetaT = sin2ThetaI / (eta * eta);
    auto subtraction = cuComplexFloatToDouble(1.0f - sin2ThetaT);
    cuComplex cosThetaT = cuComplexDoubleToFloat(cuCsqrt(subtraction));

    cuComplex para = (eta * cosThetaI - cosThetaT) / (eta * cosThetaI + cosThetaT);
    cuComplex perp = (cosThetaI - eta * cosThetaT) / (cosThetaI + eta * cosThetaT);

    return (cuCnorm(para) + cuCnorm(perp)) / 2.0f;
}
