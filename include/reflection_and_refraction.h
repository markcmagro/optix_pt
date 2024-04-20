#pragma once

#include <sutil/vec_math.h>

inline __host__ __device__ float3 reflectRay(float3 n, float3 i)
{
    // n is the unit normal
    // i is the unit incident ray, pointing *away* from the surface

    // Interestingly, if the normal is flipped the same result is obtained:
    // dot(-n, i) * -n = -dot(n, i) * -n = dot(n, i) * n

    return -i + 2.0f * dot(n, i) * n;
}

inline __host__ __device__ float3 refractRay(float3 n, float3 i, float eta)
{
    // n is the surface normal, always pointing out of the surface
    // i is the incident ray
    // eta is eta_t / eta_i where:
    // - eta_t is the ior of the target medium
    // - eta_i is the ior of the source medium

    // Assumptions
    // 1. n and i are normalized
    // 2. i is pointing *away* from the surface

    // If total internal reflection occurs, the reflected ray is returned.

    float cosThetaI = dot(n, i);
    if (cosThetaI < 0.0f) { // exiting material
        eta = 1.0f / eta;
        cosThetaI = -cosThetaI;
        n = -n;
    }

    float sin2ThetaI = fmax(0.0f, 1.0f - (cosThetaI * cosThetaI));
    float sin2ThetaT = sin2ThetaI / (eta * eta);

    if (sin2ThetaT >= 1.0f) // total internal reflection
        return reflectRay(n, i);

    // Sanitise the argument to sqrt because it may be negative due to
    // rounding errors.
    float cosThetaT = sqrtf(fmax(0.0f, 1.0f - sin2ThetaT));

    return -i / eta + (cosThetaI / eta - cosThetaT) * n;
}

inline __host__ __device__ float3 transmissionHalfVectorThinWalled(const float3 &n, const float3 &l, const float3 &v)
{
    // n is the surface normal, always pointing out of the material.
    // l is the incident ray that keeps going straight through or out of the
    //   material (no refraction) but pointing opposite the direction of travel
    //   (it points to the light)
    // v is the outgoing ray, pointing in the actual direction light travels.

    // Let l' be the reflection of l on the opposite hemisphere (where v is).
    // The calculation is the same when entering or exiting the material.
    // Entering the material:
    // l' = l - 2(l.n)n = l - 2(n.l)n
    // Exiting the material:
    // l' = l + 2(-l.n)n = l - 2(l.n)n = l - 2(n.l)n

    // Calculate l' and the half vector (normalize(v + l')).
    return normalize(v + l - 2.0f * dot(n, l) * n);
}

inline __host__ __device__ float3 transmissionHalfVectorVolumetric(const float3 &n, const float3 &l, const float3 &v,
    float etaI, float etaV)
{
    float3 nTemp = n;

    if (dot(nTemp, v) < 0.0f) { // inside
        float etaTemp = etaI;
        etaI = etaV;
        etaV = etaTemp;
        nTemp = -n;

        //if (debug) {
        //    printf("Inside\n");
        //    printf("n=(%f, %f, %f)\n", n.x, n.y, n.z);
        //    printf("v=(%f, %f, %f)\n", v.x, v.y, v.z);
        //    printf("l=(%f, %f, %f)\n", l.x, l.y, l.z);
        //}
    }
    else {
        //if (debug) {
        //    printf("Outside\n");
        //    printf("n=(%f, %f, %f)\n", n.x, n.y, n.z);
        //    printf("v=(%f, %f, %f)\n", v.x, v.y, v.z);
        //    printf("l=(%f, %f, %f)\n", l.x, l.y, l.z);
        //}
    }

    return normalize(-etaI * v -etaV * l);
}
