#pragma once

#include "light.h"

inline __host__ __device__ float3 sampleSphere(const float2 &xi)
{
    // Generate a random uniformly distributed point on the surface of a unit
    // sphere.
    // From http://corysimon.github.io/articles/uniformdistn-on-sphere/.

    float theta = 2.0f * M_PIf * xi.x;
    float phi = acos(1.0f - 2.0f * xi.y);
    float x = sinf(phi) * cosf(theta);
    float y = sinf(phi) * sinf(theta);
    float z = cosf(phi);

    return make_float3(x, y, z);
}

inline __host__ __device__ float3 sampleHemisphere(const float2 &xi)
{
    float3 sample = sampleSphere(xi);
    sample.z = fabs(sample.z);
    return sample;
}

inline __device__ float2 sampleDisk(const float2 &xi)
{
    float r = sqrtf(xi.x);
    float phi = xi.y * 2.0f * M_PIf;
    return make_float2(r * cosf(phi), r * sinf(phi));
}

inline __device__ float3 cosineSampleHemisphere(const float2 &xi)
{
    float2 p = sampleDisk(xi);

    // Project up to hemisphere.
    float z = sqrtf(fmaxf(0.0f, 1.0f - p.x * p.x - p.y * p.y));

    return make_float3(p.x, p.y, z);
}

inline __host__ __device__ float3 sampleCone(const float cosWidth, const float2 &xi)
{
    // cosWidth is the cosine of the angle from the cone axis to the edge of the
    // cone.

    // See https://math.stackexchange.com/questions/56784/
    // generate-a-random-direction-within-a-cone (joriki)

    float z = xi.x * (1.0f - cosWidth) + cosWidth;
    float phi = xi.y * 2.0f * M_PIf;
    float tmp = sqrtf(1.0f - z * z);
    float x = tmp * cosf(phi);
    float y = tmp * sinf(phi);

    //float solidAngle = 2.0f * M_PIf * (1.0f - cosWidth);
    //float pdf = 1.0f / solidAngle;

    return make_float3(x, y, z);
}

inline __host__ __device__ float falloff(const Light &light, const float3 &w)
{
    float cosWidth = light.cosWidth;
    float cosFalloffStart = light.cosFalloffStart;

    float cosTheta = dot(light.normal, w);
    if (cosTheta < cosWidth)
        return 0.0f;
    if (cosTheta > cosFalloffStart)
        return 1.0f;

    float delta = (cosTheta - cosWidth) / (cosFalloffStart - cosWidth);
    return (delta * delta) * (delta * delta);
}
