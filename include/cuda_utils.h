#pragma once

#include <sutil/vec_math.h>

#define MAX_DEPTH     8
#define SCENE_EPSILON 0.0001f
#define SHADOW_BIAS   0.001f

inline __device__ float calcLuminance(float3 colour)
{
    return dot(colour, make_float3(0.2126f, 0.7152f, 0.0722f));
}

inline __device__ float maxValue(float3 colour)
{
    return fmaxf(colour.x, fmaxf(colour.y, colour.z));
}

inline __device__ float lerp(float a, float b, float t)
{
  return a + t * (b - a);
}

inline __device__ float3 sphericalToCartesian(float theta, float phi)
{
    float x = sinf(theta) * cosf(phi);
    float y = sinf(theta) * sinf(phi);
    float z = cosf(theta);
    return make_float3(x, y, z);
}

// From https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/MDL_renderer/shaders/transform.h.
// Matrix3x4 * vector. v.w == 0.0f.
inline __device__ float3 transformVector(const float4 *m, const float3 &v)
{
    float3 r;

    r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
    r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
    r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

    return r;
}
