#pragma once

#include <sutil/vec_math.h>

inline __host__ __device__ float4 toneMapReinhard(const float4 &c, float limit)
{
    return c / (c + 1.0f);

    /*
    // Perceptual alternative.
    float luminance = 0.3f * c.x + 0.6f * c.y + 0.1f * c.z;
    float4 col = c * 1.0f / (1.0f + luminance / limit);
    return make_float4(col.x, col.y, col.z, 1.0f);
    */
}

inline __host__ __device__ float4 toneMapAcesNarkowicz(const float4 &col)
{
    // References:
    // - https://github.com/KhronosGroup/glTF-Sample-Viewer/blob/main/source/Renderer/shaders/tonemapping.glsl.

    const float a = 2.51f;
    const float b = 0.03f;
    const float c = 2.43f;
    const float d = 0.59f;
    const float e = 0.14f;

    return clamp((col * (a * col + b)) / (col * (c * col + d) + e), 0.0f, 1.0f);
}
