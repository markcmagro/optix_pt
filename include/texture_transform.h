#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>

struct TextureTransform
{
    float2 offset = make_float2(0.0f);
    float2 scale = make_float2(1.0f);
    float rotation = 0.0f;
};

#else
#include <glm/glm.hpp>

struct TextureTransform
{
    glm::vec2 offset = glm::vec2(0.0f);
    glm::vec2 scale = glm::vec2(1.0f);
    float rotation = 0.0f;
};
#endif
