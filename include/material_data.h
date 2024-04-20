#pragma once

#include "alpha_mode.h"
#include "texture_transform.h"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#define VEC2 float2
#define VEC3 float3
#define VEC4 float4
#else
#include <glm/glm.hpp>
#define VEC2 glm::vec2
#define VEC3 glm::vec3
#define VEC4 glm::vec4
#endif

struct alignas(16) MaterialData
{
    cudaTextureObject_t baseColourTexture;
    cudaTextureObject_t normalTexture;
    cudaTextureObject_t metallicRoughnessTexture;
    cudaTextureObject_t emissiveTexture;
    cudaTextureObject_t sheenColourTexture;
    cudaTextureObject_t sheenRoughnessTexture;

    VEC4 baseColour;
    AlphaMode alphaMode;
    float alphaCutoff;

    VEC3 sheenColour;
    VEC3 emission;

    alignas(16) TextureTransform baseColourTexTf;
    alignas(16) TextureTransform normalTexTf;
    alignas(16) TextureTransform metallicRoughnessTexTf;
    alignas(16) TextureTransform emissiveTexTf;
    alignas(16) TextureTransform sheenColourTexTf;
    alignas(16) TextureTransform sheenRoughnessTexTf;

    alignas(8) float ior;
    float roughness;
    float metallic;
    float specular; // specular factor / weight
    float transmissionWeight;
    float subsurfaceWeight;
    float coatWeight;
    float coatRoughness;
    float sheenRoughness;

    unsigned int type;     // MaterialType
    unsigned int category; // MaterialFlags

    bool hasBaseColourTexture;
    bool hasNormalTexture;
    bool hasMetallicRoughnessTexture;
    bool hasEmissiveTexture;
    bool hasSheenColourTexture;
    bool hasSheenRoughnessTexture;
};
