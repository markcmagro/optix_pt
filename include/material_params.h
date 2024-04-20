#pragma once

#include <iostream>
#include <string>
#include "alpha_mode.h"
#include "material_flags.h"
#include "sheen_data.h"

#include <glm/ext/scalar_common.hpp>

#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <tiny_gltf.h>

struct MaterialParams
{
    std::string name;
    unsigned int category; // MaterialFlags

    glm::vec4 baseColour;
    AlphaMode alphaMode;
    float alphaCutoff;

    glm::vec3 emission;

    float ior;             // index of refraction
    float roughness;       // default 0; > 0 => glossy
    float metallic;        // 0 => dielectric; 1 => metal
    float specular;        // specular factor / weight
    float transmissionWeight;
    float subsurfaceWeight;
    float coatWeight;
    float coatRoughness;
    SheenData sheenData;

    bool baseColourTexPresent;
    int baseColourTexWidth;
    int baseColourTexHeight;
    int baseColourTexNumComponents;
    unsigned char *baseColourTexRgba;
    tinygltf::Sampler *baseColourTexSampler = nullptr;
    TextureTransform baseColourTexTf;

    bool normalTexPresent;
    int normalTexWidth;
    int normalTexHeight;
    int normalTexNumComponents;
    unsigned char *normalTexRgba;
    tinygltf::Sampler *normalTexSampler = nullptr;
    TextureTransform normalTexTf;

    bool metallicRoughnessTexPresent;
    int metallicRoughnessTexWidth;
    int metallicRoughnessTexHeight;
    int metallicRoughnessTexNumComponents;
    unsigned char *metallicRoughnessTexRgba;
    TextureTransform metallicRoughnessTexTf;

    bool emissiveTexPresent;
    int emissiveTexWidth;
    int emissiveTexHeight;
    int emissiveTexNumComponents;
    unsigned char *emissiveTexRgba;
    TextureTransform emissiveTexTf;

    bool sheenColourTexPresent;
    int sheenColourTexWidth;
    int sheenColourTexHeight;
    int sheenColourTexNumComponents;
    unsigned char *sheenColourTexRgba;

    bool sheenRoughnessTexPresent;
    int sheenRoughnessTexWidth;
    int sheenRoughnessTexHeight;
    int sheenRoughnessTexNumComponents;
    unsigned char *sheenRoughnessTexRgba;

    void init(
        const char *name,
        glm::vec4 baseColour = glm::vec4(0.0f),
        AlphaMode alphaMode = AM_OPAQUE, float alphaCutoff = 0.5f,
        glm::vec3 emission = glm::vec3(0.0f),
        float ior = 1.0f, float roughness = 0.0f, float metallic = 0.0f, float specular = 0.0f,
        float transmissionWeight = 0.0f, float coatWeight = 0.0f, float coatRoughness = 0.0f,
        SheenData *sheenData = nullptr)
    {
        this->name = name;
        this->baseColour = baseColour;
        this->alphaMode = alphaMode;
        this->alphaCutoff = alphaCutoff;
        this->emission = emission;
        this->ior = ior;
        this->roughness = roughness;
        this->metallic = metallic;
        this->specular = specular;
        this->transmissionWeight = transmissionWeight;

        subsurfaceWeight = 0.0f;

        this->coatWeight = coatWeight;
        this->coatRoughness = coatRoughness;

        if (sheenData)
            this->sheenData = *sheenData;

        // Material blending is supported.
        // For example, a material is both a metal and a dielectric if metallic
        // is in (0, 1).

        category = MaterialFlags::Uncategorised;

        if (isDeterministic())
            categorise();
    }

    bool isDeterministic()
    {
        // A material is deterministic if there is only the base layer and there
        // is no blending at all.

        // Therefore:
        // 1. There must only be a base layer (coatWeight and sheenWeight must both be 0)
        // and
        // 2. The material is emissive
        //    or
        //    the material is 100% metallic
        //    or
        //    the material is 100% dielectric
        // and
        // 3. If the material is dielectric:
        //    The material is 100% transmissive
        //      (this allows both reflection and refraction determined by
        //      Fresnel; both are specular hits)
        //    or
        //    The material is 0% transmissive
        // 
        //    If the material is 0% transmissive:
        //    specular (IOR level) is 0 (all reflections are turned off)
        //
        //    If specular is 0:
        //    The subsurface weight is either 0% or 100%
        
        // Multiple layers
        glm::vec3 &sheenCol = sheenData.colourFactor;
        if (coatWeight != 0.0f || glm::fmax(sheenCol.r, sheenCol.g, sheenCol.b) != 0.0f)
            return false;

        // Emission
        if (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f) {
            if (emissiveTexPresent)
                return false;
            return true;
        }

        // Metallic/Dielectric blend
        if (metallic != 0.0f && metallic != 1.0f)
            return false;
        if (metallicRoughnessTexPresent)
            return false;

        // Metal
        if (metallic == 1.0f)
            return true;

        // The material is a dielectric.

        // Opaque/Transmission blend
        if (transmissionWeight != 0.0f && transmissionWeight != 1.0f)
            return false;

        // Transmission
        if (transmissionWeight == 1.0f)
            return true;

        // Reflection/Diffuse/Subsurface blend
        if (specular != 0.0f)
            return false;

        // Diffuse/Subsurface blend
        if (subsurfaceWeight != 0.0f && subsurfaceWeight != 1.0f)
            return false;

        // 100% Diffuse or 100% Subsurface
        return true;
    }

    void categorise()
    {
        category = MaterialFlags::Uncategorised;

        if (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f) {
            category |= MaterialFlags::Emission;
            return;
        }

        if (metallic == 1.0f) {
            category |= MaterialFlags::Metal;
        }
        else {
            category |= MaterialFlags::Dielectric;

            if (transmissionWeight == 1.0f) {
                category |= MaterialFlags::Transmission;
            }
            else {
                category |= MaterialFlags::Opaque;

                if (subsurfaceWeight == 1.0f) {
                    category |= MaterialFlags::Subsurface;
                }
                else {
                    category |= MaterialFlags::Diffuse;
                    if (roughness == 0.0f)
                        category |= MaterialFlags::Lambertian;
                }
            }
        }

        if (category & MaterialFlags::Metal || category & MaterialFlags::Transmission) {
            if (roughness == 0.0f)
                category |= MaterialFlags::Specular;
        }
    }

    void printFlags(const char *indent)
    {
        std::printf("%sFlags = 0x%08X\n", indent, category);

        if (category == 0) {
            std::printf("%s  Uncategorised\n", indent);
            return;
        }

        if (category & MaterialFlags::Metal)
            std::printf("%s  Metal\n", indent);
        if (category & MaterialFlags::Dielectric)
            std::printf("%s  Dielectric\n", indent);
        if (category & MaterialFlags::Emission)
            std::printf("%s  Emission\n", indent);
        if (category & MaterialFlags::Opaque)
            std::printf("%s  Opaque\n", indent);
        if (category & MaterialFlags::Transmission)
            std::printf("%s  Transmission\n", indent);
        if (category & MaterialFlags::Reflection)
            std::printf("%s  Reflection\n", indent);
        if (category & MaterialFlags::Diffuse)
            std::printf("%s  Diffuse\n", indent);
        if (category & MaterialFlags::Subsurface)
            std::printf("%s  Subsurface\n", indent);
        if (category & MaterialFlags::Coat)
            std::printf("%s  Coat\n", indent);
        if (category & MaterialFlags::Sheen)
            std::printf("%s  Sheen\n", indent);
        if (category & MaterialFlags::Lambertian)
            std::printf("%s  Lambertian\n", indent);
        if (category & MaterialFlags::Specular)
            std::printf("%s  Specular\n", indent);
    }
};
