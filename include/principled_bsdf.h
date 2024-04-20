#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include "bsdf_sample.h"
#include "cuda_utils.h"
#include "fresnel.h"
#include "launch_params.h"
#include "material_data.h"
#include "material_flags.h"
#include "material_metallic_roughness.h"
#include "oren_nayar.h"
#include "orthonormal_basis.h"
#include "prd.h"
#include "principled_bsdf_helpers.h"
#include "reflection_and_refraction.h"
#include "sampling.h"
#include "schlick.h"
#include "sheen.h"
#include "smith.h"
#include "trowbridge_reitz.h"

extern "C" __constant__ LaunchParams params;

inline __device__ unsigned int categoriseMaterial(PRD &prd, const MaterialData &mat, const float3 &v,
    const float3 &n, const float3 &albedo, float roughness, float metallic, const float3 &emission,
    const float3 &sheenColour, float sheenRoughness, float &categoryWeight)
{
    // Determine main material aspects.
    // Simplifies rendering and needed for the specular hit check.

    // Layer 1 - Base layer, always present
    //   Metal (metallic, reflection only, uses roughness)
    //   Dielectric (not metallic)
    //     Opaque
    //       Reflection (specular > 0, reflection only, uses roughness)
    //       Diffuse/Subsurface (specular = 0, not a specular hit)
    //         Diffuse (subsurface weight = 0, uses roughness)
    //         Subsurface (subsurface weight > 0)
    //     Transmission (transmission weight > 0, reflection and transmission, uses roughness)
    //   Emission (colour * strength > 0, not a specular hit)
    // Layer 2 - Coat (reflection only, has own roughness, IOR and tint)
    // Layer 3 - Sheen (reflection only, has own roughness and tint)

    // Choices done during rendering:
    // 1. Diffuse or Subsurface is determined by subsurface weight
    // 2. Diffuse: roughness = 0 => Lambertian, roughness > 0 => microfacet model such as Oren-Nayar
    // 3. Metal, Reflection, Transmission, Coat, Sheen: roughness = 0 => perfect reflection
    // 4. Metal should use fresnelMetal(). Currently using fresnelSchlick().
    // 5. Transmission uses fresnelDielectric().

    // To determine roughness, use the roughness argument not mat.roughness.
    // The roughness argument is equal to mat.roughness * roughness from the metallic-roughness texture.

    // To determine metalness, use the metallic argument not mat.metallic.
    // The metallic argument is equal to mat.metallic * metalness from the metallic-roughness texture.

    // To determine emission, use the emission argument not mat.emission.
    // The emission argument is equal to mat.emission * emission from the emissive texture.

    // To determine sheenColour and sheenRoughness, use the sheenColour and
    // sheenRoughness arguments respectively instead of mat.sheenColour and
    // mat.sheenRoughness.
    // sheenColour = mat.sheenColour * sheenRoughnessTexture.rgb.
    // sheenRoughness = mat.sheenRoughness * sheenRoughnessTexture.a.

    unsigned int category = mat.category;
    categoryWeight = 1.0f;

    if (category != MaterialFlags::Uncategorised)
        return category;

    const unsigned int reflectiveMaterialFlags =
        MaterialFlags::Metal |
        MaterialFlags::Reflection |
        MaterialFlags::Transmission |
        MaterialFlags::Coat |
        MaterialFlags::Sheen;

    //if (prd.debug) {
    //    printf("baseColour=(%f, %f, %f)\n", mat.baseColour.x, mat.baseColour.y, mat.baseColour.z);
    //    printf("albedo=(%f, %f, %f)\n", albedo.x, albedo.y, albedo.z);
    //}

    // + Determine layer.
    float f0 = calcF0(1.5f); // 0.04; specular highlights barely visible
    float F = fresnelSchlick(f0, dot(v, n));

    float coatWeight = mat.coatWeight * F;
    float baseLayerWeight = 1.0f - coatWeight;
    float sheenWeight = maxValue(sheenColour);

    float3 layerSelection = make_float3(baseLayerWeight, coatWeight, sheenWeight);
    layerSelection /= layerSelection.x + layerSelection.y + layerSelection.z;
    layerSelection.y += layerSelection.x;
    layerSelection.z += layerSelection.y;

    float layerWeight = 1.0f;
    float xi = prd.random.rng();

    /*
    if (prd.debug) {
        printf("categoriseMaterial(): layerSelection=(%f, %f, %f); xi=%f\n", layerSelection.x,layerSelection.y,
            layerSelection.z, xi);
    }
    */

    if (xi < layerSelection.x) { // base layer
        layerWeight = layerSelection.x;
        //if (prd.debug)
        //    printf("categoriseMaterial(): base layer\n");
    }
    else if (xi < layerSelection.y) { // coat
        layerWeight = layerSelection.y;
        category |= MaterialFlags::Coat;
        //if (prd.debug)
        //    printf("categoriseMaterial(): coat layer\n");
    }
    else { // sheen
        layerWeight = layerSelection.z;
        category |= MaterialFlags::Sheen;
        //if (prd.debug)
        //    printf("categoriseMaterial(): sheen layer\n");
    }

    categoryWeight *= layerWeight;
    // -

    // + Coat and Sheen: Specular
    if (category & MaterialFlags::Coat) {
        if (mat.coatRoughness == 0.0f)
            category |= MaterialFlags::Specular;
        return category;
    }

    if (category & MaterialFlags::Sheen) {
        //if (mat.sheenRoughness == 0.0f)
        if (sheenRoughness == 0.0f)
            category |= MaterialFlags::Specular;
        return category;
    }
    // -

    // Base layer

    // Emissive vs Metal vs Dielectric
    if (maxValue(emission) > 0.0f) {
        category |= MaterialFlags::Emission;
    }
    else if (metallic == 1.0f) {
        category |= MaterialFlags::Metal;
    }
    else if (metallic == 0.0f) {
        category |= MaterialFlags::Dielectric;
    }
    else {
        float xi = prd.random.rng();
        if (xi < metallic) {
            category |= MaterialFlags::Metal;
            categoryWeight *= metallic;
        }
        else {
            category |= MaterialFlags::Dielectric;
            categoryWeight *= 1.0f - metallic;
        }
    }

    if (category & MaterialFlags::Dielectric) {
        // Transmission vs Opaque
        if (mat.transmissionWeight == 1.0f) {
            category |= MaterialFlags::Transmission;
        }
        else if (mat.transmissionWeight == 0.0f) {
            category |= MaterialFlags::Opaque;
        }
        else {
            float xi = prd.random.rng();
            if (xi < mat.transmissionWeight) {
                category |= MaterialFlags::Transmission;
                categoryWeight *= mat.transmissionWeight;
            }
            else {
                category |= MaterialFlags::Opaque;
                categoryWeight *= 1.0f - mat.transmissionWeight;
            }
        }
    }

    if (category & MaterialFlags::Opaque) {
        // Reflection
        if (mat.specular > 0.0f) {
            float xi = prd.random.rng();
            if (xi < F) {
                category |= MaterialFlags::Reflection;
                if (roughness == 0.0f)
                    categoryWeight *= F;
            }
            else {
                if (roughness == 0.0f)
                    categoryWeight *= 1.0f - F;
            }
        }

        // Diffuse vs Subsurface
        if (!(category & MaterialFlags::Reflection)) {
            if (mat.subsurfaceWeight == 1.0f) {
                category |= MaterialFlags::Subsurface;
            }
            else if (mat.subsurfaceWeight == 0.0f) {
                category |= MaterialFlags::Diffuse;
            }
            else {
                float xi = prd.random.rng();
                if (xi < mat.subsurfaceWeight) {
                    category |= MaterialFlags::Subsurface;
                    categoryWeight *= mat.subsurfaceWeight;
                }
                else {
                    category |= MaterialFlags::Diffuse;
                    categoryWeight *= 1.0f - mat.subsurfaceWeight;
                }
            }

            // Lambertian
            if (category & MaterialFlags::Diffuse) {
                if (roughness == 0.0f)
                    category |= MaterialFlags::Lambertian;
            }
        }

    }

    // Base layer: Specular
    if (category & reflectiveMaterialFlags) {
        if (roughness == 0.0f)
            category |= MaterialFlags::Specular;
    }
    return category;
}

//
// Diffuse (Lambertian and Oren-Nayar)
//

inline __device__ float3 diffuseGetNewRayDir(float3 normal, float2 xi, bool debug)
{
    float3 newRayDir = cosineSampleHemisphere(xi);
    OrthonormalBasis onb(normal);
    onb.localToWorld(newRayDir);
    return newRayDir;
}

inline __device__ float3 diffuseBrdf(const MaterialMetallicRoughness &mat, float3 l, float3 v, bool debug)
{
    if (debug) {
        printf("      diffuseBrdf()\n");
        printf("        baseColour=(%f, %f, %f)\n", mat.baseColour.x, mat.baseColour.y, mat.baseColour.z);
    }

    if (mat.category & MaterialFlags::Lambertian)
        return mat.baseColour * M_1_PIf;

    return evalOrenNayar(mat, l, v);
}

inline __device__ float diffusePdf(float3 normal, float3 l, float3 v, bool debug)
{
    float cosTheta = dot(normal, l);
    return cosTheta * M_1_PIf;
}

inline __device__ bool diffuseSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 v, float2 xi,
    BsdfSample &sample)
{
    sample.newRayDir = diffuseGetNewRayDir(mat.shadingNormal, xi, prd.debug);
    sample.bsdf = diffuseBrdf(mat, sample.newRayDir, v, prd.debug);
    sample.pdf = diffusePdf(mat.shadingNormal, sample.newRayDir, v, prd.debug);
    return true;
}

//
// Specular reflection
//

inline __device__ bool specularReflectionSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 wo,
    BsdfSample &sample)
{
    sample.newRayDir = reflectRay(mat.shadingNormal, wo);
    sample.bsdf = mat.baseColour / fabs(dot(sample.newRayDir, mat.shadingNormal));
    sample.pdf = 1.0f;
    return true;
}

//
// Specular transmission
//

inline __device__ float3 specularTransmissionGetNewRayDir(const MaterialMetallicRoughness& mat, const float3 v,
    bool debug)
{
    return refractRay(mat.shadingNormal, v, mat.ior);
}

inline __device__ float3 specularTransmissionBrdf(const MaterialMetallicRoughness& mat, const float3 &n,
    const float3 &l, const float3 &v, bool debug)
{
    float nDotL = dot(n, l);
    float3 ret = mat.baseColour / fabs(nDotL);

    if (debug) {
        printf("specularTransmissionBrdf()\n");
        if (dot(n, v) >= 0.0f)
            printf("Outside\n");
        else
            printf("Inside\n");
        printf("n=(%f, %f, %f)\n", n.x, n.y, n.z);
        printf("v=(%f, %f, %f)\n", v.x, v.y, v.z);
        printf("l=(%f, %f, %f)\n", l.x, l.y, l.z);
    }

    return ret;
}

inline __device__ float specularTransmissionPdf(const MaterialMetallicRoughness& mat, float3 l, float3 v, bool debug)
{
    return 1.0f;
}

inline __device__ bool specularTransmissionSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 wo,
    BsdfSample &sample)
{
    if (prd.debug)
        printf("specularTransmissionSample()\n");

    sample.newRayDir = specularTransmissionGetNewRayDir(mat, wo, prd.debug);
    sample.bsdf = specularTransmissionBrdf(mat, mat.shadingNormal, sample.newRayDir, wo, prd.debug);
    sample.pdf = specularTransmissionPdf(mat, sample.newRayDir, wo, prd.debug);
    return true;
}

//
// Specular (combines specular reflection and specular transmission)
//

inline __device__ bool specularSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 wo, float xi,
    BsdfSample &sample)
{
    bool ret = true;

    // Use the normal as the half vector.
    float vDotH = dot(wo, mat.shadingNormal);

    float f0 = calcF0(mat.ior); // returns 0.04 when ior is 1.5
    float F = fresnelSchlick(f0, vDotH);

    if (xi < F) { // reflection
        if (prd.debug && prd.depth == 1) {
            printf("specularSample:\n");
            printf("  reflection; f0=%f; F=%f; xi=%f\n", f0, F, xi);
        }

        ret = specularReflectionSample(prd, mat, wo, sample);
        sample.bsdf *= F;
        sample.pdf *= F;
    }
    else { // transmission
        if (prd.debug && prd.depth == 1) {
            printf("specularSample:\n");
            printf("  transmission; f0=%f; F=%f; xi=%f\n", f0, F, xi);
        }

        ret = specularTransmissionSample(prd, mat, wo, sample);
        sample.bsdf *= (1.0f - F);
        sample.pdf *= (1.0f - F);
    }

    return ret;
}

//
// Reflection
//

inline __device__ float3 reflectionGetNewRayDir(PRD &prd, const MaterialMetallicRoughness &mat, const float3 &n,
    const float3 &v, const float2 &xi)
{
    if (mat.category & MaterialFlags::Specular) {
        return reflectRay(n, v);
    }
    else {
        float3 h = trowbridgeReitzGetHalfVector(mat, n, v, xi, prd.debug);

        if (prd.debug/* && prd.depth == 1*/) {
            float3 r = reflectRay(h, v);

            printf("reflectionGetNewRayDir:\n");
            printf("  normal    = (%f, %f, %f)\n", n.x, n.y, n.z);
            printf("  v         = (%f, %f, %f)\n", v.x, v.y, v.z);
            printf("  h         = (%f, %f, %f)\n", h.x, h.y, h.z);
            printf("  theta     = %f\n", acosf(dot(h, v)) * 180.0f * M_1_PIf);
            printf("  newRayDir = (%f, %f, %f)\n", r.x, r.y, r.z);
        }

        return reflectRay(h, v);
    }
}

inline __device__ float3 reflectionBrdf(const MaterialMetallicRoughness &mat, const float3 &n, const float3 &l,
    const float3 &v, bool debug)
{
    if (mat.category & MaterialFlags::Specular) {
        return mat.baseColour / fabs(dot(l, n));
    }
    else {
        // Microfacet BRDF.

        float3 h = normalize(l + v);

        //float3 f0 = (mat.category & MaterialFlags::Metal) ? mat.baseColour : make_float3(calcF0(1.5f));
        //float3 F = fresnelSchlick(f0, dot(v, h));

        //if (debug)
        //    printf("reflectionBrdf(): f0=(%f, %f, %f); F=(%f, %f, %f)\n", f0.x, f0.y, f0.z, F.x, F.y, F.z);

        float hDotL = dot(h, l);
        float hDotV = dot(h, v);
        float magNDotL = fmaxf(fabs(dot(n, l)), 0.00001f); // to avoid division by zero
        float magNDotV = fmaxf(fabs(dot(n, v)), 0.00001f); // to avoid division by zero
        float V = smithV(hDotL, magNDotL, hDotV, magNDotV, mat.alpha);
        // Alternative (no noticeable difference):
        //float V = smithVJoint(hDotL, magNDotL, hDotV, magNDotV, mat.alpha);

        float nDotH = dot(n, h);
        float D = trowbridgeReitzD(nDotH, mat.alpha, debug);

        //float3 ret = F * V * D;
        float3 ret = make_float3(V * D);

        if (debug) {
            printf("reflectionBrdf:\n");
            printf("  l        = (%f, %f, %f)\n", l.x, l.y, l.z);
            printf("  v        = (%f, %f, %f)\n", v.x, v.y, v.z);
            printf("  h        = (%f, %f, %f)\n", h.x, h.y, h.z);
            printf("  hDotL    = %f\n", hDotL);
            printf("  hDotV    = %f\n", hDotV);
            printf("  nDotL    = %f\n", dot(n, l));
            printf("  nDotV    = %f\n", dot(n, v));
            printf("  magNDotL = %f\n", magNDotL);
            printf("  magNDotV = %f\n", magNDotV);
            //printf("  F        = (%f, %f, %f)\n", F.x, F.y, F.z);
            printf("  V        = %f\n", V);
            printf("  D        = %f\n", D);
            printf("  ret      = (%f, %f, %f)\n", ret.x, ret.y, ret.z);
        }

        return ret;
    }
}

// wi = l = newRayDir
// wo = v = -rayDir
inline __device__ float reflectionPdf(const MaterialMetallicRoughness &mat, const float3 &n, const float3 &l,
    const float3 &v, bool debug)
{
    if (mat.category & MaterialFlags::Specular) {
        return 1.0f;
    }
    else {
        float3 h = normalize(l + v);

        //if (debug) {
        //    printf("reflectionPdf:\n");
        //    printf("  normal = (%f, %f, %f)\n", mat.shadingNormal.x, mat.shadingNormal.y, mat.shadingNormal.z);
        //    printf("  l      = (%f, %f, %f)\n", l.x, l.y, l.z);
        //    printf("  v      = (%f, %f, %f)\n", v.x, v.y, v.z);
        //    printf("  h      = (%f, %f, %f)\n", h.x, h.y, h.z);
        //}

        return trowbridgeReitzPdf(n, h, l, v, mat.alpha, debug);
    }
}

inline __device__ bool reflectionSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 v, float2 xi,
    BsdfSample &sample)
{
    if (mat.category & MaterialFlags::Specular)
        return specularReflectionSample(prd, mat, v, sample);

    sample.newRayDir = reflectionGetNewRayDir(prd, mat, mat.shadingNormal, v, make_float2(xi.x, xi.y));

    float3 h = normalize(sample.newRayDir + v);

    /*
    if (prd.debug) {
        float nDotV = dot(v, mat.shadingNormal);
        float nDotL = dot(sample.newRayDir, mat.shadingNormal);
        printf("angle between v and n=%.2f\n", acosf(nDotV) * 180.0f * M_1_PIf);
        printf("angle between l and n=%.2f\n", acosf(nDotL) * 180.0f * M_1_PIf);
        printf("l=(%f, %f, %f)\n", sample.newRayDir.x, sample.newRayDir.y, sample.newRayDir.z);
        printf("v=(%f, %f, %f)\n", v.x, v.y, v.z);
        printf("n=(%f, %f, %f)\n", mat.shadingNormal.x, mat.shadingNormal.y, mat.shadingNormal.z);
        printf("h=(%f, %f, %f)\n", h.x, h.y, h.z);
        printf("dot(v, mat.shadingNormal)=%f\n", dot(v, mat.shadingNormal));
        printf("dot(sample.newRayDir, h)=%f\n", nDotV);
        printf("dot(sample.newRayDir, mat.shadingNormal)=%f\n", dot(sample.newRayDir, mat.shadingNormal));
    }
    */

    if (dot(sample.newRayDir, h) > 0.0f && dot(sample.newRayDir, mat.shadingNormal) > 0.0f) {
        sample.pdf = reflectionPdf(mat, mat.shadingNormal, sample.newRayDir, v, prd.debug);
        sample.bsdf = reflectionBrdf(mat, mat.shadingNormal, sample.newRayDir, v, prd.debug);
    }
    else {
        sample.bsdf = make_float3(0.0f);
        sample.pdf = 0.0f;
        return false;
    }

    return true;
}

//
// Transmission
//

inline __device__ float3 transmissionGetNewRayDir(const MaterialMetallicRoughness &mat, const float3 v,
    const float2 xi, bool debug)
{
    if (debug)
        printf("transmissionGetNewRayDir()\n");

    if (mat.category & MaterialFlags::Specular) {
        float3 newRayDir = specularTransmissionGetNewRayDir(mat, v, debug);
        if (debug)
            printf("newRayDir=(%f, %f, %f)\n", newRayDir.x, newRayDir.y, newRayDir.z);
        return newRayDir;
    }
    else {
        float3 h = trowbridgeReitzGetHalfVector(mat, mat.shadingNormal, v, xi, debug);
        float3 newRayDir = refractRay(h, v, mat.ior);

        if (debug) {
            printf("shadingNormal=(%f, %f, %f)\n", mat.shadingNormal.x, mat.shadingNormal.y, mat.shadingNormal.z);
            printf("h=(%f, %f, %f)\n", h.x, h.y, h.z);
            printf("newRayDir=(%f, %f, %f)\n", newRayDir.x, newRayDir.y, newRayDir.z);

            float3 specularNewRayDir = specularTransmissionGetNewRayDir(mat, v, debug);
            printf("specularNewRayDir=(%f, %f, %f)\n", specularNewRayDir.x, specularNewRayDir.y, specularNewRayDir.z);

            printf("Angle between newRayDir and specularNewRayDir=%f degrees\n", acosf(dot(newRayDir,
                specularNewRayDir)) * 180.0f * M_1_PIf);
        }

        return newRayDir;
    }
}

inline __device__ float3 transmissionBrdf(const MaterialMetallicRoughness &mat, const float3 &n, const float3 &l,
    const float3 &v, bool debug)
{
    if (debug)
        printf("specularTransmissionBrdf()\n");

    if (mat.category & MaterialFlags::Specular) {
        float3 ret = specularTransmissionBrdf(mat, n, l, v, debug);
        return ret;
    }
    else {
        // Microfacet BRDF.

        // Calculate the transmission half vector.
        //float3 h = transmissionHalfVectorThinWalled(n, l, v);
        float3 h = transmissionHalfVectorVolumetric(n, l, v, 1.0f, 1.5f);

        float hDotL_on_nDotL = dot(h, l) / dot(n, l);
        float hDotV_on_nDotV = dot(h, v) / dot(n, v);
        float magNDotL = fmaxf(fabs(dot(n, l)), 0.00001f); // to avoid division by zero
        float magNDotV = fmaxf(fabs(dot(n, v)), 0.00001f); // to avoid division by zero
        float V = smithV(hDotL_on_nDotL, magNDotL, hDotV_on_nDotV, magNDotV, mat.alpha);

        float nDotH = dot(n, h);
        float D = trowbridgeReitzD(nDotH, mat.alpha, debug);

        float3 ret = make_float3(V * D);

        if (debug) {
            printf("transmissionBrdf:\n");
            printf("  V = %f\n", V);
            printf("  D = %f\n", D);
            printf("  brdf = (%f, %f, %f)\n", ret.x, ret.y, ret.z);
        }

        return ret;
    }
}

inline __device__ float transmissionPdf(const MaterialMetallicRoughness &mat, float3 l, float3 v, bool debug)
{
    if (mat.category & MaterialFlags::Specular) {
        return specularTransmissionPdf(mat, l, v, debug);
    }
    else {
        //float3 h = transmissionHalfVectorThinWalled(mat.shadingNormal, l, v);
        float3 h = transmissionHalfVectorVolumetric(mat.shadingNormal, l, v, 1.0f, 1.5f);

        //if (debug) {
        //    printf("transmissionBrdf:\n");
        //    printf("  normal = (%f, %f, %f)\n", mat.shadingNormal.x, mat.shadingNormal.y, mat.shadingNormal.z);
        //    printf("  l      = (%f, %f, %f)\n", l.x, l.y, l.z);
        //    printf("  v      = (%f, %f, %f)\n", v.x, v.y, v.z);
        //    printf("  h      = (%f, %f, %f)\n", h.x, h.y, h.z);
        //}

        //return trowbridgeReitzPdf(mat.shadingNormal, h, l, v, mat.alpha, debug) / (4.0f * abs(dot(h, v)));
        return trowbridgeReitzPdf(mat.shadingNormal, h, l, v, mat.alpha, debug);
    }
}

inline __device__ bool transmissionSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 v, float2 xi,
    BsdfSample &sample)
{
    if (mat.category & MaterialFlags::Specular)
        return specularTransmissionSample(prd, mat, v, sample);

    sample.newRayDir = transmissionGetNewRayDir(mat, v, make_float2(xi.x, xi.y), prd.debug);
    sample.bsdf = transmissionBrdf(mat, mat.shadingNormal, sample.newRayDir, v, prd.debug);
    sample.pdf = transmissionPdf(mat, sample.newRayDir, v, prd.debug);

    return true;
}

//
// Dielectric (diffuse, reflection and transmission)
//

inline __device__ float3 dielectricBrdf(PRD &prd, const MaterialMetallicRoughness& mat, float3 l, float3 v)
{
    if (mat.category & MaterialFlags::Transmission) {
        if (prd.debug)
            printf("  transmission (dielectricBrdf)\n");

        float f0 = calcF0(mat.ior);
        //float f0 = calcF0(1.5f);
        float vDotH = dot(v, mat.shadingNormal); // use the normal as the half vector
        float F = fresnelSchlick(f0, vDotH);

        float xi = prd.random.rng();

        if (xi < F) {
            if (prd.debug)
                printf("    reflection (dielectricBrdf)\n");

            return reflectionBrdf(mat, mat.shadingNormal, l, v, prd.debug);
        }
        else {
            if (prd.debug)
                printf("    refraction (dielectricBrdf)\n");

            return transmissionBrdf(mat, mat.shadingNormal, l, v, prd.debug);
        }
    }
    else { // opaque
        if (prd.debug)
            printf("  opaque (dielectricBrdf)\n");

        if (mat.category & MaterialFlags::Reflection) {
            if (prd.debug)
                printf("    specular (dielectricBrdf)\n");

            return reflectionBrdf(mat, mat.shadingNormal, l, v, prd.debug);
        }
        else { // mat.category & MaterialFlags::Diffuse
            if (prd.debug)
                printf("    diffuse (dielectricBrdf)\n");

            return diffuseBrdf(mat, l, v, prd.debug);
        }
    }
}

inline __device__ bool dielectricSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 v, float2 xi,
    BsdfSample &sample)
{
    bool ret = true;

    if (mat.category & MaterialFlags::Transmission) {
        if (prd.debug)
            printf("  transmission (dielectricSample)\n");

        float f0 = calcF0(mat.ior);
        f0 = fmin(f0, 1.0f);
        float vDotH = dot(v, mat.shadingNormal); // use the normal as the half vector
        float F = fresnelSchlick(f0, vDotH);

        float r1 = prd.random.rng();

        if (r1 < F) {
            if (prd.debug)
                printf("    reflection (dielectricSample) F=%f\n", F);

            ret = reflectionSample(prd, mat, v, xi, sample);
            sample.bsdf *= F;
            sample.pdf *= F;
        }
        else {
            if (prd.debug)
                printf("    refraction (dielectricSample) F=%f\n", F);

            ret = transmissionSample(prd, mat, v, xi, sample);
            sample.bsdf *= 1.0f - F;
            sample.pdf *= 1.0f - F;
        }
    }
    else { // opaque
        if (prd.debug)
            printf("  opaque (dielectricSample)\n");

        if (mat.category & MaterialFlags::Reflection) {
            if (prd.debug)
                printf("    specular (dielectricSample)\n");

            ret = reflectionSample(prd, mat, v, xi, sample);
        }
        else { // mat.category & MaterialFlags::Diffuse
            if (prd.debug)
                printf("    diffuse (dielectricSample)\n");

            ret = diffuseSample(prd, mat, v, xi, sample);
        }
    }

    return ret;
}

//
// Metal (reflection only)
// Similar to Reflection but the BRDF is slightly modified.
//

inline __device__ float3 metalGetNewRayDir(PRD &prd, const MaterialMetallicRoughness &mat, const float3 wo,
    const float2 xi)
{
    return reflectionGetNewRayDir(prd, mat, mat.shadingNormal, wo, xi);
}

inline __device__ float3 metalBrdf(const MaterialMetallicRoughness &mat, float3 l, float3 v, bool debug)
{
    float3 fSpecular = reflectionBrdf(mat, mat.shadingNormal, l, v, debug);

    float3 h = normalize(l + v);
    float vDotH = dot(v, h);
    float3 ret = conductorFresnel(mat.baseColour, fSpecular, vDotH);

    if (debug) {
        printf("metalBrdf(): fSpecular=(%f, %f, %f); vDotH=%f; ret=(%f, %f, %f)\n", fSpecular.x, fSpecular.y,
            fSpecular.z, vDotH, ret.x, ret.y, ret.z);
    }

    return ret;
}

inline __device__ float metalPdf(const MaterialMetallicRoughness &mat, float3 h, float3 wo, bool debug)
{
    return reflectionPdf(mat, mat.shadingNormal, h, wo, debug);
}

inline __device__ bool metalSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 v, float2 xi,
    BsdfSample &sample)
{
    sample.newRayDir = metalGetNewRayDir(prd, mat, v, xi);

    if (prd.debug) {
        printf("metalSample():\n");
        printf("  v = (%f, %f, %f)\n", v.x, v.y, v.z);
        float3 l = sample.newRayDir;
        printf("  l = (%f, %f, %f)\n", l.x, l.y, l.z);
    }

    sample.pdf = metalPdf(mat, sample.newRayDir, v, prd.debug);
    sample.bsdf = metalBrdf(mat, sample.newRayDir, v, prd.debug);

     return true;
}

//
// Sheen
//

inline __device__ float3 sheenGetNewRayDir(PRD &prd, const MaterialMetallicRoughness &mat, const float3 wo,
    const float2 xi)
{
    return reflectionGetNewRayDir(prd, mat, mat.shadingNormal, wo, xi);
}

inline __device__ float3 sheenBrdf(const MaterialMetallicRoughness &mat, float3 l, float3 v, bool debug)
{
    if (mat.category & MaterialFlags::Specular) {
        return mat.sheenColour / fabs(dot(l, mat.shadingNormal));
    }
    else {
        float3 h = normalize(l + v);

        float hDotL = dot(h, l);
        float hDotV = dot(h, v);
        float V = charlieSheenV(hDotV, hDotL, mat.alpha);

        /*
        // Using the shading normal instead of the half vector results in an
        // extremely dark result. Why?
        float nDotL = dot(mat.shadingNormal, l);
        float nDotV = dot(mat.shadingNormal, v);
        float V = charlieSheenV(nDotV, nDotL, mat.alpha);
        */

        float nDotH = dot(mat.shadingNormal, h);
        float D = charlieSheenD(nDotH, mat.alpha, debug);

        // According to the glTF specs (KHR_materials_sheen), F can be set to 1.
        float F = 1.0f;

        float3 ret = mat.sheenColour * make_float3(F * V * D) / fabs(dot(mat.shadingNormal, v));
        return ret;
    }
}

inline __device__ float sheenPdf(const MaterialMetallicRoughness &mat, float3 l, float3 v, bool debug)
{
    // WIP.

    /*
    if (mat.category & MaterialFlags::Specular) {
        return 1.0f;
    }
    else {
        float3 h = normalize(l + v);

        float nDotH = fmax(0.00001f, dot(mat.shadingNormal, h));
        float D = charlieSheenD(nDotH, mat.alpha, debug);
        
        float hDotV = dot(h, v);
        float pdf = (D * nDotH) / (4.0f * fabs(hDotV));

        return pdf;
    }
    */

    return 1.0f;
}

inline __device__ bool sheenSample(PRD &prd, const MaterialMetallicRoughness &mat, float3 v, float2 xi, BsdfSample &sample)
{
    sample.newRayDir = sheenGetNewRayDir(prd, mat, v, xi);
    sample.bsdf = sheenBrdf(mat, sample.newRayDir, v, prd.debug);
    sample.pdf = sheenPdf(mat, sample.newRayDir, v, prd.debug);

    return true;
}

//
// Metallic Roughness
// Based on Dielectric and Metal.
//

inline __device__ float3 metallicRoughnessBrdf(PRD &prd, const MaterialMetallicRoughness& mat, float3 l, float3 v)
{
    if (mat.category & MaterialFlags::Metal) {
        if (prd.debug)
            printf("metal (metallicRoughnessBrdf)\n");

        return metalBrdf(mat, l, v, prd.debug);
    }
    else if (mat.category & MaterialFlags::Dielectric) {
        if (prd.debug)
            printf("dielectric (metallicRoughnessBrdf)\n");

        return dielectricBrdf(prd, mat, l, v);
    }
    else if (mat.category & MaterialFlags::Coat) {
        if (prd.debug)
            printf("clearcoat (metallicRoughnessBrdf)\n");

        return reflectionBrdf(mat, mat.shadingNormal, l, v, prd.debug);
    }
    else if (mat.category & MaterialFlags::Sheen) {
        if (prd.debug)
            printf("sheen (metallicRoughnessBrdf)\n");

        return sheenBrdf(mat, l, v, prd.debug);
    }
    else {
        if (prd.debug)
            printf("unknown (metallicRoughnessBrdf)\n");

        return make_float3(1.0f, 0.0f, 1.0f);
    }
}

// wo is the viewer direction (where the camera ray is coming from, possibly after bouncing).
// newDir is the light direction (the direction of the reflected ray).
inline __device__ bool metallicRoughnessSample(PRD &prd, const MaterialMetallicRoughness &mat, const float3 wo,
    const float3 xi, BsdfSample &sample)
{
    // For PBR materials, mat.metallic is either 0 (dielectric) or 1 (metal).

    /*
    if (prd.debug) {
        printf("metallicRoughnessSample:\n");
        printf("  normal    = (%f, %f, %f)\n", mat.shadingNormal.x, mat.shadingNormal.y, mat.shadingNormal.z);
        printf("  wo        = (%f, %f, %f)\n", wo.x, wo.y, wo.z);
        printf("  xi        = (%f, %f, %f)\n", xi.x, xi.y, xi.z);
    }
    */

    bool ret = true;

    if (mat.category & MaterialFlags::Metal) {
        if (prd.debug)
            printf("metal (metallicRoughnessSample)\n");

        ret = metalSample(prd, mat, wo, make_float2(xi.x, xi.y), sample);
    }
    else if (mat.category & MaterialFlags::Dielectric) {
        if (prd.debug)
            printf("dielectric (metallicRoughnessSample)\n");

        ret = dielectricSample(prd, mat, wo, make_float2(xi.x, xi.y), sample);
    }
    else if (mat.category & MaterialFlags::Coat) {
        if (prd.debug)
            printf("clearcoat (metallicRoughnessSample)\n");

        ret = reflectionSample(prd, mat, wo, make_float2(xi.x, xi.y), sample);
    }
    else if (mat.category & MaterialFlags::Sheen) {
        if (prd.debug)
            printf("sheen (metallicRoughnessSample)\n");

        ret = sheenSample(prd, mat, wo, make_float2(xi.x, xi.y), sample);
    }
    else {
        if (prd.debug)
            printf("unknown (metallicRoughnessSample)\n");

        sample.newRayDir = make_float3(0.0f, 1.0f, 0.0f);
        sample.bsdf = make_float3(1.0f, 0.0f, 1.0f);
        sample.pdf = 1.0f;
    }

    if (prd.debug) {
        printf("  ret        = %d\n", ret ? 1 : 0);
        printf("  depth      = %d\n", prd.depth);
        printf("  newRayDir  = (%f, %f, %f)\n", sample.newRayDir.x, sample.newRayDir.y, sample.newRayDir.z);
        printf("  brdf       = (%f, %f, %f)\n", sample.bsdf.x, sample.bsdf.y, sample.bsdf.z);
        printf("  pdf        = %f\n", sample.pdf);
        printf("----\n");
    }

    return ret;
}
