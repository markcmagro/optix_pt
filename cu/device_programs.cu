#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>
#include "cuda_utils.h"
#include "gamma_correction.h"
#include "launch_params.h"
#include "material_flags.h"
#include "optix_utils.h"
#include "prd.h"
#include "sbt_record.h"
#include "tone_mapping.h"

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __raygen__renderFrame()
{
    const uint3 idx = optixGetLaunchIndex();
    const uint3 dim = optixGetLaunchDimensions();
    uint32_t threadId = idx.y * dim.x + idx.x;

    PRD prd;
#ifdef USE_CURAND
    prd.random.init(&params.randState[threadId]); // CURAND
#else
    prd.random.init(threadId, params.seed); // LCG
#endif
    prd.radiance = make_float3(0.0f);
    prd.throughput = make_float3(1.0f);
    prd.origin = params.eye;
    prd.depth = 0;
    prd.done = false;
    prd.specularHit = false;
    prd.debug = false;
    prd.error = false;

    // Enable debugging for a specific pixel.
    //if (params.seed < 1 && idx.x == 1060 && idx.y == 605)
    //    prd.debug = true;

    float2 pixel = make_float2(idx.x, idx.y);

    // Add random jitter.
    float2 fragment = pixel + prd.random.rng2();

    // No jitter.
    //float2 fragment = pixel + make_float2(0.5f, 0.5f);

    // Use normalised device coordinates [-1, 1] for x and [1, -1] for y. This
    // produces the image oriented the right way up (pixel (0, 0) is the top-
    // left pixel), which is great for debugging purposes.
    const float2 ndc = (fragment / make_float2(dim.x, dim.y)) * make_float2(2.0f, -2.0f) + make_float2(-1.0f, 1.0f);

    prd.dir = normalize(ndc.x * params.U + ndc.y * params.V + params.W);

    uint32_t u0, u1;
    packPointer((void *)&prd, u0, u1);

    while (!prd.done) {
        //if (prd.depth != 0)
        //    prd.debug = false;

        prd.depth++;
        optixTrace(params.traversable, prd.origin, prd.dir, 0.0f, 1e16f, 0.0f, OptixVisibilityMask(1),
            OPTIX_RAY_FLAG_NONE, (int)RayType::Radiance, (int)RayType::Count, (int)RayType::Radiance, u0, u1);
    }

    if (isnan(prd.radiance.x) || isnan(prd.radiance.y) || isnan(prd.radiance.z))
        return;

    //if (prd.debug) {
    //    printf("**** radiance (sample %d) = (%f, %f, %f)\n", params.sample, prd.radiance.x, prd.radiance.y,
    //        prd.radiance.z);
    //}

    float4 *newAccumFb = (float4 *)params.accumFrameBuffer;
    float4 accumVal = newAccumFb[threadId];
    if (params.sample > 0)
        accumVal = lerp(accumVal, make_float4(prd.radiance, 0.0f), 1.0f / (float)(params.sample + 1.0f));
    else
        accumVal = make_float4(prd.radiance, 0.0f);

    // Tone mapping and gamma correction.

    float4 processedAccumVal;
    if (params.toneMapping) {
        if (params.toneMappingImpl == 0)
            processedAccumVal = toneMapReinhard(accumVal, 1.5f);
        else
            processedAccumVal = toneMapAcesNarkowicz(accumVal);
    }
    else {
        processedAccumVal = accumVal;
    }

    if (params.gammaCorrection)
        processedAccumVal = linearToSrgb(processedAccumVal, params.gammaTerm);

    float4 *newFb = (float4 *)params.frameBuffer;
    newFb[threadId] = processedAccumVal;

    newAccumFb[threadId] = accumVal;

#ifdef USE_CURAND
    params.randState[threadId] = prd.random.state;
#endif
}

extern "C" __global__ void __anyhit__radiance()
{
    PRD &prd = *getPrd<PRD>();

    const HitGroupData &sbtData = *(const HitGroupData *)optixGetSbtDataPointer();
    const int primId = optixGetPrimitiveIndex();
    const int3 index = sbtData.triIndices[primId];

    MaterialData &mat = sbtData.matData[sbtData.matIndices[primId + sbtData.triOffset]];

    float alpha = mat.baseColour.w;
    float textureW = 1.0f;

    // Checking alpha from the baseColour texture's alpha channel.
    if (mat.hasBaseColourTexture) {
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const float2 tc = (1.0f - u - v) * sbtData.texCoords[index.x] +
                          u * sbtData.texCoords[index.y] +
                          v * sbtData.texCoords[index.z];

        float4 f = tex2D<float4>(mat.baseColourTexture, tc.x, tc.y);
        textureW = f.w;
    }

    alpha *= textureW;

    /*
    if (prd.debug) {
        printf("__anyhit__radiance():\n");
        printf("  alphaMode=%d alphaCutoff=%f\n", mat.alphaMode, mat.alphaCutoff);
        printf("  baseColour.w=%f alpha=%f hasTexture=%d textureW=%f\n", mat.baseColour.w, alpha,
            mat.hasBaseColourTexture, textureW);

        const float3 rayOrig = optixGetWorldRayOrigin();
        const float3 rayDir = optixGetWorldRayDirection();
        const float rayT = optixGetRayTmax();
        float3 hitPoint = rayOrig + rayT * rayDir;
        printf("  hitPoint=(%f, %f, %f)\n", hitPoint.x, hitPoint.y, hitPoint.z);
    }
    */

    if (mat.alphaMode == AM_MASK) {
        if (alpha < mat.alphaCutoff) {
            if (prd.debug)
                printf("  Intersection ignored.\n");
            optixIgnoreIntersection();
        }
    }
    else if (mat.alphaMode == AM_BLEND) {
        float xi = prd.random.rng();
        if (xi > alpha) {
            if (prd.debug)
                printf("  Intersection ignored.\n");
            optixIgnoreIntersection();
        }
    }
}

extern "C" __global__ void __miss__radiance()
{
    const MissData &sbtData = *(const MissData *)optixGetSbtDataPointer();
    PRD &prd = *getPrd<PRD>();
    prd.radiance += sbtData.colour * prd.throughput; // sbtData.colour is the bg colour
    prd.done = true;
}

extern "C" __global__ void __closesthit__shadow()
{
}

extern "C" __global__ void __anyhit__shadow()
{
    ShadowRayData &srd = *getPrd<ShadowRayData>();

    const HitGroupData &sbtData = *(const HitGroupData *)optixGetSbtDataPointer();
    const int primId = optixGetPrimitiveIndex();
    const int3 index = sbtData.triIndices[primId];

    MaterialData &mat = sbtData.matData[sbtData.matIndices[primId + sbtData.triOffset]];

    float alpha = mat.baseColour.w;
    float textureW = 1.0f;

    // Checking alpha from the baseColour texture's alpha channel.
    if (mat.hasBaseColourTexture) {
        const float u = optixGetTriangleBarycentrics().x;
        const float v = optixGetTriangleBarycentrics().y;

        const float2 tc = (1.0f - u - v) * sbtData.texCoords[index.x] +
                          u * sbtData.texCoords[index.y] +
                          v * sbtData.texCoords[index.z];

        float4 f = tex2D<float4>(mat.baseColourTexture, tc.x, tc.y);
        textureW *= f.w;
    }

    alpha *= textureW;

    if (mat.alphaMode == AM_MASK) {
        if (alpha < mat.alphaCutoff)
            optixIgnoreIntersection();
    }
    else if (mat.alphaMode == AM_BLEND) {
        if (srd.xi > alpha)
            optixIgnoreIntersection();
    }

    srd.lightVisibility = make_float3(0.0f);
    optixTerminateRay();
}

extern "C" __global__ void __miss__shadow()
{
}
