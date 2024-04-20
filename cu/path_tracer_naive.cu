// Naive (brute force) path tracer.

#include "cuda_utils.h"
#include "material_flags.h"
#include "optix_utils.h"
#include "orthonormal_basis.h"
#include "prd.h"
#include "reflection_and_refraction.h"
#include "sampling.h"
#include "sbt_record.h"

extern "C" __global__ void __closesthit__radiance_naive()
{
    const HitGroupData &sbtData = *(const HitGroupData *)optixGetSbtDataPointer();
    const int primId = optixGetPrimitiveIndex();
    const int3 index = sbtData.triIndices[primId];

    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    const float3 rayOrig = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();
    const float rayT = optixGetRayTmax();
    float3 hitPoint = rayOrig + rayT * rayDir;

    // Shading normal.
    const float3 normal = normalize((1.0f - u - v) * sbtData.normals[index.x] + u * sbtData.normals[index.y] +
        v * sbtData.normals[index.z]);

    // Geometric normal.
    float3 triVerts[3];
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), primId, optixGetSbtGASIndex(), 0.0f, triVerts);
    const float3 crossProduct = cross(triVerts[1] - triVerts[0], triVerts[2] - triVerts[0]);
    const float3 NgObject = normalize(crossProduct); // object space
    const float3 Ng = optixTransformNormalFromObjectToWorldSpace(NgObject); // world space

    // Face forward normal.
    const float3 ffnormal = faceforward(normal, -rayDir, Ng);

    MaterialData &mat = sbtData.matData[sbtData.matIndices[primId + sbtData.triOffset]];

    float3 albedo = make_float3(mat.baseColour);
    float3 baseColourTex = make_float3(1.0f);
    if (mat.hasBaseColourTexture) {
        const float2 tc = (1.0f - u - v) * sbtData.texCoords[index.x] +
                          u * sbtData.texCoords[index.y] +
                          v * sbtData.texCoords[index.z];
        baseColourTex = (float3 &)tex2D<float4>(mat.baseColourTexture, tc.x, tc.y);
        albedo *= baseColourTex;
    }

    PRD &prd = *getPrd<PRD>();

    if (mat.category & MaterialFlags::Emission) {
        if (dot(rayDir, normal) < 0.0f) {
            float3 radiance = mat.emission;
            prd.radiance += radiance * baseColourTex * prd.throughput;
            prd.done = true;
            return;
        }
    }

    if (prd.depth == MAX_DEPTH) {
        prd.done = true;
        return;
    }

    // Choose between diffuse, reflection and transmission.
	float3 selection = make_float3(length(make_float3(mat.baseColour)), mat.specular, mat.transmissionWeight);
	selection /= selection.x + selection.y + selection.z;
	selection.y += selection.x;
	selection.z += selection.y;

    float selectionPdf = 1.0f;

    float3 newRayDir;
    float3 bsdf;
    float pdf = 0.0f;
    //float cosTerm;

    float r = prd.random.rng();

    if (r < selection.x) { // diffuse
        selectionPdf = selection.x;

        // Calc next ray dir (cosine-weighted).
        float2 xi = prd.random.rng2();
        newRayDir = cosineSampleHemisphere(xi);

        OrthonormalBasis onb(ffnormal);
        onb.localToWorld(newRayDir);

        // Calc cosine term (cosTheta / pdf).
        //float cosTheta = fmaxf(0.0f, dot(ffnormal, newRayDir)); // ignore back faces
        float cosTheta = dot(ffnormal, newRayDir);
        pdf = cosTheta * M_1_PIf;
        //cosTerm = cosTheta / pdf;
        bsdf = albedo * M_1_PIf;
    }
    else if (r < selection.y) { // reflection
        //selectionPdf = selection.y - selection.x;
        selectionPdf = selection.y;

        // Calc next ray dir.
        newRayDir = reflect(rayDir, ffnormal);

        // Calc cosine term (cosTheta / pdf).
        float cosTheta = dot(normal, newRayDir);
        pdf = 1.0f;
        //cosTerm = cosTheta / pdf;
        bsdf = make_float3(mat.baseColour) / fabs(cosTheta);
    }
    else if (r < selection.z) { // transmission
        //selectionPdf = selection.z - selection.y;
        selectionPdf = selection.z;

        OrthonormalBasis onb(normal);
        float3 wo = onb.worldToLocal(-rayDir);

        float3 n = make_float3(0.0f, 0.0f, 1.0f);
        newRayDir = refractRay(n, wo, mat.ior);
        onb.localToWorld(newRayDir);

        // Calc cosine term (cosTheta / pdf).
        float cosTheta = dot(normal, newRayDir);
        pdf = 1.0f;
        //cosTerm = cosTheta / pdf;
        bsdf = make_float3(mat.baseColour) * fabs(cosTheta);
    }

    if (pdf < SCENE_EPSILON || fmaxf(bsdf) < SCENE_EPSILON) {
        prd.done = true;
        return;
    }

    // fabs(dot(normal, newRayDir)) is used here instead of
    // fmaxf(0.0f, dot(normal, newRayDir)) to support transmissive objects.
    //prd.throughput *= bsdf * cosTerm;
    prd.throughput *= bsdf * fabs(dot(normal, newRayDir)) / pdf / selectionPdf;

    //prd.origin = hitPoint + ffnormal * SCENE_EPSILON;
    prd.origin = hitPoint + newRayDir * SCENE_EPSILON;
    prd.dir = newRayDir;
}
