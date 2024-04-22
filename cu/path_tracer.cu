// Path tracer using next event estimation.

#include "barycentrics.h"
#include "bsdf_sample.h"
#include "cuda_utils.h"
#include "launch_params.h"
#include "material_flags.h"
#include "optix_utils.h"
#include "prd.h"
#include "principled_bsdf.h"
#include "principled_bsdf_helpers.h"
#include "sbt_record.h"
#include "schlick.h"

extern "C" __constant__ LaunchParams params;

extern "C" __global__ void __closesthit__radiance()
{
    PRD &prd = *getPrd<PRD>();
    //if (prd.debug)
    //    printf("depth = %d\n", prd.depth);

    const HitGroupData &sbtData = *(const HitGroupData *)optixGetSbtDataPointer();
    const int primId = optixGetPrimitiveIndex();
    const int3 index = sbtData.triIndices[primId];

    MaterialData &mat = sbtData.matData[sbtData.matIndices[primId + sbtData.triOffset]];

    const float u = optixGetTriangleBarycentrics().x;
    const float v = optixGetTriangleBarycentrics().y;

    float2 tc = make_float2(0.0f);
    if (mat.hasBaseColourTexture || mat.hasNormalTexture || mat.hasMetallicRoughnessTexture || mat.hasEmissiveTexture ||
        mat.hasSheenColourTexture || mat.hasSheenRoughnessTexture) {
        tc = (1.0f - u - v) * sbtData.texCoords[index.x] +
             u * sbtData.texCoords[index.y] +
             v * sbtData.texCoords[index.z];
    }

    const float3 rayOrig = optixGetWorldRayOrigin();
    const float3 rayDir = optixGetWorldRayDirection();
    const float rayT = optixGetRayTmax();
    float3 hitPoint = rayOrig + rayT * rayDir;

    //if (prd.debug) {
    //    printf("__closesthit__radiance():\n");
    //    printf("  hitPoint=(%f, %f, %f)\n", hitPoint.x, hitPoint.y, hitPoint.z);
    //}

    // Geometric normal.
    float3 triVerts[3];
    optixGetTriangleVertexData(optixGetGASTraversableHandle(), primId, optixGetSbtGASIndex(), 0.0f, triVerts);
    const float3 crossProduct = cross(triVerts[1] - triVerts[0], triVerts[2] - triVerts[0]);
    const float3 NgObject = normalize(crossProduct); // object space
    const float3 Ng = optixTransformNormalFromObjectToWorldSpace(NgObject); // world space

    //if (prd.debug) {
    //    printf("Geometric normal (object space)=(%f, %f, %f)\n", NgObject.x, NgObject.y, NgObject.z);
    //    printf("Geometric normal (world space)=(%f, %f, %f)\n", Ng.x, Ng.y, Ng.z);
    //}

    // Shading normal.
    float3 normal;
    if (mat.hasNormalTexture) {
        // Sample the normal from the normal map.
        float2 tempTc = tc;
        tempTc *= mat.normalTexTf.scale;
        tempTc += mat.normalTexTf.offset;
        float3 sampledN = (float3 &)tex2D<float4>(mat.normalTexture, tempTc.x, tempTc.y);
        //if (prd.debug)
        //    printf("Sampled normal (pre-map)=(%f, %f, %f)\n", sampledN.x, sampledN.y, sampledN.z);

        sampledN = sampledN * 2.0f - 1.0f;
        sampledN = normalize(sampledN);
        //if (prd.debug)
        //    printf("Sampled normal=(%f, %f, %f)\n", sampledN.x, sampledN.y, sampledN.z);

        // Generate T, B and N.

        float3 edge1 = triVerts[1] - triVerts[0];
        float3 edge2 = triVerts[2] - triVerts[0];
        float2 deltaUv1 = sbtData.texCoords[index.y] - sbtData.texCoords[index.x];
        float2 deltaUv2 = sbtData.texCoords[index.z] - sbtData.texCoords[index.x];

        float f = 1.0f / (deltaUv1.x * deltaUv2.y - deltaUv2.x * deltaUv1.y);

        float3 tLocal;
        tLocal.x = f * (deltaUv2.y * edge1.x - deltaUv1.y * edge2.x);
        tLocal.y = f * (deltaUv2.y * edge1.y - deltaUv1.y * edge2.y);
        tLocal.z = f * (deltaUv2.y * edge1.z - deltaUv1.y * edge2.z);
        tLocal = normalize(tLocal);

        //if (prd.debug)
        //    printf("tLocal=(%f, %f, %f)\n", tLocal.x, tLocal.y, tLocal.z);

        /*
        float3 bLocal;
        bLocal.x = f * (-deltaUv2.x * edge1.x + deltaUv1.x * edge2.x);
        bLocal.y = f * (-deltaUv2.x * edge1.y + deltaUv1.x * edge2.y);
        bLocal.z = f * (-deltaUv2.x * edge1.z + deltaUv1.x * edge2.z);
        bLocal = normalize(bLocal);
        */

        float3 T = optixTransformNormalFromObjectToWorldSpace(tLocal);
        //float3 B = optixTransformNormalFromObjectToWorldSpace(bLocal);

        float3 N = Ng;
        // Use shading normals if available.
        if (sbtData.normals) {
            N = normalize(
                (1.0f - u - v) * sbtData.normals[index.x] +
                u * sbtData.normals[index.y] +
                v * sbtData.normals[index.z]);
        }

        // Re-orthogonalise T w.r.t. N.
        T = normalize(T - dot(T, N) * N);

        float3 B = cross(N, T);

        //if (prd.debug) {
        //    printf("T=(%f, %f, %f)\n", T.x, T.y, T.z);
        //    printf("B=(%f, %f, %f)\n", B.x, B.y, B.z);
        //    printf("N=(%f, %f, %f)\n", N.x, N.y, N.z);
        //}

        // transformVector() expects the transform matrix in row-major order.
        float TBN[12];
        TBN[0]  = T.x;
        TBN[1]  = B.x;
        TBN[2]  = N.x;
        TBN[3]  = 0.0f;
        TBN[4]  = T.y;
        TBN[5]  = B.y;
        TBN[6]  = N.y;
        TBN[7]  = 0.0f;
        TBN[8]  = T.z;
        TBN[9]  = B.z;
        TBN[10] = N.z;
        TBN[11] = 0.0f;

        // Transform the sampled normal.
        normal = normalize(transformVector((float4 *)TBN, sampledN));
    }
    else if (sbtData.normals) {
        normal = normalize(
            (1.0f - u - v) * sbtData.normals[index.x] +
            u * sbtData.normals[index.y] +
            v * sbtData.normals[index.z]);
    }
    else {
        normal = Ng;
    }
    //if (prd.debug)
    //    printf("  shading normal  = (%f, %f, %f)\n", normal.x, normal.y, normal.z);

    // Face forward normal.
    const float3 ffnormal = faceforward(normal, -rayDir, Ng);
    //if (prd.debug)
    //    printf("  ffnormal        = (%f, %f, %f)\n", ffnormal.x, ffnormal.y, ffnormal.z);

    //if (prd.debug)
    //    printf("  alphaMode=%d alphaCutoff=%f\n", mat.alphaMode, mat.alphaCutoff);

    float3 albedo = make_float3(mat.baseColour);
    float alpha = mat.baseColour.w;

    float4 baseColourTex = make_float4(1.0f);

    if (mat.hasBaseColourTexture) {
        float2 tempTc = tc;
        tempTc *= mat.baseColourTexTf.scale;
        tempTc += mat.baseColourTexTf.offset;
        baseColourTex = tex2D<float4>(mat.baseColourTexture, tempTc.x, tempTc.y);
        albedo *= make_float3(baseColourTex);
        alpha *= baseColourTex.w;
    }

    float occlusion = 1.0f;
    float roughness = mat.roughness;
    float metallic = mat.metallic;
    float3 metallicRoughnessTexture = make_float3(1.0f); // r=occlusion, g=roughness, b=metalness (specs 3.9.2)
    if (mat.hasMetallicRoughnessTexture) {
        float2 tempTc = tc;
        tempTc *= mat.metallicRoughnessTexTf.scale;
        tempTc += mat.metallicRoughnessTexTf.offset;
        metallicRoughnessTexture = (float3 &)tex2D<float4>(mat.metallicRoughnessTexture, tempTc.x, tempTc.y);
        occlusion *= metallicRoughnessTexture.x;
        roughness *= metallicRoughnessTexture.y;
        if (metallicRoughnessTexture.z < 0.04f) // threshold to fix incorrect textures
            metallic = 0.0f;
        else
            metallic *= metallicRoughnessTexture.z;

        if (prd.debug) {
            printf("orm=(%f, %f, %f)\n", metallicRoughnessTexture.x, metallicRoughnessTexture.y,
                metallicRoughnessTexture.z);
        }
    }

    float3 emission = mat.emission;
    float3 emissiveTexture = make_float3(1.0f);
    if (mat.hasEmissiveTexture) {
        float2 tempTc = tc;
        tempTc *= mat.emissiveTexTf.scale;
        tempTc += mat.emissiveTexTf.offset;
        emissiveTexture = (float3 &)tex2D<float4>(mat.emissiveTexture, tempTc.x, tempTc.y);
    }
    emission *= emissiveTexture;

    float3 sheenColour = mat.sheenColour;
    float3 sheenColourTexture = make_float3(1.0f);
    if (mat.hasSheenColourTexture) {
        float2 tempTc = tc;
        tempTc *= mat.sheenColourTexTf.scale;
        tempTc += mat.sheenColourTexTf.offset;
        sheenColourTexture = (float3 &)tex2D<float4>(mat.sheenColourTexture, tempTc.x, tempTc.y);
    }
    sheenColour *= sheenColourTexture;

    float sheenRoughness = mat.sheenRoughness;
    float4 sheenRoughnessTexture = make_float4(1.0f);
    if (mat.hasSheenRoughnessTexture) {
        float2 tempTc = tc;
        tempTc *= mat.sheenRoughnessTexTf.scale;
        tempTc += mat.sheenRoughnessTexTf.offset;
        sheenRoughnessTexture = tex2D<float4>(mat.sheenRoughnessTexture, tempTc.x, tempTc.y);
    }
    sheenRoughness *= sheenRoughnessTexture.w;

    float categoryWeight = 1.0f;
    unsigned int category = categoriseMaterial(prd, mat, -rayDir, normal, albedo, roughness, metallic, emission,
        sheenColour, sheenRoughness, categoryWeight);

    /*
    if (prd.debug) {
        //printf("metallic=%f\n", metallic);
        //printf("roughness=%f\n", roughness);
        //printf("mat.roughness=%f\n", mat.roughness);
        printMaterialFlags(category);
    }
    */

    bool specularHit = category & MaterialFlags::Specular;

    //if (prd.debug) {
    //    printf("  specularHit     = %d\n", specularHit);
    //    printf("  matType         = %08X\n", mat.type);
    //    printf("  matCategory     = %08X\n", mat.category);
    //}

    if (params.vis != 0) {
        switch (params.vis) {
        case 1:
            prd.radiance = make_float3(metallic);
            break;

        case 2:
            prd.radiance = make_float3(roughness);
            break;

        case 3:
            prd.radiance = make_float3(occlusion);
            break;

        case 4:
            prd.radiance = emission;
            break;

        case 5:
            prd.radiance = normal * 0.5f + 0.5f;
            break;

        case 6:
            prd.radiance = Ng * 0.5f + 0.5f;
            break;

        case 7:
            prd.radiance = albedo;
            break;

        case 8: // f0 "glossy colour"
            //prd.radiance = lerp(make_float3(0.04f), albedo, metallic);
            //if (category & MaterialFlags::Diffuse)
            //    prd.radiance = make_float3(mat.specular);
            //else
            //    prd.radiance = make_float3(mat.specular) * albedo;
            prd.radiance = metallic * albedo * make_float3(mat.specular);
            break;

        case 9: // specular colour
            // To do:
            //if (category & MaterialFlags::Specular)
            //    prd.radiance = make_float3(mat.specular);
            //else
            //    prd.radiance = make_float3(0.0f);
            break;

        case 10: // sheen colour
            prd.radiance = sheenColour;
            //prd.radiance = albedo * sheenColour;
            break;

        case 11: // sheen roughness
            prd.radiance = make_float3(sheenRoughness);
            break;

        case 12: // alpha
            prd.radiance = make_float3(alpha);
            break;
        }

        prd.done = true;
        return;
    }

    if (!specularHit) {
        // Shift the hit point slightly to avoid the shadow terminator problem.
        // Nudging in the -rayDir direction works well for both reflection and
        // refraction.
        hitPoint -= rayDir * SHADOW_BIAS;
    }

    if (category & MaterialFlags::Emission) {
        //if (prd.debug)
        //    printf("  Emitter hit!\n");

        if (dot(rayDir, normal) < 0.0f && (prd.depth == 1 || prd.specularHit)) {
            // Assuming emission contains radiance.
            float3 radiance = emission;
            prd.radiance += radiance + albedo * prd.throughput;

            //if (prd.debug)
            //    printf("Emitter radiance = (%f, %f, %f\n", radiance.x, radiance.y, radiance.z);
        }

        prd.done = true;
        return;
    }

    prd.specularHit = specularHit;

    MaterialMetallicRoughness matMR;
    matMR.baseColour = albedo;
    matMR.ior = mat.ior;
    matMR.metallic = metallic;
    matMR.specular = mat.specular;
    matMR.transmissionWeight = mat.transmissionWeight;
    matMR.subsurfaceWeight = mat.subsurfaceWeight;
    matMR.coatWeight = mat.coatWeight;
    matMR.sheenColour = sheenColour;
    matMR.type = mat.type;
    matMR.category = category;
    matMR.shadingNormal = normal;
    matMR.geometricNormal = Ng;
    matMR.faceForwardNormal = ffnormal;

    if (category & MaterialFlags::Coat) {
        matMR.roughness = mat.coatRoughness;
        matMR.alpha = mat.coatRoughness * mat.coatRoughness;
    }
    else if (category & MaterialFlags::Sheen) {
        matMR.roughness = mat.sheenRoughness;
        matMR.alpha = mat.sheenRoughness * mat.sheenRoughness;
    }
    else {
        matMR.roughness = roughness;
        matMR.alpha = roughness * roughness;
    }
    matMR.alpha2 = matMR.alpha * matMR.alpha;

    // Assuming we have either light sources or emissive triangles (not both at
    // the same time).

    if (!specularHit && params.numLights != 0) {
        // Compute direct lighting.

        int lightIndex = prd.random.rng() * params.numLights; // pick a light source at random
        Light light = params.lights[lightIndex];
        float lightSelectionPdf = 1.0f / params.numLights;

        LightSample sample;
        optixDirectCall<void, Light &, PRNG &, const float3 &, LightSample &>(0, light, prd.random, hitPoint, sample);

        float cosThetaX = fmax(dot(normal, sample.lightDir), 0.0f);

        // Fire a shadow ray potentially calling the anyhit program where
        // lightVisibility is set to (0.0f, 0.0f, 0.0f).

        ShadowRayData srd;
        srd.lightVisibility = make_float3(1.0f);
        srd.xi = prd.random.rng();
        uint32_t u0, u1;
        packPointer(&srd, u0, u1);

        optixTrace(params.traversable, hitPoint, sample.lightDir, SCENE_EPSILON, sample.lightDist - SCENE_EPSILON, 0.0f,
            OptixVisibilityMask(1), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            (int)RayType::Shadow, (int)RayType::Count, (int)RayType::Shadow, u0, u1);

        float3 brdf = metallicRoughnessBrdf(prd, matMR, sample.lightDir, -rayDir);
        float3 colour = srd.lightVisibility * brdf * sample.radiance * cosThetaX / sample.pdf / lightSelectionPdf;
        prd.radiance += (colour * prd.throughput);

        /*
        if (prd.debug) {
            printf("Direct light:\n");
            printf("  alphaMode=%d alphaCutoff=%f\n", mat.alphaMode, mat.alphaCutoff);
            printf("  mat.baseColour.w=%f alpha=%f\n", mat.baseColour.w, alpha);
            //printf("sample.radiance=(%f, %f, %f)\n", sample.radiance.x, sample.radiance.y, sample.radiance.z);
        }
        */
    }

    // Using emissive triangles.
    if (!specularHit && params.numEmissiveTriangles != 0)
    {
        // Compute direct lighting.

        // Pick a random emissive triangle.
        const int lightIndex = prd.random.rng() * params.numEmissiveTriangles;
        const float lightSelectionPdf = 1.0f / params.numEmissiveTriangles;

        //printf("0: (%f, %f, %f)\n", sbtData.emissiveTriVertices[0].x, sbtData.emissiveTriVertices[0].y,
        //    sbtData.emissiveTriVertices[0].z);

        // + Pick a random position on the emissive triangle.
        float2 xi = prd.random.rng2();
        if (xi.x + xi.y > 1.0f) {
            xi.x = 1.0f - xi.x;
            xi.y = 1.0f - xi.y;
        }
        const float3 p0 = sbtData.emissiveTriVertices[lightIndex * 3 + 0];
        const float3 p1 = sbtData.emissiveTriVertices[lightIndex * 3 + 1];
        const float3 p2 = sbtData.emissiveTriVertices[lightIndex * 3 + 2];
        const float3 v1 = p1 - p0;
        const float3 v2 = p2 - p0;
        const float3 lightPos = p0 + v1 * xi.x + v2 * xi.y;
        // -

        float u, v, w;
        calcBarycentric(p0, p1, p2, lightPos, u, v, w);
        
        const float2 tc = w * sbtData.emissiveTriTexCoords[lightIndex * 3 + 0] +
                          u * sbtData.emissiveTriTexCoords[lightIndex * 3 + 1] +
                          v * sbtData.emissiveTriTexCoords[lightIndex * 3 + 2];

        // Calculate the unnormalised normal vector.
        // The magnitude of this vector is the area of the parallelogram defined
        // by the vectors v1 and v2.
        const float3 lightNormalVec = cross(v1, v2);

        const float3 lightNormal = normalize(lightNormalVec);
        
        const float lightSamplePdf = 1.0f / (length(lightNormalVec) * 0.5f); // one over area

        const float3 lightVec = lightPos - hitPoint; // the "to the light" vector
        const float lightDist = length(lightVec);
        const float3 lightDir = lightVec / lightDist; // normalisation

        const float cosThetaY = fmaxf(0.0f, dot(lightNormal, -lightDir));
        float3 lightRadiance = sbtData.emissiveTriRadiances[lightIndex];

        float3 emissiveTexture = make_float3(1.0f);
        int matIndex = sbtData.emissiveTriMatIndices[lightIndex];
        if (sbtData.matData[matIndex].hasEmissiveTexture) {
            float2 tempTc = tc;
            tempTc *= mat.emissiveTexTf.scale;
            tempTc += mat.emissiveTexTf.offset;
            emissiveTexture = (float3 &)tex2D<float4>(sbtData.matData[matIndex].emissiveTexture, tempTc.x, tempTc.y);
        }
        lightRadiance *= emissiveTexture;

        const float cosThetaX = fmaxf(0.0f, dot(normal, lightDir));
        const float gTerm = cosThetaX * cosThetaY / (lightDist * lightDist);

        // Fire a shadow ray potentially calling the anyhit program where
        // lightVisibility is set to (0.0f, 0.0f, 0.0f).

        ShadowRayData srd;
        srd.lightVisibility = make_float3(1.0f);
        srd.xi = prd.random.rng();
        uint32_t u0, u1;
        packPointer(&srd, u0, u1);

        optixTrace(params.traversable, hitPoint, lightDir, SCENE_EPSILON, lightDist - SCENE_EPSILON, 0.0f,
            OptixVisibilityMask(1), OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT,
            (int)RayType::Shadow, (int)RayType::Count, (int)RayType::Shadow, u0, u1);

        float3 brdf = metallicRoughnessBrdf(prd, matMR, lightDir, -rayDir);
        float3 colour = srd.lightVisibility * lightRadiance * brdf * gTerm / (lightSamplePdf * lightSelectionPdf);
        prd.radiance += colour * prd.throughput;

        /*
        if (prd.debug) {
            printf("Direct light:\n");
            printf("  hitPoint          = (% 6.4f, % 6.4f, % 6.4f)\n", hitPoint.x, hitPoint.y, hitPoint.z);
            printf("  normal            = (% 6.4f, % 6.4f, % 6.4f)\n", normal.x, normal.y, normal.z);
            printf("  ffnormal          = (% 6.4f, % 6.4f, % 6.4f)\n", ffnormal.x, ffnormal.y, ffnormal.z);
            printf("  lightPos          = (% 6.4f, % 6.4f, % 6.4f)\n", lightPos.x, lightPos.y, lightPos.z);
            printf("  lightNormal       = (% 6.4f, % 6.4f, % 6.4f)\n", lightNormal.x, lightNormal.y, lightNormal.z);
            printf("  lightDist         = %f\n", lightDist);
            printf("  lightDir          = (% 6.4f, % 6.4f, % 6.4f)\n", lightDir.x, lightDir.y, lightDir.z);
            printf("  nDotL             = %f\n", cosThetaX);
            printf("  lightVisibility   = (%f, %f, %f)\n", srd.lightVisibility.x, srd.lightVisibility.y, srd.lightVisibility.z);
            printf("  lightRadiance     = (%f, %f, %f)\n", lightRadiance.x, lightRadiance.y, lightRadiance.z);
            printf("  gTerm             = %f\n", gTerm);
            printf("  lightSamplePdf    = %f\n", lightSamplePdf);
            printf("  lightSelectionPdf = %f\n", lightSelectionPdf);
            printf("  brdf              = (%f, %f, %f)\n", brdf.x, brdf.y, brdf.z);
            printf("  colour            = (%f, %f, %f)\n", colour.x, colour.y, colour.z);
            float3 contrib = colour * prd.throughput;
            printf("  radiance contrib  = (%f, %f, %f)\n", contrib.x, contrib.y, contrib.z);
        }
        */
    }

    if (prd.depth == MAX_DEPTH) {
        prd.done = true;
        return;
    }

    BsdfSample bsdfSample;

    float3 xi = prd.random.rng3();
    metallicRoughnessSample(prd, matMR, -rayDir, xi, bsdfSample);
    bsdfSample.pdf *= categoryWeight;
    
    //if (prd.debug)
    //    printf("  metallic xi     = (%f, %f, %f)\n", xi.x, xi.y, xi.z);

    /*
    if (prd.debug) {
        printf("Surface sampling:\n");
        //printf("  selectionPdf    = %f\n", selectionPdf);
        printf("  ffnormal        = (%f, %f, %f)\n", ffnormal.x, ffnormal.y, ffnormal.z);
        printf("  ray dir         = (%f, %f, %f)\n", rayDir.x, rayDir.y, rayDir.z);
        printf("  new ray dir     = (%f, %f, %f)\n", bsdfSample.newRayDir.x, bsdfSample.newRayDir.y, bsdfSample.newRayDir.z);
        printf("  bsdf            = (%f, %f, %f)\n", bsdfSample.bsdf.x, bsdfSample.bsdf.y, bsdfSample.bsdf.z);
        printf("  ray dir pdf     = %f\n", bsdfSample.pdf);
    }
    */

    if (bsdfSample.pdf < SCENE_EPSILON || fmaxf(bsdfSample.bsdf) < SCENE_EPSILON) {
        prd.done = true;
        return;
    }

    // fabs(dot(normal, bsdfSample.newRayDir)) is used here instead of
    // fmaxf(0.0f, dot(normal, bsdfSample.newRayDir)) to support transmissive objects.
    //prd.throughput *= bsdfSample.bsdf * fabs(dot(normal, bsdfSample.newRayDir)) / bsdfSample.pdf / selectionPdf;
    prd.throughput *= bsdfSample.bsdf * fabs(dot(normal, bsdfSample.newRayDir)) / bsdfSample.pdf;
    //if (prd.debug)
    //    printf("  throughput      = (%f, %f, %f)\n", prd.throughput.x, prd.throughput.y, prd.throughput.z);

    float maxComponent = fmaxf(prd.throughput);
    if (prd.depth > 3 || maxComponent < 0.3f) {
        // Use Russian roulette to terminate the path.
        float q = fmaxf(0.05f, 1.0f - maxComponent);
        float xi = prd.random.rng();
        if (xi < q)
            prd.done = true;
        else
            prd.throughput /= (1.0f - q);
    }

    if (!prd.done) {
        if (!specularHit) {
            // Reverse shadow bias.
            hitPoint += rayDir * SHADOW_BIAS;
        }

        // Nudge in the new direction.
        prd.origin = hitPoint + bsdfSample.newRayDir * SCENE_EPSILON;

        prd.dir = bsdfSample.newRayDir;

        //if (prd.debug) {
        //    printf("  new ray origin  = (%f, %f, %f)\n", prd.origin.x, prd.origin.y, prd.origin.z);
        //    printf("  new ray dir     = (%f, %f, %f)\n", prd.dir.x, prd.dir.y, prd.dir.z);
        //}
    }
}
