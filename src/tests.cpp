//#include "principled_bsdf.h" // for testMetallicRoughness()
//#include "schlick.h"

/*
void linear_RGB_to_sRGB01(float r_l, float g_l, float b_l, float *r, float *g, float *b)
{
    if (r_l <= 0.0031308f)
        *r = 12.92f * r_l;
    else
        *r = 1.055f * powf(r_l, 1.0f / 2.4f) - 0.055f;

    if (g_l <= 0.0031308f)
        *g = 12.92f * g_l;
    else
        *g = 1.055f * powf(g_l, 1.0f / 2.4f) - 0.055f;

    if (b_l <= 0.0031308f)
        *b = 12.92f * b_l;
    else
        *b = 1.055f * powf(b_l, 1.0f / 2.4f) - 0.055f;

    *r = clamp(*r, 0.0f, 1.0f);
    *g = clamp(*g, 0.0f, 1.0f);
    *b = clamp(*b, 0.0f, 1.0f);
}

void sRGB01_to_sRGB8(float r, float g, float b, uint8_t *r8, uint8_t *g8, uint8_t *b8)
{
    *r8 = r * 255.0f + 0.5f;
    *g8 = g * 255.0f + 0.5f;
    *b8 = b * 255.0f + 0.5f;
}

void sRGB8_to_sRGB01(uint8_t r8, uint8_t g8, uint8_t b8, float *r, float *g, float *b)
{
    *r = r8 / 255.0f;
    *g = g8 / 255.0f;
    *b = b8 / 255.0f;
}

void sRGB01_to_linear_RGB(float r, float g, float b, float *r_l, float *g_l, float *b_l)
{
    if (r <= 0.04045f)
        *r_l = r / 12.92f;
    else
        *r_l = powf((r + 0.055f) / 1.055f, 2.4f);

    if (g <= 0.04045f)
        *g_l = g / 12.92f;
    else
        *g_l = powf((g + 0.055f) / 1.055f, 2.4f);

    if (b <= 0.04045f)
        *b_l = b / 12.92f;
    else
        *b_l = powf((b + 0.055f) / 1.055f, 2.4f);
}

void testColourTransferFunctions()
{
    float r_l, g_l, b_l;
    float r, g, b;
    uint8_t r8, g8, b8;

    r_l = 0.0f;
    g_l = 0.0f;
    b_l = 0.0f;
    linear_RGB_to_sRGB01(r_l, g_l, b_l, &r, &g, &b);
    std::printf("Linear (%f, %f, %f) -> sRGB01 (%f, %f, %f)\n", r_l, g_l, b_l, r, g, b);
    sRGB01_to_sRGB8(r, g,b, &r8, &g8, &b8);
    std::printf("sRGB01 (%f, %f, %f) -> sRGB8  (%u, %u, %u)\n", r, g, b, r8, g8, b8);
    sRGB8_to_sRGB01(r8, g8, b8, &r, &g, &b);
    std::printf("sRGB8  (%u, %u, %u) -> sRGB01 (%f, %f, %f)\n", r8, g8, b8, r, g, b);
    sRGB01_to_linear_RGB(r, g, b, &r_l, &g_l, &b_l);
    std::printf("sRGB01 (%f, %f, %f) -> Linear (%f, %f, %f)\n\n", r, g, b, r_l, g_l, b_l);

    r_l = 0.1f;
    g_l = 0.1f;
    b_l = 0.1f;
    linear_RGB_to_sRGB01(r_l, g_l, b_l, &r, &g, &b);
    std::printf("Linear (%f, %f, %f) -> sRGB01 (%f, %f, %f)\n", r_l, g_l, b_l, r, g, b);
    sRGB01_to_sRGB8(r, g,b, &r8, &g8, &b8);
    std::printf("sRGB01 (%f, %f, %f) -> sRGB8  (%u, %u, %u)\n", r, g, b, r8, g8, b8);
    sRGB8_to_sRGB01(r8, g8, b8, &r, &g, &b);
    std::printf("sRGB8  (%u, %u, %u) -> sRGB01 (%f, %f, %f)\n", r8, g8, b8, r, g, b);
    sRGB01_to_linear_RGB(r, g, b, &r_l, &g_l, &b_l);
    std::printf("sRGB01 (%f, %f, %f) -> Linear (%f, %f, %f)\n\n", r, g, b, r_l, g_l, b_l);

    r_l = 0.5f;
    g_l = 0.5f;
    b_l = 0.5f;
    linear_RGB_to_sRGB01(r_l, g_l, b_l, &r, &g, &b);
    std::printf("Linear (%f, %f, %f) -> sRGB01 (%f, %f, %f)\n", r_l, g_l, b_l, r, g, b);
    sRGB01_to_sRGB8(r, g,b, &r8, &g8, &b8);
    std::printf("sRGB01 (%f, %f, %f) -> sRGB8  (%u, %u, %u)\n", r, g, b, r8, g8, b8);
    sRGB8_to_sRGB01(r8, g8, b8, &r, &g, &b);
    std::printf("sRGB8  (%u, %u, %u) -> sRGB01 (%f, %f, %f)\n", r8, g8, b8, r, g, b);
    sRGB01_to_linear_RGB(r, g, b, &r_l, &g_l, &b_l);
    std::printf("sRGB01 (%f, %f, %f) -> Linear (%f, %f, %f)\n\n", r, g, b, r_l, g_l, b_l);

    r_l = 0.9f;
    g_l = 0.9f;
    b_l = 0.9f;
    linear_RGB_to_sRGB01(r_l, g_l, b_l, &r, &g, &b);
    std::printf("Linear (%f, %f, %f) -> sRGB01 (%f, %f, %f)\n", r_l, g_l, b_l, r, g, b);
    sRGB01_to_sRGB8(r, g,b, &r8, &g8, &b8);
    std::printf("sRGB01 (%f, %f, %f) -> sRGB8  (%u, %u, %u)\n", r, g, b, r8, g8, b8);
    sRGB8_to_sRGB01(r8, g8, b8, &r, &g, &b);
    std::printf("sRGB8  (%u, %u, %u) -> sRGB01 (%f, %f, %f)\n", r8, g8, b8, r, g, b);
    sRGB01_to_linear_RGB(r, g, b, &r_l, &g_l, &b_l);
    std::printf("sRGB01 (%f, %f, %f) -> Linear (%f, %f, %f)\n\n", r, g, b, r_l, g_l, b_l);

    r_l = 1.0f;
    g_l = 1.0f;
    b_l = 1.0f;
    linear_RGB_to_sRGB01(r_l, g_l, b_l, &r, &g, &b);
    std::printf("Linear (%f, %f, %f) -> sRGB01 (%f, %f, %f)\n", r_l, g_l, b_l, r, g, b);
    sRGB01_to_sRGB8(r, g,b, &r8, &g8, &b8);
    std::printf("sRGB01 (%f, %f, %f) -> sRGB8  (%u, %u, %u)\n", r, g, b, r8, g8, b8);
    sRGB8_to_sRGB01(r8, g8, b8, &r, &g, &b);
    std::printf("sRGB8  (%u, %u, %u) -> sRGB01 (%f, %f, %f)\n", r8, g8, b8, r, g, b);
    sRGB01_to_linear_RGB(r, g, b, &r_l, &g_l, &b_l);
    std::printf("sRGB01 (%f, %f, %f) -> Linear (%f, %f, %f)\n\n", r, g, b, r_l, g_l, b_l);
}
*/

/*
void testMetallicRoughness()
{
    float3 albedo = make_float3(1.0f);
    float metallic = 0.0f;
    float roughness = 0.0f;
    float3 normal = make_float3(-0.570125, 0.602392f, 0.558642f);
    float3 lightDir = make_float3(0.0f, 1.0f, 0.0f);
    float3 rayDir = make_float3(0.075302f, -0.078242f, -0.994086f);
    float3 xi = make_float3(0.736983f, 0.261954f, 0.133479f);

    //float3 brdf = metallicRoughnessBrdf(albedo, metallic, roughness, normal, lightDir, -rayDir);
    //std::printf("brdf        = (%f, %f, %f)\n", brdf.x, brdf.y, brdf.z);

    float3 newRayDir;
    float3 bsdf;
    float pdf;

    metallicRoughnessSample(normal, -rayDir, albedo, metallic, roughness, xi, newRayDir, bsdf, pdf);

    std::printf("Metallic:\n");
    std::printf("new ray dir = (%f, %f, %f)\n", newRayDir.x, newRayDir.y, newRayDir.z);
    std::printf("bsdf        = (%f, %f, %f)\n", bsdf.x, bsdf.y, bsdf.z);
    std::printf("pdf         = %f\n", pdf);

    newRayDir = cosineSampleHemisphere(make_float2(xi.x, xi.y));
    OrthonormalBasis onb(normal);
    onb.localToWorld(newRayDir);
    float cosTheta = dot(normal, newRayDir);
    pdf = cosTheta * M_1_PIf;
    bsdf = albedo * M_1_PIf;

    std::printf("Diffuse:\n");
    std::printf("new ray dir = (%f, %f, %f)\n", newRayDir.x, newRayDir.y, newRayDir.z);
    std::printf("bsdf        = (%f, %f, %f)\n", bsdf.x, bsdf.y, bsdf.z);
    std::printf("pdf         = %f\n", pdf);
}
*/

/*
void testFresnel()
{
    std::printf("Fresnel (Schlick)\n");
    std::printf("-----------------\n");

    float colours[] = { 0.0f, 0.25f, 0.5f, 0.75f, 1.0f };
    float angles[] = { 0.0f, 15.0f, 30.0f, 60.0f, 75.0f, 90.0f };

    int numColours = sizeof(colours) / sizeof(colours[0]);
    int numAngles = sizeof(angles) / sizeof(angles[0]);

    for (int i = 0; i < numColours; ++i) {
        for (int j = 0; j < numAngles; ++j) {
            float cosTheta = cosf(glm::radians(angles[j]));
            float F = fresnelSchlick(colours[i], cosTheta);
            std::printf("colour=%.2f; angle=%4.1f; F=%.4f\n", colours[i], angles[j], F);
        }
        std::printf("--------\n");
    }
}
*/

/*
float2 testReflect(float2 n, float2 wi)
{
    std::printf("  Reflect()\n");
    std::printf("  n=(%f, %f)\n", n.x, n.y);
    std::printf("  wi=(%f, %f)\n", wi.x, wi.y);

    float2 wr = -wi + 2.0f * dot(n, wi) * n;
    std::printf("  wr=(%f, %f)\n", wr.x, wr.y);
    std::printf("  norm(wr)=%f\n", length(wr));

    return wr;
}

void testReflection()
{
    float2 n = make_float2(0.0, 1.0f);
    float2 wi = normalize(make_float2(-1.0f, 1.0f)); // incident ray (inverted direction)
    testReflect(n, wi);

    n = -n;
    testReflect(n, wi);
}
*/

/*
bool testRefract(float2 n, float2 wi, float eta, float2 *wt)
{
    // See https://www.pbr-book.org/4ed/Reflection_Models/Specular_Reflection_and_Transmission.
    
    std::printf("  Refract()\n");
    std::printf("  n=(%f, %f)\n", n.x, n.y);
    std::printf("  wi=(%f, %f)\n", wi.x, wi.y);
    std::printf("  eta=%f\n", eta);

    // + Calculate cosThetaI.
    float cosThetaI = dot(n, wi);
    // -
    // + Debug output.
    float thetaIRad = acosf(cosThetaI);
    std::printf("  thetaI=%.2f degrees\n", glm::degrees(thetaIRad));
    // -

    if (cosThetaI >= 0.0f) { // ray coming from outside
        std::printf("  Outside\n");
    }
    else { // ray coming from inside
        std::printf("  Inside (flipping n, eta, cosThetaI)\n");

        cosThetaI = -cosThetaI;
        n = -n;
        eta = 1.0f / eta;

        std::printf("  n=(%f, %f)\n", n.x, n.y);
        std::printf("  eta=%f\n", eta);
        thetaIRad = acosf(cosThetaI);
        std::printf("  thetaI=%.2f degrees\n", glm::degrees(thetaIRad));
    }

    // + Calculate cosThetaT (using sin, asin, cos).
    //float sinThetaI = sinf(thetaIRad);
    //float sinThetaT = sinThetaI / eta; // Snell's law
    //float thetaTRad = asinf(sinThetaT);
    //float cosThetaT = cosf(thetaTRad);
    // -
    // + Debug output.
    //std::printf("  thetaT=%.2f degrees\n", glm::degrees(thetaTRad));
    // -

    // + Calculate cosThetaT (using sqrt).
    float sin2ThetaI = fmax(0.0f, 1.0f - (cosThetaI * cosThetaI));
    float sin2ThetaT = sin2ThetaI / (eta * eta); // Snell's law, squaring both sides
    std::printf("  Critical angle=%f degrees\n", glm::degrees(asinf(eta)));
    if (sin2ThetaT >= 1.0f) { // total internal reflection
        std::printf("  Total internal reflection!\n");
        testReflect(n, wi);
        return false;
    }
    float cosThetaT = sqrtf(fmax(0.0f, 1.0f - sin2ThetaT));
    // -
    // + Debug output.
    float thetaTRad = acosf(cosThetaT);
    std::printf("  thetaT=%.2f degrees\n", glm::degrees(thetaTRad));
    // -

    // Calculate the transmitted ray.
    *wt = -wi / eta + (cosThetaI / eta - cosThetaT) * n;

    std::printf("  wt=(%f, %f)\n", wt->x, wt->y);
    std::printf("  norm(wt)=%f\n", length(*wt));

    return true;
}

void testRefraction()
{
    float etaI = 1.0f;
    float etaT = 1.5f;
    float eta = etaT / etaI;
    float2 n = make_float2(0.0, 1.0f);
    float2 wi = normalize(make_float2(-1.0f, 1.0f)); // incident ray (inverted direction)
    float2 wt;

    std::printf("Ray #1\n");
    bool ret = testRefract(n, wi, eta, &wt);
    if (!ret)
        return;

    //std::printf("Ray #2\n");
    //n = normalize(make_float2(0.3f, -0.7f));
    //ret = testRefract(n, -wt, eta, &wi);

    // Total internal reflection.
    std::printf("Ray #2\n");
    n = normalize(make_float2(0.8f, -0.2f));
    ret = testRefract(n, -wt, eta, &wi);
    if (!ret)
        testReflect(n, -wt);
}
*/

/*
void testSnellsLaw()
{
    {
        float etaI = 1.0f;
        float etaT = 1.5f;
        std::printf("etaI=%f; etaT=%f\n", etaI, etaT);

        float thetaIDeg[] = { 0.0f, 5.0f, 15.0f, 30.0f, 45.0f, 60.0f, 75.0f, 85.0f, 90.0f };
        int numAngles = sizeof(thetaIDeg) / sizeof(float);
        for (int i = 0; i < numAngles; ++i) {
            float thetaTDeg = glm::degrees(asinf(etaI / etaT * sinf(glm::radians(thetaIDeg[i]))));
            std::printf("thetaI=%5.2f thetaT=%5.2f\n", thetaIDeg[i], thetaTDeg);
        }

        std::printf("\n");
    }

    {
        float etaI = 1.5f;
        float etaT = 1.0f;
        std::printf("etaI=%f; etaT=%f\n", etaI, etaT);

        float thetaIDeg[] = { 0.0f, 5.0f, 15.0f, 30.0f, 35.0f, 40.0f, 41.0f, 41.5f, 41.6f, 41.7f, 41.8f };
        int numAngles = sizeof(thetaIDeg) / sizeof(float);
        for (int i = 0; i < numAngles; ++i) {
            float thetaTDeg = glm::degrees(asinf(etaI / etaT * sinf(glm::radians(thetaIDeg[i]))));
            std::printf("thetaI=%5.2f thetaT=%5.2f\n", thetaIDeg[i], thetaTDeg);
        }

        std::printf("\n");
    }
}
*/

/*
#define __CUDACC__
#include "material_data.h"

void testStructAlignment()
{
#ifdef __CUDACC__
    std:printf("__CUDACC__ defined\n\n");
#else
    std:printf("__CUDACC__ not defined\n\n");
#endif

    std::printf("MaterialData\n");
    std::printf("------------\n\n");

    std::printf("Size      : %zd (MOD 16 = %zd)\n", sizeof(MaterialData), sizeof(MaterialData) % 16);
    std::printf("Alignment : %zd\n\n", alignof(MaterialData));

    std::printf("Offsets:\n");
    std::printf("MaterialData::baseColourTexture        : %zd\n", offsetof(MaterialData, baseColourTexture));
    std::printf("MaterialData::metallicRoughnessTexture : %zd\n", offsetof(MaterialData, metallicRoughnessTexture));
    std::printf("MaterialData::emissiveTexture          : %zd\n", offsetof(MaterialData, emissiveTexture));
    std::printf("MaterialData::sheenColourTexture       : %zd\n", offsetof(MaterialData, sheenColourTexture));
    std::printf("MaterialData::sheenRoughnessTexture    : %zd\n", offsetof(MaterialData, sheenRoughnessTexture));
    std::printf("MaterialData::emission                 : %zd\n", offsetof(MaterialData, emission));
    std::printf("MaterialData::baseColour               : %zd\n", offsetof(MaterialData, baseColour));
    std::printf("MaterialData::sheenColour              : %zd\n", offsetof(MaterialData, sheenColour));
    std::printf("MaterialData::sheenColourTexTf         : %zd\n", offsetof(MaterialData, sheenColourTexTf));
    std::printf("MaterialData::sheenRoughnessTexTf      : %zd\n", offsetof(MaterialData, sheenRoughnessTexTf));
    std::printf("MaterialData::ior                      : %zd\n", offsetof(MaterialData, ior));
    std::printf("MaterialData::hasSheenRoughnessTexture : %zd\n", offsetof(MaterialData, hasSheenRoughnessTexture));
}
*/

/*
// From https://github.com/NVIDIA/OptiX_Apps/blob/master/apps/MDL_renderer/shaders/transform.h.

// Matrix3x4 * point. v.w == 1.0f
__forceinline__ __device__ float3 transformPoint(const float4 *m, const float3 &v)
{
    float3 r;

    r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z + m[0].w;
    r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z + m[1].w;
    r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z + m[2].w;

    return r;
}

// Matrix3x4 * vector. v.w == 0.0f
float3 transformVector(const float4 *m, const float3 &v)
{
    float3 r;

    r.x = m[0].x * v.x + m[0].y * v.y + m[0].z * v.z;
    r.y = m[1].x * v.x + m[1].y * v.y + m[1].z * v.z;
    r.z = m[2].x * v.x + m[2].y * v.y + m[2].z * v.z;

    return r;
}

void testOptixMatrices()
{
    float tx = 1.0f;
    float ty = 2.0f;
    float tz = 3.0f;

    float translate[12] = { 1.0f, 0.0f, 0.0f, tx, 0.0f, 1.0f, 0.0f, ty, 0.0f, 0.0f, 1.0f, tz };

    float3 v = make_float3(0.0f);
    float3 r = transformPoint((float4 *)translate, v);
    std::printf("(%f, %f, %f) -> (%f, %f, %f)\n", v.x, v.y, v.z, r.x, r.y, r.z);
}
*/
