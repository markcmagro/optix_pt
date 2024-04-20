#pragma once

#include <cuda_runtime.h>
#include <optix.h>
#include <sutil/vec_math.h>

// The values of enum LightType are also indices into a buffer of light sampling
// program IDs, so they need to start from zero and be in sequence. With these
// restrictions in place, the last enum value (NumLightTypes) will automatically
// store the number of values in the enum.
enum class LightType
{
    PointLight,
    DirectionalLight,
    SpotLight,
    NumLightTypes
};

struct Light
{
    LightType type;

    // If type is Point, this value stores intensity instead of radiance.
    float3 radiance; // all

    float3 colour;

    // Power in Watts (a float) multiplied by the light colour (a float3).
    float3 power;    // all

    float3 p0;       // all
    float3 p1;       // Quad (anchor), Triangle, Directional (anchor)
    float3 p2;       // Quad, Triangle, Directional

    float3 normal;   // Quad, Triangle, Directional, Spot

    float3 v1;       // Quad (p0 - p1), Directional (p0 - p1)
    float3 v2;       // Quad (p2 - p1), Directional (p0 - p1)

    float area;             // Quad, Sphere, Triangle
    float radius;           // Sphere
    float distance;         // Directional

    float widthRad;         // SpotLight
    float cosWidth;         // SpotLight (cos of angle with the cone axis)
    float cosFalloffStart;  // SpotLight (cos of angle with the cone axis)

#ifndef NO_OPTIX
#ifndef __CUDACC__
    void createPointLight(const float3 &origin, float wattage, const float3 &colour = make_float3(1.0f));

    void createDirectionalLight(const float3 &p0, const float3 &p1, const float3 &p2, const float3 &direction,
        float distance, float wattage, const float3 &colour = make_float3(1.0f));

    void createSpotLight(const float3 &origin, const float3 &lookAt, float widthDeg, float falloffStartDeg,
        float wattage, const float3 &colour = make_float3(1.0f));
#endif
#endif
};

struct LightSample
{
    float3 position;
    float3 normal;
    float3 lightDir;
    float3 radiance;
    float3 dir;   // random dir from light, spotlight only
    float pdf;    // position PDF
    float dirPdf; // direction PDF, spotlight only
    float lightDist;
    float radianceScalingFactor;
};
