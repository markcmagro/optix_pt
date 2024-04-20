#pragma once

struct MaterialMetallicRoughness
{
    float3 baseColour;
    float ior;
    float roughness; // base layer / coat / sheen
    float metallic;
    float specular; // weight
    float transmissionWeight;
    float subsurfaceWeight;
    float coatWeight;
    float3 sheenColour;
    unsigned int type;
    unsigned int category;

    float3 shadingNormal;
    float3 geometricNormal;
    float3 faceForwardNormal;

    float alpha;
    float alpha2;
};
