#pragma once

#include <optix.h>
#include "material_data.h"

struct RayGenData
{
    float3 *emissiveTriVertices;
    float3 *emissiveTriRadiances;
};

struct MissData
{
    float3 colour;
};

struct HitGroupData
{
    float3 *normals;
    float2 *texCoords;
    int3 *triIndices;
    int *matIndices;
    MaterialData *matData;
    int triOffset;

    float3 *emissiveTriVertices;
    float2 *emissiveTriTexCoords;
    float3 *emissiveTriRadiances;
    int *emissiveTriMatIndices;
};

struct CallablesData
{
};

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

typedef SbtRecord<RayGenData> RayGenSbtRecord;
typedef SbtRecord<MissData> MissSbtRecord;
typedef SbtRecord<HitGroupData> HitGroupSbtRecord;
typedef SbtRecord<CallablesData> CallablesSbtRecord;
