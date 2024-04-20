#pragma once

#include <optix.h>
#include "launch_params.h"
#include "pointer_packing.h"

template<typename T>
static __forceinline__ __device__ T *getPrd()
{
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return reinterpret_cast<T *>(unpackPointer(u0, u1));
}
