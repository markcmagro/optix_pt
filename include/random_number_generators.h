
#pragma once

#include <cuda_runtime.h>
#include <curand_globals.h>
#include <curand_kernel.h>
#include <curand_uniform.h>

#define USE_CURAND

#ifdef USE_CURAND
#define PRNG CURAND
#else
#define PRNG LCG<8>
#endif

template<unsigned int N=8>
struct LCG
{
    unsigned int state;

    inline __host__ __device__ LCG()
    {
    }

    inline __host__ __device__ void init(const unsigned int val0, const unsigned int val1)
    {
        // Tiny Encryption Algorithm (TEA) to calculate a seed per launch index and iteration.

        unsigned int v0 = val0;
        unsigned int v1 = val1;
        unsigned int s0 = 0;

        for (unsigned int n = 0; n < N; ++n)
        {
            s0 += 0x9e3779b9;
            v0 += ((v1 << 4) + 0xA341316C) ^ (v1 + s0) ^ ((v1 >> 5) + 0xC8013EA4);
            v1 += ((v0 << 4) + 0xAD90777D) ^ (v0 + s0) ^ ((v0 >> 5) + 0x7E95761E);
        }

        state = v0;
    }

    // Return a random sample in the range [0, 1) with a simple Linear Congruential Generator.
    inline __host__ __device__ float rng()
    {
        state = state * 1664525u + 1013904223u;
        return (state & 0X00FFFFFF) / (float)0x01000000; // use the lower 24 bits
    }

    // Convenience function to generate 2 random samples in the range [0, 1).
    inline __host__ __device__ float2 rng2()
    {
        return make_float2(rng(), rng());
    }

    // Convenience function to generate 3 random samples in the range [0, 1).
    inline __host__ __device__ float3 rng3()
    {
        return make_float3(rng(), rng(), rng());
    }
};

struct CURAND
{
    curandState state;

    inline __device__ void init(curandState *stateParam)
    {
        state = *stateParam;
    }

    inline __device__ float rng()
    {
        // Return a float in the range (0, 1].
        return curand_uniform(&state);
    }

    inline __device__ float2 rng2()
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        return make_float2(x, y);
    }

    inline __device__ float3 rng3()
    {
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);
        float z = curand_uniform(&state);
        return make_float3(x, y, z);
    }
};
