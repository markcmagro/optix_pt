#include <iostream>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <device_launch_parameters.h>
#include "cuda_helpers.h"
#include "kernel_helpers.h"

#include "stb/stb_image_write.h"

#undef __CUDACC__ // to avoid conflicts with the half type
#include <ImfRgbaFile.h>
#include <ImfArray.h>

#define M 512

// From https://github.com/NVIDIA/cuda-samples/blob/master/Common/helper_math.h.
inline __device__ __host__ float clamp(float f, float a, float b)
{
    return fmaxf(a, fminf(f, b));
}

__global__ void processImage(float4 *srcImage, uint32_t *dstImage, int n)
{
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < n) {
        // Clamp values to cater for input values out of the [0, 1] range.
        const int r = int(clamp(255.99f * srcImage[index].x, 0.0f, 255.0f));
        const int g = int(clamp(255.99f * srcImage[index].y, 0.0f, 255.0f));
        const int b = int(clamp(255.99f * srcImage[index].z, 0.0f, 255.0f));

        const uint32_t rgba = 0xff000000 | (r << 0) | (g << 8) | (b << 16);
        dstImage[index] = rgba;
    }
}

void dumpFbToPng(float *fb, int fbWidth, int fbHeight)
{
    uint32_t *dstImage;
    cudaMalloc((void **)&dstImage, fbWidth * fbHeight * sizeof(uint32_t));

    uint32_t *hostDstImage;
    hostDstImage = new uint32_t[fbWidth * fbHeight];

    if (dstImage && hostDstImage) {
        int n = fbWidth * fbHeight;
        processImage<<<(n + M - 1) / M, M>>>((float4 *)fb, dstImage, n);

        CUDA(cudaMemcpy(hostDstImage, dstImage, fbWidth * fbHeight * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        std::cout << "Dumping framebuffer ...\n";
        if (stbi_write_png("output.png", (int)fbWidth, (int)fbHeight, 4, hostDstImage, (int)(fbWidth * sizeof(uint32_t))) == 0)
            std::cerr << "stbi_write_png() failed!" << std::endl;
    }

    delete [] hostDstImage;
    cudaFree(dstImage);
}

void dumpAccumFbToOpenExr(float *hostAccumFb, int fbWidth, int fbHeight)
{
    int numPixels = fbWidth * fbHeight;
    Imf::Rgba *rgba = new Imf::Rgba[numPixels];

    for (int i = 0; i < numPixels; ++i)
        rgba[i] = Imf::Rgba(hostAccumFb[i * 4 + 0], hostAccumFb[i * 4 + 1], hostAccumFb[i * 4 + 2], 1.0f);

    try {
        Imf::RgbaOutputFile file("output.exr", fbWidth, fbHeight, Imf::WRITE_RGBA);
        file.setFrameBuffer(rgba, 1, fbWidth);
        file.writePixels(fbHeight);
    } catch (const std::exception &e) {
        std::cerr << "Error writing image file output.exr:" << e.what() << std::endl;
    }

    delete [] rgba;
}

__global__ void initRandStateKernel(int maxX, int maxY, curandState *randState)
{
    int pixelIndex = threadIdx.x + blockIdx.x * blockDim.x;
    if (pixelIndex >= maxX * maxY)
        return;

    // Using the same seed and a different sequence number for each thread.
    curand_init(0, pixelIndex, 0, &randState[pixelIndex]);
}

void initRandState(int maxX, int maxY, curandState *randState)
{
    // There are 32 threads in one warp.
    // Try to use block sizes that are multiples of 32.

    // Max 1024 threads per block
    // Max block dimensions are 1024 x 1024 x 64; x * y * z <= 1024
    // Max Block grid dimensions are 2^31 - 1, 64K, 64K

    // 1D: threadId = blockIdx.x * blockDim.x + threadIdx.x

    // 2D: blockId  = blockIdx.x + blockIdx.y * gridDim.x
    //     threadId = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x

    //dim3 blocks;  // thread organisation within the grid
    //dim3 threads; // thread origanisaion within a block
    // <<<blocks, threads per block>>>

    //std::printf("initRandState()\n");
    //std::printf("  maxX=%d maxY=%d\n", maxX, maxY);

    // Using 1D indexing. blocks=4050; threads=512.
    initRandStateKernel<<<((maxX * maxY) + M - 1) / M, M>>>(maxX, maxY, randState);
}
