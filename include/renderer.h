#pragma once

#include <cuda.h>
#include <curand_kernel.h>
#include <optix.h>
#include "camera.h"
#include "launch_params.h"
#include "scene_desc.h"
#include "scene_properties.h"

enum class Module
{
    DevicePrograms,
    LightSampling,
    PathTracer,
    PathTracerNaive
};

class Renderer
{
    SceneDesc &sceneDesc;

    bool useNextEventEstimation; // false => naive path tacer

    bool allowCompaction;
    bool allowUpdate;

    bool initDenoiser;
    bool enableDenoiser;

    CUstream stream;
    OptixDeviceContext optixContext;

    OptixTraversableHandle gasHandle;
    CUdeviceptr outputBuffer;
    size_t outputBufferSize;
    CUdeviceptr compactedBuffer;
    size_t compactedBufferSize;
    CUdeviceptr tempBuffer;
    size_t tempBufferSize;
    CUdeviceptr compactedSizeBuffer;

    OptixModuleCompileOptions moduleCompileOptions;
    OptixPipelineCompileOptions pipelineCompileOptions;
    OptixPipelineLinkOptions pipelineLinkOptions;

    std::vector<OptixModule> modules;

    OptixProgramGroup rayGenPG;
    std::vector<OptixProgramGroup> missPGs;
    std::vector<OptixProgramGroup> hitGroupPGs;
    std::vector<OptixProgramGroup> callablesPGs;

    OptixPipeline pipeline;

    OptixShaderBindingTable sbt {};

    void *fb;
    void *accumFb;
    float *hostFb;
    float *hostAccumFb;
    uint32_t fbWidth;
    uint32_t fbHeight;

    uint32_t sample;
    uint32_t seed;

    bool dumpFb = false;
    bool dumpAccumFbRgbe = false;
    bool dumpAccumFbExr = false;

    LaunchParams launchParamsHost;
    LaunchParams *launchParamsDevice;

    char log[2048];
    size_t logSize;

    SceneProperties properties;

    std::vector<CUdeviceptr> devVerticesColl;
    std::vector<CUdeviceptr> devNormalsColl;
    std::vector<CUdeviceptr> devTexCoordsColl;
    std::vector<CUdeviceptr> devTriIndicesColl;

    CUdeviceptr devMatIndices;
    CUdeviceptr devMatParams;

    std::vector<cudaArray_t> baseColourTextureArrays;
    std::vector<cudaTextureObject_t> baseColourTextureObjects;

    std::vector<cudaArray_t> normalTextureArrays;
    std::vector<cudaTextureObject_t> normalTextureObjects;

    std::vector<cudaArray_t> metallicRoughnessTextureArrays;
    std::vector<cudaTextureObject_t> metallicRoughnessTextureObjects;

    std::vector<cudaArray_t> emissiveTextureArrays;
    std::vector<cudaTextureObject_t> emissiveTextureObjects;

    std::vector<cudaArray_t> sheenColourTextureArrays;
    std::vector<cudaTextureObject_t> sheenColourTextureObjects;

    std::vector<cudaArray_t> sheenRoughnessTextureArrays;
    std::vector<cudaTextureObject_t> sheenRoughnessTextureObjects;

    CUdeviceptr devEmissiveTriVerts;
    CUdeviceptr devEmissiveTriTexCoords;
    CUdeviceptr devEmissiveTriRadiances;
    CUdeviceptr devEmissiveTriMatIndices;

    OptixBuildInput triangleInput;
    std::vector<OptixBuildInput> triangleInputColl;
    uint32_t triangleInputFlags[1];

    OptixDenoiser denoiser;
    CUdeviceptr denoiserState;
    CUdeviceptr denoiserScratch;
    size_t denoiserStateSizeInBytes;
    size_t denoiserScratchSizeInBytes;
    void *denoisedFb;

    curandState *randState;

    //  0 = none
    //  1 = metalness
    //  2 = roughness
    //  3 = occlusion
    //  4 = emission
    //  5 = shading normals
    //  6 = geometric normals
    //  7 = albedo
    //  8 = glossy colour
    //  9 = specular colour
    // 10 = sheen colour
    // 11 = sheen roughness
    // 12 = alpha
    int vis = 0;

    bool toneMapping = true;

    // 0 = Reinhard
    // 1 = ACES (Narkowicz)
    int toneMappingImpl = 1;

    bool gammaCorrection = true;
    float gammaTerm = 2.2f;

public:
    Camera cam;

    Renderer(SceneDesc &sceneDesc);
    ~Renderer();

    void initOptix();
    void createContext();
    void loadScene();
    void buildAccel(bool build = true);
    void setupCompilationOptions();
    void createModules();
    void createProgramGroups();
    void createPipeline();
    void setupSbt();
    void setupOutputBuffer(uint32_t width, uint32_t height);
    void setupCamera(uint32_t width, uint32_t height);
    void setupLaunchParams(SceneDesc &sceneDesc);
    void updateLaunchParams();
    void launch();

    void updateWorldMatrix(size_t nodeIndex, glm::mat4 &globalTf);
    void updateScene();
    void updateAccel();

    void createDenoiser();
    void createRandomState();

    void *getFrameBuffer();
    uint32_t getSampleCount();
    void resetSampleCount();

    void dumpFramebuffer();
    void dumpAccumFramebufferRgbe();
    void dumpAccumFramebufferExr();

    bool getDenoiserState();
    void setDenoiserState(bool on);

    void setVis(int vis);
    int getVis();

    void setToneMappingState(bool enable);
    bool getToneMappingState();

    void setToneMappingImpl(int toneMappingImpl);
    int getToneMappingImpl();

    void setGammaCorrectionState(bool enable);
    int getGammaCorrectionState();

    void setGammaCorrectionTerm(float value);
    float getGammaCorrectionTerm();

private:
    void loadTextures();

    void createProgramGroupsPathTracer();
};
