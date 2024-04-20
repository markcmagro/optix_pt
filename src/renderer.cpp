#include <array>
#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include "cuda_helpers.h"
#include "gltf_loader.h"
#include "kernel_helpers.h"
#include "material_data.h"
#include "mesh.h"
#include "optix7_helpers.h"
#include "renderer.h"
#include "scene_desc.h"
#include "sbt_record.h"
#include "utils.h"

#include "stb/stb_image.h"
#include "stb/stb_image_write.h"

#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

void optixLogCallback(unsigned int level, const char *tag, const char *message, void *data);

Renderer::Renderer(SceneDesc &sceneDesc) :
    sceneDesc(sceneDesc)
{
    useNextEventEstimation = true;

    allowCompaction = false;
    allowUpdate = true;

    initDenoiser = true;
    enableDenoiser = false;

    stream = nullptr;
    optixContext = nullptr;

    outputBuffer = 0;
    outputBufferSize = 0;
    compactedBuffer = 0;
    compactedBufferSize = 0;
    tempBuffer = 0;
    tempBufferSize = 0;
    compactedSizeBuffer = 0;

    rayGenPG = nullptr;
    pipeline = nullptr;

    fb = nullptr;
    accumFb = nullptr;
    hostFb = nullptr;
    hostAccumFb = nullptr;
    fbWidth = 0;
    fbHeight = 0;

    sample = 0;
    seed = 0;

    launchParamsDevice = nullptr;

    devMatIndices = 0;
    devMatParams = 0;

    devEmissiveTriVerts = 0;
    devEmissiveTriTexCoords = 0;
    devEmissiveTriRadiances = 0;
    devEmissiveTriMatIndices = 0;

    denoiser = nullptr;
    denoiserState = 0;
    denoiserScratch = 0;
    denoiserStateSizeInBytes = 0;
    denoiserScratchSizeInBytes = 0;
    denoisedFb = nullptr;

    randState = nullptr;
}

Renderer::~Renderer()
{
    CUDA_NOEXCEPT(cudaFree(launchParamsDevice));

    CUDA_NOEXCEPT(cudaFree(randState));

    if (initDenoiser) {
        CUDA_NOEXCEPT(cudaFree(denoisedFb));
        CUDA_NOEXCEPT(cudaFree((void *)denoiserState));
        CUDA_NOEXCEPT(cudaFree((void *)denoiserScratch));
        if (denoiser)
            OPTIX_NOEXCEPT(optixDenoiserDestroy(denoiser));
    }

    delete [] hostFb;
    delete [] hostAccumFb;
    CUDA_NOEXCEPT(cudaFree(fb));
    CUDA_NOEXCEPT(cudaFree(accumFb));

    CUDA_NOEXCEPT(cudaFree((void *)sbt.raygenRecord));
    CUDA_NOEXCEPT(cudaFree((void *)sbt.missRecordBase));
    CUDA_NOEXCEPT(cudaFree((void *)sbt.hitgroupRecordBase));
    CUDA_NOEXCEPT(cudaFree((void *)sbt.callablesRecordBase));

    CUDA_NOEXCEPT(cudaFree((void *)outputBuffer));
    CUDA_NOEXCEPT(cudaFree((void *)compactedBuffer));
    CUDA_NOEXCEPT(cudaFree((void *)tempBuffer));
    CUDA_NOEXCEPT(cudaFree((void *)compactedSizeBuffer));

    if (pipeline)
        OPTIX_NOEXCEPT(optixPipelineDestroy(pipeline));
    if (rayGenPG)
        OPTIX_NOEXCEPT(optixProgramGroupDestroy(rayGenPG));
    for (auto g : missPGs)
        OPTIX_NOEXCEPT(optixProgramGroupDestroy(g));
    for (auto g : hitGroupPGs)
        OPTIX_NOEXCEPT(optixProgramGroupDestroy(g));
    for (auto g : callablesPGs)
        OPTIX_NOEXCEPT(optixProgramGroupDestroy(g));

    for (auto &m : modules)
        OPTIX_NOEXCEPT(optixModuleDestroy(m));

    for (size_t i = 0; i < properties.mesh.numShapes; ++i) {
        CUDA_NOEXCEPT(cudaFree((void *)devVerticesColl[i]));
        CUDA_NOEXCEPT(cudaFree((void *)devNormalsColl[i]));
        CUDA_NOEXCEPT(cudaFree((void *)devTexCoordsColl[i]));
        CUDA_NOEXCEPT(cudaFree((void *)devTriIndicesColl[i]));
    }

    CUDA_NOEXCEPT(cudaFree((void *)devMatIndices));
    CUDA_NOEXCEPT(cudaFree((void *)devMatParams));

    CUDA_NOEXCEPT(cudaFree((void *)devEmissiveTriVerts));
    CUDA_NOEXCEPT(cudaFree((void *)devEmissiveTriTexCoords));
    CUDA_NOEXCEPT(cudaFree((void *)devEmissiveTriRadiances));
    CUDA_NOEXCEPT(cudaFree((void *)devEmissiveTriMatIndices));

    if (optixContext)
        OPTIX_NOEXCEPT(optixDeviceContextDestroy(optixContext));

    if (stream)
        CUDA_NOEXCEPT(cudaStreamDestroy(stream));
}

void Renderer::initOptix()
{
    // Initialise CUDA subsystem (trigger CUDA context creation and initialisation).
    CUDA(cudaFree(nullptr));

    int cudaDriverVersion;
    cuDriverGetVersion(&cudaDriverVersion);
    int cudaMajVersion = cudaDriverVersion / 1000;
    int cudaMinVersion = (cudaDriverVersion - cudaMajVersion * 1000) / 10;
    std::cout << "CUDA version        : " << cudaMajVersion << "." << cudaMinVersion << std::endl;

    int numCudaDevices;
    CUDA(cudaGetDeviceCount(&numCudaDevices));
    if (numCudaDevices == 0)
        throw std::runtime_error("No CUDA devices found!");
    //std::cout << numCudaDevices << " CUDA device(s) found." << std::endl;

    int computeCapMajVersion;
    int computeCapMinVersion;
    cuDeviceGetAttribute(&computeCapMajVersion, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, 0);
    cuDeviceGetAttribute(&computeCapMinVersion, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, 0);
    std::cout << "Compute capability  : " << computeCapMajVersion << "." << computeCapMinVersion << std::endl;

    OPTIX(optixInit());
    std::cout << "OptiX version       : " << OPTIX_VERSION / 10000 << "." << (OPTIX_VERSION % 10000) / 100 << std::endl;
    std::cout << "OptiX ABI version   : " << OPTIX_ABI_VERSION << std::endl;
}

void Renderer::createContext()
{
    const int deviceId = 0;
    CUDA(cudaSetDevice(deviceId));
    CUDA(cudaStreamCreate(&stream));

    cudaDeviceProp deviceProps;
    CUDA(cudaGetDeviceProperties(&deviceProps, deviceId));
    std::cout << "GPU                 : " << deviceProps.name << std::endl;

    CUcontext cudaContext;
    CUresult ret = cuCtxGetCurrent(&cudaContext);
    if (ret != CUDA_SUCCESS)
        std::cerr << "cuCtxGetCurrent failed. Error " << ret << "." << std::endl;

    OPTIX(optixDeviceContextCreate(cudaContext, nullptr, &optixContext));
    OPTIX(optixDeviceContextSetLogCallback(optixContext, optixLogCallback, nullptr, 4));
}

void Renderer::loadScene()
{
    std::cout << sceneDesc.gltfSceneFile << std::endl;

    GltfLoader gltfLoader(sceneDesc.gltfSceneFile, properties);
    if (!gltfLoader.parse())
        throw std::runtime_error("loadScene() failed!");

    gltfLoader.loadScene();

    //
    // Emission
    //

    // + Initialise properties.emissiveVertexIndices,
    //   properties.emissiveTriRadiances and properties.emissiveTriMatIndices.
    //   properties.emissiveVertices and properties.emissiveTexCoords are
    //   resized here and copied in updateScene().

    std::vector<int> &emissiveVertexIndices = properties.emissiveVertexIndices;
    std::vector<glm::vec3> &emissiveVertices = properties.emissiveVertices;
    std::vector<glm::vec2> &emissiveTexCoords = properties.emissiveTexCoords;
    std::vector<int> &emissiveTriMatIndices = properties.emissiveTriMatIndices;

    for (auto &s : properties.mesh.shapes) {
        for (auto i : s.emissiveTriangles) {
            size_t globalTriangleIndex = s.triangleOffset + i;
            glm::ivec3 &v = properties.mesh.triIndices[globalTriangleIndex];
            emissiveVertexIndices.push_back((int)s.verticesIndex + v.x);
            emissiveVertexIndices.push_back((int)s.verticesIndex + v.y);
            emissiveVertexIndices.push_back((int)s.verticesIndex + v.z);

            // Calculate triangle radiance (assuming diffuse emitters).
            // emission (radiance) = emission colour * emission strength.
            // This value can later be scaled by an emission texture.
            glm::vec3 radiance = properties.mesh.matParams[properties.mesh.matIndices[globalTriangleIndex]].emission;
            properties.emissiveTriRadiances.push_back(radiance);

            properties.emissiveTriMatIndices.push_back(properties.mesh.matIndices[globalTriangleIndex]);
        }
    }

    size_t numEmissiveVertices = emissiveVertexIndices.size();
    std::cout << "Emissive triangles : " << numEmissiveVertices / 3 << std::endl;

    emissiveVertices.resize(numEmissiveVertices);
    emissiveTexCoords.resize(numEmissiveVertices);

    // -

    //
    // Camera
    //

    glm::vec3 origin = gltfLoader.getCamOrigin();
    sceneDesc.camOrigin = make_float3(origin.x, origin.y, origin.z);
    sceneDesc.camUp = sceneDesc.gltfUpHint;
    glm::vec3 target = gltfLoader.getCamTarget();
    sceneDesc.camTarget = make_float3(target.x, target.y, target.z);
    sceneDesc.camFovYDeg = gltfLoader.getCamFovYDeg();
    sceneDesc.camAspectRatio = gltfLoader.getCamAspectRatio();

    //
    // Lights
    //

    for (auto &light : properties.lights)
        sceneDesc.lights.push_back(light);

    //
    // Textures
    //

    loadTextures();
}

void Renderer::buildAccel(bool build)
{
    Mesh &mesh = properties.mesh;

    if (build) {
        triangleInputColl.resize(mesh.numShapes);

        devVerticesColl.resize(mesh.numShapes);
        devNormalsColl.resize(mesh.numShapes);
        devTexCoordsColl.resize(mesh.numShapes);
        devTriIndicesColl.resize(mesh.numShapes);

        size_t positionsIndex = 0;
        size_t normalsIndex = 0;
        size_t texCoordsIndex = 0;
        size_t triIndicesIndex = 0;

        for (size_t i = 0; i < mesh.numShapes; ++i) {
            const size_t numVertices = mesh.shapes[i].numVertices;
            const size_t verticesSize = sizeof(float3) * numVertices;
            CUDA(cudaMalloc((void **)&devVerticesColl[i], verticesSize));
            CUDA(cudaMemcpy((void *)devVerticesColl[i], &mesh.positions[positionsIndex], verticesSize,
                cudaMemcpyHostToDevice));
            positionsIndex += numVertices;

            if (mesh.hasNormals) {
                assert(mesh.shapes[i].numNormals == numVertices);
                const size_t numNormals = numVertices;
                const size_t normalsSize = sizeof(float3) * numNormals;
                CUDA(cudaMalloc((void **)&devNormalsColl[i], normalsSize));
                CUDA(cudaMemcpy((void *)devNormalsColl[i], &mesh.normals[normalsIndex], normalsSize,
                    cudaMemcpyHostToDevice));
                normalsIndex += numNormals;
            }
            else {
                devNormalsColl[i] = 0;
            }

            if (mesh.hasTexCoords) {
                assert(mesh.shapes[i].numTexCoords == numVertices);
                const size_t numTexCoords = numVertices;
                const size_t texCoordsSize = sizeof(float2) * numVertices;
                CUDA(cudaMalloc((void **)&devTexCoordsColl[i], texCoordsSize));
                CUDA(cudaMemcpy((void *)devTexCoordsColl[i], &mesh.texCoords[texCoordsIndex], texCoordsSize,
                    cudaMemcpyHostToDevice));
                texCoordsIndex += numTexCoords;
            }
            else {
                devTexCoordsColl[i] = 0;
            }

            const size_t numTriangles = mesh.shapes[i].numTriangles;
            const size_t numTriIndices = mesh.shapes[i].numTriIndices;
            const size_t triIndicesSize = sizeof(int) * numTriIndices;
            // Log
            //for (size_t j = 0; j < mesh.shapes[i].numTriangles; ++j) {
            //    glm::ivec3 &tri = mesh.triIndices[triIndicesIndex + j];
            //    std::cout << tri.x << " " << tri.y << " " << tri.z << " ";
            //}
            //std::cout << std::endl;
            CUDA(cudaMalloc((void **)&devTriIndicesColl[i], triIndicesSize));
            CUDA(cudaMemcpy((void *)devTriIndicesColl[i], &mesh.triIndices[triIndicesIndex], triIndicesSize,
                cudaMemcpyHostToDevice));
            triIndicesIndex += numTriangles;
        }

        const size_t numMatIndices = mesh.numTriangles;
        const size_t matIndicesSize = sizeof(int) * numMatIndices;
        CUDA(cudaMalloc((void **)&devMatIndices, matIndicesSize));
        CUDA(cudaMemcpy((void *)devMatIndices, mesh.matIndices.data(), matIndicesSize, cudaMemcpyHostToDevice));

        std::vector<MaterialData> matData;
        matData.resize(mesh.numMaterials);
        int i = 0;
        for (auto &m : mesh.matParams) {
            matData[i].baseColour = m.baseColour;
            matData[i].alphaMode = m.alphaMode;
            matData[i].alphaCutoff = m.alphaCutoff;
            matData[i].emission = m.emission;
            matData[i].ior = m.ior;
            matData[i].roughness = m.roughness;
            matData[i].metallic = m.metallic;
            matData[i].specular = m.specular;
            matData[i].transmissionWeight = m.transmissionWeight;
            matData[i].subsurfaceWeight = m.subsurfaceWeight;
            matData[i].coatWeight = m.coatWeight;
            matData[i].coatRoughness = m.coatRoughness;
            matData[i].sheenColour = m.sheenData.colourFactor;
            matData[i].sheenRoughness = m.sheenData.roughnessFactor;

            if (!m.baseColourTexPresent) {
                matData[i].hasBaseColourTexture = false;
            }
            else {
                matData[i].hasBaseColourTexture = true;
                matData[i].baseColourTexture = baseColourTextureObjects[i];
                matData[i].baseColourTexTf= m.baseColourTexTf;
            }

            if (!m.normalTexPresent) {
                matData[i].hasNormalTexture = false;
            }
            else {
                matData[i].hasNormalTexture = true;
                matData[i].normalTexture = normalTextureObjects[i];
                matData[i].normalTexTf= m.normalTexTf;
            }

            if (!m.metallicRoughnessTexPresent) {
                matData[i].hasMetallicRoughnessTexture = false;
            }
            else {
                matData[i].hasMetallicRoughnessTexture = true;
                matData[i].metallicRoughnessTexture = metallicRoughnessTextureObjects[i];
                matData[i].metallicRoughnessTexTf= m.metallicRoughnessTexTf;
            }

            if (!m.emissiveTexPresent) {
                matData[i].hasEmissiveTexture = false;
            }
            else {
                matData[i].hasEmissiveTexture = true;
                matData[i].emissiveTexture = emissiveTextureObjects[i];
                matData[i].emissiveTexTf = m.emissiveTexTf;
            }

            if (!m.sheenColourTexPresent) {
                matData[i].hasSheenColourTexture = false;
            }
            else {
                matData[i].hasSheenColourTexture = true;
                matData[i].sheenColourTexture = sheenColourTextureObjects[i];
                matData[i].sheenColourTexTf = m.sheenData.colourTexTf;
            }

            if (!m.sheenRoughnessTexPresent) {
                matData[i].hasSheenRoughnessTexture = false;
            }
            else {
                matData[i].hasSheenRoughnessTexture = true;
                matData[i].sheenRoughnessTexture = sheenRoughnessTextureObjects[i];
                matData[i].sheenRoughnessTexTf = m.sheenData.roughnessTexTf;
            }

            matData[i].category = m.category;

            ++i;
        }

        const size_t numMaterials = mesh.numMaterials;
        const size_t matParamsSize = sizeof(MaterialData) * numMaterials;
        CUDA(cudaMalloc((void **)&devMatParams, matParamsSize));
        CUDA(cudaMemcpy((void *)devMatParams, matData.data(), matParamsSize, cudaMemcpyHostToDevice));

        const size_t emissiveTriVertsSize = sizeof(float3) * properties.emissiveVertices.size();
        CUDA(cudaMalloc((void **)&devEmissiveTriVerts, emissiveTriVertsSize));
        CUDA(cudaMemcpy((void *)devEmissiveTriVerts, properties.emissiveVertices.data(), emissiveTriVertsSize,
            cudaMemcpyHostToDevice));

        const size_t emissiveTriTexCoordsSize = sizeof(float2) * properties.emissiveTexCoords.size();
        CUDA(cudaMalloc((void **)&devEmissiveTriTexCoords, emissiveTriTexCoordsSize));
        CUDA(cudaMemcpy((void *)devEmissiveTriTexCoords, properties.emissiveTexCoords.data(), emissiveTriTexCoordsSize,
            cudaMemcpyHostToDevice));

        const size_t emissiveTriRadiancesSize = sizeof(float3) * properties.emissiveTriRadiances.size();
        CUDA(cudaMalloc((void **)&devEmissiveTriRadiances, emissiveTriRadiancesSize));
        CUDA(cudaMemcpy((void *)devEmissiveTriRadiances, properties.emissiveTriRadiances.data(),
            emissiveTriRadiancesSize, cudaMemcpyHostToDevice));

        const size_t emissiveTriMatIndicesSize = sizeof(int) * properties.emissiveTriMatIndices.size();
        CUDA(cudaMalloc((void **)&devEmissiveTriMatIndices, emissiveTriMatIndicesSize));
        CUDA(cudaMemcpy((void *)devEmissiveTriMatIndices, properties.emissiveTriMatIndices.data(),
            emissiveTriMatIndicesSize, cudaMemcpyHostToDevice));

        for (size_t i = 0; i < mesh.numShapes; ++i) {
            triangleInputColl[i] = {};
            triangleInputColl[i].type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
            triangleInputColl[i].triangleArray.vertexBuffers = &devVerticesColl[i];
            triangleInputColl[i].triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
            triangleInputColl[i].triangleArray.numVertices = static_cast<unsigned int>(mesh.shapes[i].numVertices);

            triangleInputColl[i].triangleArray.indexBuffer = devTriIndicesColl[i];
            triangleInputColl[i].triangleArray.numIndexTriplets = static_cast<unsigned int>(
                mesh.shapes[i].numTriangles);
            triangleInputColl[i].triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;

            triangleInputFlags[0] = OPTIX_GEOMETRY_FLAG_NONE;
            triangleInputColl[i].triangleArray.flags = triangleInputFlags;
            triangleInputColl[i].triangleArray.numSbtRecords = 1;

            // Only needed if numSbtRecords > 1.
            //triangleInputColl[i].triangleArray.sbtIndexOffsetBuffer = ...;
        }
    }

    OptixAccelBuildOptions accelOptions = {};
    if (allowCompaction)
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    if (allowUpdate)
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_UPDATE;
    if (build) {
        accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;
        // Use this flag to be able to retrieve arbitrary vertices from the
        // acceleration structure.
        accelOptions.buildFlags |= OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS;
    }
    else {
        accelOptions.operation = OPTIX_BUILD_OPERATION_UPDATE;
    }

    OptixAccelBufferSizes gasBufferSizes;
    OPTIX(optixAccelComputeMemoryUsage(optixContext, &accelOptions, triangleInputColl.data(),
        static_cast<int>(mesh.numShapes), &gasBufferSizes));
    //std::cout << (build ? "Build" : "Update") << std::endl;
    //std::cout << "output buffer size        = " << gasBufferSizes.outputSizeInBytes << std::endl;
    //std::cout << "temp buffer size          = " << gasBufferSizes.tempSizeInBytes << std::endl;
    //std::cout << "temp update buffer size   = " << gasBufferSizes.tempUpdateSizeInBytes << std::endl;

    //CUdeviceptr outputBuffer; // do not delete this buffer if compaction is not used!
    //CUDA(cudaFree((void *)outputBuffer));
    //CUDA(cudaMalloc((void **)&outputBuffer, gasBufferSizes.outputSizeInBytes));

    if (outputBufferSize < gasBufferSizes.outputSizeInBytes) {
        CUDA(cudaFree((void *)outputBuffer));
        CUDA(cudaMalloc((void **)&outputBuffer, gasBufferSizes.outputSizeInBytes));
        outputBufferSize = gasBufferSizes.outputSizeInBytes;
    }

    size_t requestedTempSizeInBytes = build ? gasBufferSizes.tempSizeInBytes : gasBufferSizes.tempUpdateSizeInBytes;
    if (tempBufferSize < requestedTempSizeInBytes) {
        CUDA(cudaFree((void *)tempBuffer));
        CUDA(cudaMalloc((void **)&tempBuffer, requestedTempSizeInBytes));
        tempBufferSize = requestedTempSizeInBytes;
    }

    if (allowCompaction && compactedSizeBuffer == 0)
        CUDA(cudaMalloc((void **)&compactedSizeBuffer, sizeof(uint64_t)));

    OptixAccelEmitDesc emitDesc;
    if (allowCompaction) {
        emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emitDesc.result = compactedSizeBuffer;
    }

    OPTIX(optixAccelBuild(optixContext, stream, &accelOptions, triangleInputColl.data(),
        static_cast<int>(mesh.numShapes), tempBuffer, requestedTempSizeInBytes, outputBuffer,
        gasBufferSizes.outputSizeInBytes, &gasHandle, allowCompaction ? &emitDesc : nullptr, allowCompaction ? 1 : 0));

    CUDA(cudaDeviceSynchronize());

    if (allowCompaction) {
        size_t newCompactedBufferSize;
        CUDA(cudaMemcpy(&newCompactedBufferSize, (void *)compactedSizeBuffer, sizeof(uint64_t),
            cudaMemcpyDeviceToHost));

        if (compactedBufferSize < newCompactedBufferSize) {
            CUDA(cudaFree((void *)compactedBuffer));
            CUDA(cudaMalloc((void **)&compactedBuffer, newCompactedBufferSize));
            compactedBufferSize = newCompactedBufferSize;
        }

        OPTIX(optixAccelCompact(optixContext, stream, gasHandle, compactedBuffer, newCompactedBufferSize, &gasHandle));
        CUDA(cudaDeviceSynchronize());
    }

    // Do not free the vertex and index buffers if the geometry may be updated!
}

void Renderer::setupCompilationOptions()
{
    moduleCompileOptions.maxRegisterCount = 0;

    //moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    //moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE;

    // For debugging, replace the above with these:
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;   // disable optimisation
#if (OPTIX_VERSION == 70100)
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO; // turn on line info
#else
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL; // turn on line info
#endif

    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

#if (OPTIX_VERSION == 70700)
    pipelineCompileOptions.allowOpacityMicromaps = 0;
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
#endif

    pipelineLinkOptions.maxTraceDepth = 2;
#if (OPTIX_VERSION == 70100)
    pipelineLinkOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_NONE; // not available in OptiX 7.7
#endif
    //pipelineLinkOptions.overrideUsesMotionBlur = false; // OptiX 7.0
}

void Renderer::createModules()
{
    // The modules vector *must* be populated in the order the modules appear in the Module enum.

    OptixModule module;
    std::string str;

    str = readFile("ptx\\device_programs.ptx");
    logSize = sizeof(log);
#if (OPTIX_VERSION >= 70700)
    OPTIX(optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#else
    OPTIX(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#endif
    modules.push_back(module);
    if (logSize > 1)
        std::cout << log << std::endl;

    str = readFile("ptx\\light_sampling.ptx");
    logSize = sizeof(log);
#if (OPTIX_VERSION >= 70700)
    OPTIX(optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#else
    OPTIX(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#endif
    modules.push_back(module);
    if (logSize > 1)
        std::cout << log << std::endl;

    str = readFile("ptx\\path_tracer.ptx");
    logSize = sizeof(log);
#if (OPTIX_VERSION >= 70700)
    OPTIX(optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#else
    OPTIX(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#endif
    modules.push_back(module);
    if (logSize > 1)
        std::cout << log << std::endl;

    str = readFile("ptx\\path_tracer_naive.ptx");
    logSize = sizeof(log);
#if (OPTIX_VERSION >= 70700)
    OPTIX(optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#else
    OPTIX(optixModuleCreateFromPTX(optixContext, &moduleCompileOptions, &pipelineCompileOptions, str.data(), str.size(),
        log, &logSize, &module));
#endif
    modules.push_back(module);
    if (logSize > 1)
        std::cout << log << std::endl;
}

void Renderer::createProgramGroupsPathTracer()
{
    OptixProgramGroupOptions progGroupOptions = {};

    // RayGen

    rayGenPG = nullptr;
    OptixProgramGroupDesc rayGenPGDesc = {};

    rayGenPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;

    rayGenPGDesc.raygen.module = modules[(int)Module::DevicePrograms];
    rayGenPGDesc.raygen.entryFunctionName = "__raygen__renderFrame";

    logSize = sizeof(log);
    OPTIX(optixProgramGroupCreate(optixContext, &rayGenPGDesc, 1, &progGroupOptions, log, &logSize, &rayGenPG));
    if (logSize > 1)
        std::cout << log << std::endl;

    // Miss

    missPGs.resize((int)RayType::Count);
    OptixProgramGroupDesc missPGDesc = {};

    missPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    missPGDesc.miss.module = modules[(int)Module::DevicePrograms];
    missPGDesc.miss.entryFunctionName = "__miss__radiance";

    logSize = sizeof(log);
    OPTIX(optixProgramGroupCreate(optixContext, &missPGDesc, 1, &progGroupOptions, log, &logSize,
        &missPGs[(int)RayType::Radiance]));
    if (logSize > 1)
        std::cout << log << std::endl;

    missPGDesc.miss.module = modules[(int)Module::DevicePrograms];
    missPGDesc.miss.entryFunctionName = "__miss__shadow";

    logSize = sizeof(log);
    OPTIX(optixProgramGroupCreate(optixContext, &missPGDesc, 1, &progGroupOptions, log, &logSize,
        &missPGs[(int)RayType::Shadow]));
    if (logSize > 1)
        std::cout << log << std::endl;

    // HitGroup

    hitGroupPGs.resize((int)RayType::Count);
    OptixProgramGroupDesc hitGroupPGDesc = {};

    hitGroupPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

    if (useNextEventEstimation) {
        hitGroupPGDesc.hitgroup.moduleCH = modules[(int)Module::PathTracer];
        hitGroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance";
    }
    else {
        hitGroupPGDesc.hitgroup.moduleCH = modules[(int)Module::PathTracerNaive];
        hitGroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__radiance_naive";
    }

    hitGroupPGDesc.hitgroup.moduleAH = modules[(int)Module::DevicePrograms];
    hitGroupPGDesc.hitgroup.entryFunctionNameAH = "__anyhit__radiance";

    logSize = sizeof(log);
    OPTIX(optixProgramGroupCreate(optixContext, &hitGroupPGDesc, 1, &progGroupOptions, log, &logSize,
        &hitGroupPGs[(int)RayType::Radiance]));
    if (logSize > 1)
        std::cout << log << std::endl;

    hitGroupPGDesc.hitgroup.moduleCH = modules[(int)Module::DevicePrograms];
    hitGroupPGDesc.hitgroup.entryFunctionNameCH = "__closesthit__shadow";
    hitGroupPGDesc.hitgroup.moduleAH = modules[(int)Module::DevicePrograms];
    hitGroupPGDesc.hitgroup.entryFunctionNameAH = "__anyhit__shadow";

    logSize = sizeof(log);
    OPTIX(optixProgramGroupCreate(optixContext, &hitGroupPGDesc, 1, &progGroupOptions, log, &logSize,
        &hitGroupPGs[(int)RayType::Shadow]));
    if (logSize > 1)
        std::cout << log << std::endl;

    // Callables

    callablesPGs.resize(1);
    OptixProgramGroupDesc callablesPGDesc = {};

    callablesPGDesc.kind = OPTIX_PROGRAM_GROUP_KIND_CALLABLES;
    callablesPGDesc.callables.moduleDC = modules[(int)Module::LightSampling];
    callablesPGDesc.callables.entryFunctionNameDC = "__direct_callable__sampleLight";

    logSize = sizeof(log);
    OPTIX(optixProgramGroupCreate(optixContext, &callablesPGDesc, 1, &progGroupOptions, log, &logSize,
        &callablesPGs[0]));
    if (logSize > 1)
        std::cout << log << std::endl;
}

void Renderer::createProgramGroups()
{
    createProgramGroupsPathTracer();
}

void Renderer::createPipeline()
{
    pipeline = nullptr;

    std::vector<OptixProgramGroup> progGroups;
    progGroups.push_back(rayGenPG);
    for (auto g : missPGs)
        progGroups.push_back(g);
    for (auto g : hitGroupPGs)
        progGroups.push_back(g);
    for (auto g : callablesPGs)
        progGroups.push_back(g);

    logSize = sizeof(log);
    OPTIX(optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions, progGroups.data(),
        static_cast<unsigned int>(progGroups.size()), log, &logSize, &pipeline));
    if (logSize > 1)
        std::cout << log << std::endl;

    OPTIX(optixPipelineSetStackSize(pipeline, 2048, 2048, 2048, 3));
}

void Renderer::setupSbt()
{
    // RayGen

    CUdeviceptr rayGenRecMem;
    const size_t rayGenRecSize = sizeof(RayGenSbtRecord);
    CUDA(cudaMalloc(reinterpret_cast<void **>(&rayGenRecMem), rayGenRecSize));

    RayGenSbtRecord rayGenRec;
    rayGenRec.data.emissiveTriVertices = (float3 *)devEmissiveTriVerts;
    rayGenRec.data.emissiveTriRadiances = (float3 *)devEmissiveTriRadiances;

    OPTIX(optixSbtRecordPackHeader(rayGenPG, &rayGenRec));
    CUDA(cudaMemcpy(reinterpret_cast<void *>(rayGenRecMem), &rayGenRec, rayGenRecSize, cudaMemcpyHostToDevice));

    // Miss

    std::vector<MissSbtRecord> missRecColl;
    missRecColl.resize(missPGs.size());
    for (size_t i = 0; i < missPGs.size(); ++i) {
        missRecColl[i].data.colour = sceneDesc.bgColour;
        OPTIX(optixSbtRecordPackHeader(missPGs[i], &missRecColl[i]));
    }

    CUdeviceptr missRecMem;
    const size_t missRecSize = sizeof(MissSbtRecord);
    CUDA(cudaMalloc(reinterpret_cast<void **>(&missRecMem), missRecColl.size() * missRecSize));
    CUDA(cudaMemcpy(reinterpret_cast<void *>(missRecMem), missRecColl.data(), missRecColl.size() * missRecSize,
        cudaMemcpyHostToDevice));

    // HitGroup

    std::vector<HitGroupSbtRecord> hitGroupRecColl;

    for (size_t j = 0; j < properties.mesh.numShapes; ++j) {
        for (size_t i = 0; i < hitGroupPGs.size(); ++i) {
            HitGroupSbtRecord rec;
            rec.data.normals = (float3 *)devNormalsColl[j];
            rec.data.texCoords = (float2 *)devTexCoordsColl[j];
            rec.data.triIndices = (int3 *)devTriIndicesColl[j];
            rec.data.matIndices = (int *)devMatIndices;
            rec.data.matData = (MaterialData *)devMatParams;
            rec.data.triOffset = static_cast<int>(properties.mesh.shapes[j].triangleOffset);

            rec.data.emissiveTriVertices = (float3 *)devEmissiveTriVerts;
            rec.data.emissiveTriTexCoords = (float2 *)devEmissiveTriTexCoords;
            rec.data.emissiveTriRadiances = (float3 *)devEmissiveTriRadiances;
            rec.data.emissiveTriMatIndices = (int *)devEmissiveTriMatIndices;

            OPTIX(optixSbtRecordPackHeader(hitGroupPGs[i], &rec));
            hitGroupRecColl.push_back(rec);
        }
    }

    CUdeviceptr hitGroupRecMem;
    const size_t hitGroupRecSize = sizeof(HitGroupSbtRecord);
    CUDA(cudaMalloc(reinterpret_cast<void **>(&hitGroupRecMem), hitGroupRecColl.size() * hitGroupRecSize));
    CUDA(cudaMemcpy(reinterpret_cast<void *>(hitGroupRecMem), hitGroupRecColl.data(), hitGroupRecColl.size() *
        hitGroupRecSize, cudaMemcpyHostToDevice));

    // Callables

    std::vector<CallablesSbtRecord> callablesRecColl;
    callablesRecColl.resize(callablesPGs.size());
    for (size_t i = 0; i < callablesPGs.size(); ++i)
        OPTIX(optixSbtRecordPackHeader(callablesPGs[i], &callablesRecColl[i]));

    CUdeviceptr callablesRecMem;
    const size_t callablesRecSize = sizeof(CallablesSbtRecord);
    CUDA(cudaMalloc(reinterpret_cast<void **>(&callablesRecMem), callablesRecColl.size() * callablesRecSize));
    CUDA(cudaMemcpy(reinterpret_cast<void *>(callablesRecMem), callablesRecColl.data(), callablesRecColl.size() *
        callablesRecSize, cudaMemcpyHostToDevice));

    // Sbt

    sbt = {};
    sbt.raygenRecord = rayGenRecMem;
    sbt.missRecordBase = missRecMem;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = static_cast<unsigned int>(missRecColl.size());
    sbt.hitgroupRecordBase = hitGroupRecMem;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = static_cast<unsigned int>(hitGroupRecColl.size());
    sbt.callablesRecordBase = callablesRecMem;
    sbt.callablesRecordStrideInBytes = sizeof(CallablesSbtRecord);
    sbt.callablesRecordCount = static_cast<unsigned int>(callablesRecColl.size());
}

void Renderer::setupOutputBuffer(uint32_t width, uint32_t height)
{
    if (width != fbWidth || height != fbHeight) {
        CUDA(cudaFree(fb));
        CUDA(cudaMalloc(&fb, (size_t)width * height * 4 * sizeof(float)));
        CUDA(cudaFree(accumFb));
        CUDA(cudaMalloc(&accumFb, (size_t)width * height * 4 * sizeof(float)));
        fbWidth = width;
        fbHeight = height;

        delete [] hostFb;
        delete [] hostAccumFb;
        hostFb = new float[(size_t)width * height * 4 * sizeof(float)];
        hostAccumFb = new float[(size_t)width * height * 4 * sizeof(float)];

        if (initDenoiser) {
            CUDA(cudaFree(denoisedFb));
            CUDA(cudaMalloc(&denoisedFb, (size_t)width * height * 4 * sizeof(float)));
        }
    }
}

void Renderer::setupCamera(uint32_t width, uint32_t height)
{
    if (sceneDesc.camAspectRatio > 0.0f)
        height = static_cast<uint32_t>(width / sceneDesc.camAspectRatio);

    /*
    float3 &eye = sceneDesc.camOrigin;
    float3 &tgt = sceneDesc.camTarget;
    std::cout << "Eye    : " << eye.x << ", " << eye.y << ", " << eye.z << std::endl;
    std::cout << "Target : " << tgt.x << ", " << tgt.y << ", " << tgt.z << std::endl;
    */

    cam.setParams(width, height,
        glm::vec3(sceneDesc.camOrigin.x, sceneDesc.camOrigin.y, sceneDesc.camOrigin.z),
        glm::vec3(sceneDesc.camTarget.x, sceneDesc.camTarget.y, sceneDesc.camTarget.z),
        glm::vec3(sceneDesc.camUp.x, sceneDesc.camUp.y, sceneDesc.camUp.z),
        sceneDesc.camFovYDeg);

    cam.setupUvwCoordinatSystem();
}

void Renderer::setupLaunchParams(SceneDesc &sceneDesc)
{
    launchParamsHost.frameBuffer = (float *)fb;
    launchParamsHost.accumFrameBuffer = (float *)accumFb;
    launchParamsHost.w = fbWidth;
    launchParamsHost.h = fbHeight;

    launchParamsHost.sample = sample;
    launchParamsHost.seed = seed;

    launchParamsHost.eye = make_float3(cam.eye.x, cam.eye.y, cam.eye.z);
    launchParamsHost.U = make_float3(cam.U.x, cam.U.y, cam.U.z);
    launchParamsHost.V = make_float3(cam.V.x, cam.V.y, cam.V.z);
    launchParamsHost.W = make_float3(cam.W.x, cam.W.y, cam.W.z);

    launchParamsHost.traversable = gasHandle;

    launchParamsHost.numEmissiveTriangles = static_cast<uint32_t>(properties.emissiveVertices.size() / 3);

    CUDA(cudaMalloc(&launchParamsHost.lights, sizeof(Light) * sceneDesc.lights.size()));
    CUDA(cudaMemcpy(launchParamsHost.lights, sceneDesc.lights.data(), sizeof(Light) * sceneDesc.lights.size(),
        cudaMemcpyHostToDevice));
    launchParamsHost.numLights = static_cast<uint32_t>(sceneDesc.lights.size());

    CUDA(cudaMalloc((void **)&launchParamsDevice, sizeof(LaunchParams)));
    CUDA(cudaMemcpy(launchParamsDevice, &launchParamsHost, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

void Renderer::updateLaunchParams()
{
    launchParamsHost.sample = sample;
    launchParamsHost.seed = seed;

    launchParamsHost.eye = make_float3(cam.eye.x, cam.eye.y, cam.eye.z);
    launchParamsHost.U = make_float3(cam.U.x, cam.U.y, cam.U.z);
    launchParamsHost.V = make_float3(cam.V.x, cam.V.y, cam.V.z);
    launchParamsHost.W = make_float3(cam.W.x, cam.W.y, cam.W.z);

    //std::printf("eye=(%.2f, %.2f, %.2f)\n", launchParamsHost.eye.x, launchParamsHost.eye.y, launchParamsHost.eye.z);
    //std::printf("U=(%.2f, %.2f, %.2f)\n", launchParamsHost.U.x, launchParamsHost.U.y, launchParamsHost.U.z);
    //std::printf("V=(%.2f, %.2f, %.2f)\n", launchParamsHost.V.x, launchParamsHost.V.y, launchParamsHost.V.z);
    //std::printf("W=(%.2f, %.2f, %.2f)\n", launchParamsHost.W.x, launchParamsHost.W.y, launchParamsHost.W.z);

    launchParamsHost.randState = randState;

    launchParamsHost.vis = vis;

    launchParamsHost.toneMapping = toneMapping;
    launchParamsHost.toneMappingImpl = toneMappingImpl;

    launchParamsHost.gammaCorrection = gammaCorrection;
    launchParamsHost.gammaTerm = gammaTerm;



    CUDA(cudaMemcpy(launchParamsDevice, &launchParamsHost, sizeof(LaunchParams), cudaMemcpyHostToDevice));
}

void Renderer::launch()
{
    // Note that the launch is async.
    OPTIX(optixLaunch(pipeline, stream, (CUdeviceptr)launchParamsDevice, sizeof(LaunchParams), &sbt, fbWidth, fbHeight,
        1));
    sample++;
    seed++;

    // Wait for completion.
    CUDA(cudaDeviceSynchronize());

    // Check for errors.
    //if (cudaGetLastError() != cudaSuccess) {
    //}

    // Copy framebuffer to host.
    if (!enableDenoiser) {
        CUDA(cudaMemcpy(hostFb, fb, fbWidth * fbHeight * 4 * sizeof(float), cudaMemcpyDeviceToHost));
    }
    else {
#if (OPTIX_VERSION == 70100)
        OptixImage2D inputImage;
        inputImage.data = (CUdeviceptr)fb;
        inputImage.width = sceneDesc.resX;
        inputImage.height = sceneDesc.resY;
        inputImage.rowStrideInBytes = sceneDesc.resX * 4 * sizeof(float);
        inputImage.pixelStrideInBytes = 4 * sizeof(float);
        inputImage.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixImage2D outputImage;
        outputImage.data = (CUdeviceptr)denoisedFb;
        outputImage.width = sceneDesc.resX;
        outputImage.height = sceneDesc.resY;
        outputImage.rowStrideInBytes = sceneDesc.resX * 4 * sizeof(float);
        outputImage.pixelStrideInBytes = 4 * sizeof(float);
        outputImage.format = OPTIX_PIXEL_FORMAT_FLOAT4;

        OptixDenoiserParams params = {};
#if (OPTIX_VERSION == 70100)
        params.denoiseAlpha = 0;
#else
        params.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
#endif
        params.hdrIntensity = (CUdeviceptr)0;
        params.blendFactor  = 0.0f;

        OPTIX(optixDenoiserInvoke(denoiser, stream, &params, denoiserState, denoiserStateSizeInBytes, &inputImage, 1, 0,
            0, &outputImage, denoiserScratch, denoiserScratchSizeInBytes));

        // Note that the denoised image may contain values that slightly exceed
        // 1.0f. Clamp these values if dumping the output to file as LDR.

        CUDA(cudaDeviceSynchronize());

        CUDA(cudaMemcpy(hostFb, (void *)outputImage.data, fbWidth * fbHeight * 4 * sizeof(float),
            cudaMemcpyDeviceToHost));
#else
        CUDA(cudaMemcpy(hostFb, fb, fbWidth * fbHeight * 4 * sizeof(float), cudaMemcpyDeviceToHost));
#endif
    }

    if (dumpFb) {
        if (enableDenoiser)
            dumpFbToPng((float *)denoisedFb, fbWidth, fbHeight);
        else
            dumpFbToPng((float *)fb, fbWidth, fbHeight);
        dumpFb = false;
    }

    if (dumpAccumFbRgbe) {
        CUDA(cudaMemcpy(hostAccumFb, accumFb, fbWidth * fbHeight * 4 * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "Dumping accum framebuffer (RGBE) ...\n";
        if (stbi_write_hdr("output.hdr", fbWidth, fbHeight, 4, hostAccumFb) == 0)
            std::cerr << "stbi_write_hdr() failed!" << std::endl;

        dumpAccumFbRgbe = false;
    }

    if (dumpAccumFbExr) {
        CUDA(cudaMemcpy(hostAccumFb, accumFb, fbWidth * fbHeight * 4 * sizeof(float), cudaMemcpyDeviceToHost));

        std::cout << "Dumping accum framebuffer (EXR) ...\n";
        dumpAccumFbToOpenExr((float *)hostAccumFb, fbWidth, fbHeight);

        dumpAccumFbExr = false;
    }
}

void Renderer::updateWorldMatrix(size_t nodeIndex, glm::mat4 &globalTf)
{
    SceneNode &node = properties.nodes[nodeIndex];

    if (!node.hasMatrix) {
        glm::mat4 translate = glm::translate(glm::mat4(1.0f), node.translation);
        glm::mat4 rotate = glm::toMat4(node.rotation);
        glm::mat4 scale = glm::scale(glm::mat4(1.0f), node.scale);
        node.localTf = translate * rotate * scale;
    }

    /*
    std::cout << "Node " << nodeId << " (gltf " << node.id << ")" << std::endl;
    std::cout << "  T : " << std::fixed << node.translation.x << " " << node.translation.y << " " <<
        node.translation.z << std::endl;
    std::cout << "  R : " << std::fixed << node.rotation.x << " " << node.rotation.y << " " << node.rotation.z << " " <<
        node.rotation.w << " [xyzw]" << std::endl;
    std::cout << "  S : " << std::fixed << node.scale.x << " " << node.scale.y << " " << node.scale.z <<
        std::endl;
    */

    node.globalTf = globalTf * node.localTf;

    globalTf = node.globalTf;

    for (auto childIndex : node.children) {
        glm::mat4 tempGlobalTf = globalTf;
        updateWorldMatrix(childIndex, tempGlobalTf);
    }
}

void Renderer::updateScene()
{
    static bool firstCall = true;

    if (properties.animShapes.empty() && !firstCall)
        return;
    firstCall = false;

    // 1. Update node transforms from animations.
    // 2. Traverse the scene graph computing the world matrix of every node.
    // 3. Render the scene graph.
    //    - For unskinned meshes a single world matrix is needed.
    //    - For skinned meshes, the mesh's world matrix is needed plus all the skin joint matrices.

    static float absoluteT = 0.0f;
    absoluteT += 0.02f;

    // 1.

    for (auto &a : properties.animShapes) {
        int numValues = static_cast<int>(a.data.size());

        float t = absoluteT;
        // This causes the animation to stop when the maximum time is reached:
        //if (t > a.data[numValues - 1].t)
        //    t = 0.0f;
        // This causes the animation to start over when the maximum time is reached:
        if (t > a.data[numValues - 1].t) {
            float integerPart;
            t = modf(t / a.data[numValues - 1].t, &integerPart) * a.data[numValues - 1].t;
        }

        int prevIndex = 0;
        int nextIndex = 0;
        for (int i = 0; i < numValues; ++i) {
            if (a.data[i].t > t) {
                prevIndex = std::max(0, i - 1);
                nextIndex = i;
                break;
            }
        }

        //std::cout << std::fixed << "t = " << t << " prev = " << prevIndex << " next = " << nextIndex << std::endl;

        if (a.type == 0) { // translation
            glm::vec3 translation = glm::vec3(0.0f, 0.0f, 0.0f);

            if (a.interpolation == 0) { // step
                translation = a.data[prevIndex].v3;
            }
            else { // linear (using this until cubic interpolation is implemented)
                float interpolationValue = (t - a.data[prevIndex].t) / (a.data[nextIndex].t - a.data[prevIndex].t);
                translation = glm::mix(a.data[prevIndex].v3, a.data[nextIndex].v3, interpolationValue);
                //std::cout << "Translation : " << std::fixed << translation.x << " " << translation.y << " " <<
                //    translation.z << std::endl;
                //translation = glm::vec3(0.0f);
            }

            properties.nodes[a.nodeId].translation = translation;
        }
        else if (a.type == 1) { // rotation
            glm::quat q = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);

            if (a.interpolation == 0) { // step
                q = a.data[prevIndex].q;
            }
            else { // linear (using this until cubic interpolation is implemented)
                float interpolationValue = (t - a.data[prevIndex].t) / (a.data[nextIndex].t - a.data[prevIndex].t);
                q = glm::slerp(a.data[prevIndex].q, a.data[nextIndex].q, interpolationValue);
                //q = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
            }

            properties.nodes[a.nodeId].rotation = q;
        }
        else if (a.type == 2) { // scaling
            glm::vec3 scale = glm::vec3(1.0f, 1.0f, 1.0f);

            if (a.interpolation == 0) { // step
                scale = a.data[prevIndex].v3;
            }
            else { // linear (using this until cubic interpolation is implemented)
                float interpolationValue = (t - a.data[prevIndex].t) / (a.data[nextIndex].t - a.data[prevIndex].t);
                scale = glm::mix(a.data[prevIndex].v3, a.data[nextIndex].v3, interpolationValue);
                //std::cout << "Scale : " << std::fixed << scale.x << " " << scale.y << " " << scale.z << std::endl;
                //scale = glm::vec3(1.0f, 1.0f, 1.0f);
            }

            properties.nodes[a.nodeId].scale = scale;
        }
    }

    // 2.

    size_t numRootNodes = properties.rootNodeIndices.size();
    for (size_t i = 0; i < numRootNodes; ++i) {
        glm::mat4 globalTf(1.0f);
        updateWorldMatrix(properties.rootNodeIndices[i], globalTf);
    }

    // 3.

    for (size_t shapeCounter = 0; shapeCounter < properties.mesh.numShapes; ++shapeCounter) {
        Shape &shape = properties.mesh.shapes[shapeCounter];
        SceneNode &node = properties.nodes[properties.gltfNodeIdToNodeIndexMap.find(shape.nodeId)->second];

        if (node.hasSkin) {
            Skin &skin = properties.skins[shape.skinIndex];
            size_t numJoints = skin.joints.size();

            // Calculate joint matrices.

            for (size_t i = 0; i < numJoints; ++i) {
                size_t jointNodeIndex = properties.gltfNodeIdToNodeIndexMap.find(skin.joints[i])->second;
                SceneNode &jointNode = properties.nodes[jointNodeIndex];
                glm::mat4 globalJointNodeTransform = jointNode.globalTf;

                skin.jointMatrices[i] =
                    glm::inverse(node.globalTf) *
                    globalJointNodeTransform *
                    skin.inverseBindMatrices[i];

                // The bind pose.
                // This produces the same result as if the mesh was rendered
                // with hasSkin == false.
                //skin.jointMatrices[i] = glm::mat4(1.0f);

                // The mesh is transformed to joint space. A jumble of parts
                // should appear all on top of each other.
                //skin.jointMatrices[i] = skin.inverseBindMatrices[i];
            }

            const size_t numVertices = shape.numVertices;
            const size_t verticesSize = sizeof(float3) * numVertices;
            const size_t verticesIndex = shape.verticesIndex;

            for (size_t i = 0; i < numVertices; ++i) {
                //std::cout << "num joint matrices = " << skin.jointMatrices.size() << std::endl;
                //std::cout << i << ": " << shape.joints[i].x << " " << shape.joints[i].y << " " << shape.joints[i].z <<
                //    " " << shape.joints[i].w << std::endl;

                glm::mat4 skinMat =
                    shape.weights[i].x * skin.jointMatrices[shape.joints[i].x] +
                    shape.weights[i].y * skin.jointMatrices[shape.joints[i].y] +
                    shape.weights[i].z * skin.jointMatrices[shape.joints[i].z] +
                    shape.weights[i].w * skin.jointMatrices[shape.joints[i].w];

                //std::cout << "Shape " << shapeCounter << " : weight " << i << " = " << std::fixed <<
                //    shape.weights[i].x << ", " <<
                //    shape.weights[i].y << ", " <<
                //    shape.weights[i].z << ", " <<
                //    shape.weights[i].w << std::endl;

                //std::cout << "Shape " << shapeCounter << " : joints " << i << " = " <<
                //    shape.joints[i].x << ", " <<
                //    shape.joints[i].y << ", " <<
                //    shape.joints[i].z << ", " <<
                //    shape.joints[i].w << std::endl;

                properties.mesh.positionsTemp[verticesIndex + i] =
                    node.globalTf * skinMat * glm::vec4(properties.mesh.positions[verticesIndex + i], 1.0f);

                //std::cout << "Shape " << shapeCounter << " : pos " << i << " = " << std::fixed <<
                //    properties.mesh.positionsTemp[shape.verticesIndex + i].x << ", " <<
                //    properties.mesh.positionsTemp[shape.verticesIndex + i].y << ", " <<
                //    properties.mesh.positionsTemp[shape.verticesIndex + i].z << std::endl;

                if (shape.numNormals != 0) {
                    assert(shape.numNormals == numVertices);
                    glm::mat4 modifiedGlobalTf = glm::transpose(glm::inverse(node.globalTf));
                    properties.mesh.normalsTemp[shape.normalsIndex + i] =
                        glm::vec3(modifiedGlobalTf * glm::vec4(properties.mesh.normals[shape.normalsIndex + i], 1.0f));
                }
            }
        }
        else {
            for (size_t i = 0; i < shape.numVertices; ++i) {
                properties.mesh.positionsTemp[shape.verticesIndex + i] =
                    node.globalTf * glm::vec4(properties.mesh.positions[shape.verticesIndex + i], 1.0f);

                //std::cout << "Shape " << shapeCounter << " : pos " << i << " = " << std::fixed <<
                //    properties.mesh.positionsTemp[shape.verticesIndex + i].x << ", " <<
                //    properties.mesh.positionsTemp[shape.verticesIndex + i].y << ", " <<
                //    properties.mesh.positionsTemp[shape.verticesIndex + i].z << std::endl;
            }

            for (size_t i = 0; i < shape.numNormals; ++i) {
                glm::mat4 modifiedGlobalTf = glm::transpose(glm::inverse(node.globalTf));
                properties.mesh.normalsTemp[shape.normalsIndex + i] =
                    glm::vec3(modifiedGlobalTf * glm::vec4(properties.mesh.normals[shape.normalsIndex + i], 1.0f));
            }
        }

        /*
        std::printf("numTriangles   = %zd\n", shape.numTriangles);
        std::printf("numTriIndices  = %zd\n", shape.numTriIndices);
        std::printf("triangleOffset = %zd\n", shape.triangleOffset);

        // Print the global triangle indices.
        for (int i = 0; i < shape.numTriangles; ++i)
            std::printf("%d\n", (int)shape.triangleOffset + i);
        */

        /*
        // Print the global vertex indices of each triangle.
        for (int i = 0; i < shape.numTriangles; ++i) {
            glm::ivec3 &indices = properties.mesh.triIndices[shape.triangleOffset + i];
            int offset = (int)shape.verticesIndex;
            std::printf("%d, %d, %d\n", offset + indices.x, offset + indices.y, offset + indices.z);
        }
        */

        const size_t verticesSize = sizeof(float3) * shape.numVertices;
        CUDA(cudaMemcpy((void *)devVerticesColl[shapeCounter], &properties.mesh.positionsTemp[shape.verticesIndex],
            verticesSize, cudaMemcpyHostToDevice));

        if (shape.numNormals != 0) {
            const size_t normalsSize = sizeof(float3) * shape.numNormals;
            CUDA(cudaMemcpy((void *)devNormalsColl[shapeCounter], &properties.mesh.normalsTemp[shape.normalsIndex],
                normalsSize, cudaMemcpyHostToDevice));
        }
    }

    // Update emissive triangle vertices, texture coordinates and mat indices.

    size_t numEmissiveVertices = properties.emissiveVertices.size();
    for (size_t i = 0; i < numEmissiveVertices; ++i) {
        properties.emissiveVertices[i] = properties.mesh.positionsTemp[properties.emissiveVertexIndices[i]];
        if (properties.mesh.hasTexCoords)
            properties.emissiveTexCoords[i] = properties.mesh.texCoords[properties.emissiveVertexIndices[i]];
    }

    CUDA(cudaMemcpy((void *)devEmissiveTriVerts, properties.emissiveVertices.data(),
        sizeof(float3) * numEmissiveVertices, cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy((void *)devEmissiveTriTexCoords, properties.emissiveTexCoords.data(),
        sizeof(float2) * numEmissiveVertices, cudaMemcpyHostToDevice));

    updateAccel();
}

void Renderer::updateAccel()
{
    buildAccel(false);
    resetSampleCount();
}

void Renderer::createDenoiser()
{
    if (!initDenoiser)
        return;

#if (OPTIX_VERSION == 70100)

    OptixDenoiserOptions options = {};
    options.inputKind = OPTIX_DENOISER_INPUT_RGB;
#if (OPTIX_VERSION == 70000)
    options.pixelFormat = OPTIX_PIXEL_FORMAT_FLOAT4;
#endif

    OPTIX(optixDenoiserCreate(optixContext, &options, &denoiser));
    OPTIX(optixDenoiserSetModel(denoiser, OPTIX_DENOISER_MODEL_KIND_LDR, nullptr, 0));

    OptixDenoiserSizes sizes;
    OPTIX(optixDenoiserComputeMemoryResources(denoiser, sceneDesc.resX, sceneDesc.resY, &sizes));

    denoiserStateSizeInBytes = sizes.stateSizeInBytes;
#if (OPTIX_VERSION == 70000)
    denoiserScratchSizeInBytes = sizes.recommendedScratchSizeInBytes;
#else
    denoiserScratchSizeInBytes = sizes.withoutOverlapScratchSizeInBytes;
#endif
    CUDA(cudaMalloc((void **)&denoiserState, denoiserStateSizeInBytes));
    CUDA(cudaMalloc((void **)&denoiserScratch, denoiserScratchSizeInBytes));

    OPTIX(optixDenoiserSetup(denoiser, stream, sceneDesc.resX, sceneDesc.resY, denoiserState, denoiserStateSizeInBytes,
        denoiserScratch, denoiserScratchSizeInBytes));

#endif
}

void Renderer::createRandomState()
{
    size_t numPixels = fbWidth * fbHeight;
    CUDA(cudaMalloc(&randState, numPixels * sizeof(curandState)));
    initRandState(fbWidth, fbHeight, randState);
}

void *Renderer::getFrameBuffer()
{
    return hostFb;
}

uint32_t Renderer::getSampleCount()
{
    return sample;
}

void Renderer::resetSampleCount()
{
    sample = 0;
}

void Renderer::dumpFramebuffer()
{
    dumpFb = true;
}

void Renderer::dumpAccumFramebufferRgbe()
{
    dumpAccumFbRgbe = true;
}

void Renderer::dumpAccumFramebufferExr()
{
    dumpAccumFbExr = true;
}

bool Renderer::getDenoiserState()
{
    return enableDenoiser;
}

void Renderer::setDenoiserState(bool on)
{
    enableDenoiser = on;
}

void Renderer::setVis(int vis)
{
    if (vis != this->vis)
        resetSampleCount();

    this->vis = vis;
}

int Renderer::getVis()
{
    return vis;
}

void Renderer::setToneMappingState(bool enable)
{
    toneMapping = enable;
}

bool Renderer::getToneMappingState()
{
    return toneMapping;
}

void Renderer::setToneMappingImpl(int toneMappingImpl)
{
    this->toneMappingImpl = toneMappingImpl;
}

int Renderer::getToneMappingImpl()
{
    return toneMappingImpl;
}

void Renderer::setGammaCorrectionState(bool enable)
{
    gammaCorrection = enable;
}

int Renderer::getGammaCorrectionState()
{
    return gammaCorrection;
}

void Renderer::setGammaCorrectionTerm(float value)
{
    gammaTerm = value;
}

float Renderer::getGammaCorrectionTerm()
{
    return gammaTerm;
}

void Renderer::loadTextures()
{
    Mesh &mesh = properties.mesh;

    //
    // Base colour
    //

    baseColourTextureArrays.resize(mesh.numMaterials);
    baseColourTextureObjects.resize(mesh.numMaterials);

    for (size_t i = 0; i < mesh.numMaterials; ++i) {
        MaterialParams &matParams = mesh.matParams[i];

        if (matParams.baseColourTexPresent) {
            cudaChannelFormatDesc channelDesc;
            int pitch = matParams.baseColourTexWidth * matParams.baseColourTexNumComponents * sizeof(uint8_t);
            channelDesc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = baseColourTextureArrays[i];
            CUDA(cudaMallocArray(&pixelArray, &channelDesc, matParams.baseColourTexWidth,
                matParams.baseColourTexHeight));
            CUDA(cudaMemcpy2DToArray(pixelArray, 0, 0, matParams.baseColourTexRgba, pitch, pitch,
                matParams.baseColourTexHeight, cudaMemcpyHostToDevice));

            delete [] matParams.baseColourTexRgba;
            matParams.baseColourTexRgba = nullptr;

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = pixelArray;

            cudaTextureDesc texDesc     = {};
            texDesc.addressMode[0]      = cudaAddressModeWrap;
            texDesc.addressMode[1]      = cudaAddressModeWrap;
            texDesc.filterMode          = cudaFilterModeLinear;
            texDesc.readMode            = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords    = 1;
            texDesc.maxAnisotropy       = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode    = cudaFilterModePoint;
            texDesc.borderColor[0]      = 1.0f;
            texDesc.sRGB                = 1; // needs conversion

            if (matParams.baseColourTexSampler) {
                switch (matParams.baseColourTexSampler->wrapS) {
                case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
                    texDesc.addressMode[0] = cudaAddressModeClamp;
                    break;

                case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
                    texDesc.addressMode[0] = cudaAddressModeMirror;
                    break;

                case TINYGLTF_TEXTURE_WRAP_REPEAT:
                default:
                    texDesc.addressMode[0] = cudaAddressModeWrap;
                    break;
                }

                switch (matParams.baseColourTexSampler->wrapT) {
                case TINYGLTF_TEXTURE_WRAP_CLAMP_TO_EDGE:
                    texDesc.addressMode[1] = cudaAddressModeClamp;
                    break;

                case TINYGLTF_TEXTURE_WRAP_MIRRORED_REPEAT:
                    texDesc.addressMode[1] = cudaAddressModeMirror;
                    break;

                case TINYGLTF_TEXTURE_WRAP_REPEAT:
                default:
                    texDesc.addressMode[1] = cudaAddressModeWrap;
                    break;
                }

                switch (matParams.baseColourTexSampler->magFilter) {
                case TINYGLTF_TEXTURE_FILTER_NEAREST:
                    texDesc.filterMode = cudaFilterModePoint;
                    break;

                case TINYGLTF_TEXTURE_FILTER_LINEAR:
                default:
                    texDesc.filterMode = cudaFilterModeLinear;
                    break;
                }
            }

            cudaTextureObject_t cudaTex = 0;
            CUDA(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
            baseColourTextureObjects[i] = cudaTex;
        }
    }

    //
    // Normal
    //

    normalTextureArrays.resize(mesh.numMaterials);
    normalTextureObjects.resize(mesh.numMaterials);

    for (size_t i = 0; i < mesh.numMaterials; ++i) {
        MaterialParams &matParams = mesh.matParams[i];

        if (matParams.normalTexPresent) {
            cudaChannelFormatDesc channelDesc;
            int pitch = matParams.normalTexWidth * matParams.normalTexNumComponents * sizeof(uint8_t);
            channelDesc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = normalTextureArrays[i];
            CUDA(cudaMallocArray(&pixelArray, &channelDesc, matParams.normalTexWidth, matParams.normalTexHeight));
            CUDA(cudaMemcpy2DToArray(pixelArray, 0, 0, matParams.normalTexRgba, pitch, pitch, matParams.normalTexHeight,
                cudaMemcpyHostToDevice));

            delete [] matParams.normalTexRgba;
            matParams.normalTexRgba = nullptr;

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = pixelArray;

            cudaTextureDesc texDesc     = {};
            texDesc.addressMode[0]      = cudaAddressModeWrap;
            texDesc.addressMode[1]      = cudaAddressModeWrap;
            texDesc.filterMode          = cudaFilterModeLinear;
            texDesc.readMode            = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords    = 1;
            texDesc.maxAnisotropy       = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode    = cudaFilterModePoint;
            texDesc.borderColor[0]      = 1.0f;
            texDesc.sRGB                = 0; // already linear

            cudaTextureObject_t cudaTex = 0;
            CUDA(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
            normalTextureObjects[i] = cudaTex;
        }
    }

    //
    // Metallic roughness
    //

    metallicRoughnessTextureArrays.resize(mesh.numMaterials);
    metallicRoughnessTextureObjects.resize(mesh.numMaterials);

    for (size_t i = 0; i < mesh.numMaterials; ++i) {
        MaterialParams &matParams = mesh.matParams[i];

        if (matParams.metallicRoughnessTexPresent) {
            cudaChannelFormatDesc channelDesc;
            int pitch = matParams.metallicRoughnessTexWidth * matParams.metallicRoughnessTexNumComponents *
                sizeof(uint8_t);
            channelDesc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = metallicRoughnessTextureArrays[i];
            CUDA(cudaMallocArray(&pixelArray, &channelDesc, matParams.metallicRoughnessTexWidth,
                matParams.metallicRoughnessTexHeight));
            CUDA(cudaMemcpy2DToArray(pixelArray, 0, 0, matParams.metallicRoughnessTexRgba, pitch, pitch,
                matParams.metallicRoughnessTexHeight, cudaMemcpyHostToDevice));

            delete [] matParams.metallicRoughnessTexRgba;
            matParams.metallicRoughnessTexRgba = nullptr;

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = pixelArray;

            cudaTextureDesc texDesc     = {};
            texDesc.addressMode[0]      = cudaAddressModeWrap;
            texDesc.addressMode[1]      = cudaAddressModeWrap;
            texDesc.filterMode          = cudaFilterModeLinear;
            texDesc.readMode            = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords    = 1;
            texDesc.maxAnisotropy       = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode    = cudaFilterModePoint;
            texDesc.borderColor[0]      = 1.0f;
            texDesc.sRGB                = 0; // already linear

            cudaTextureObject_t cudaTex = 0;
            CUDA(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
            metallicRoughnessTextureObjects[i] = cudaTex;
        }
    }

    //
    // Emissive
    //

    emissiveTextureArrays.resize(mesh.numMaterials);
    emissiveTextureObjects.resize(mesh.numMaterials);

    for (size_t i = 0; i < mesh.numMaterials; ++i) {
        MaterialParams &matParams = mesh.matParams[i];

        if (matParams.emissiveTexPresent) {
            cudaChannelFormatDesc channelDesc;
            int pitch = matParams.emissiveTexWidth * matParams.emissiveTexNumComponents * sizeof(uint8_t);
            channelDesc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = emissiveTextureArrays[i];
            CUDA(cudaMallocArray(&pixelArray, &channelDesc, matParams.emissiveTexWidth, matParams.emissiveTexHeight));
            CUDA(cudaMemcpy2DToArray(pixelArray, 0, 0, matParams.emissiveTexRgba, pitch, pitch,
                matParams.emissiveTexHeight, cudaMemcpyHostToDevice));

            delete [] matParams.emissiveTexRgba;
            matParams.emissiveTexRgba = nullptr;

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = pixelArray;

            cudaTextureDesc texDesc     = {};
            texDesc.addressMode[0]      = cudaAddressModeWrap;
            texDesc.addressMode[1]      = cudaAddressModeWrap;
            texDesc.filterMode          = cudaFilterModeLinear;
            texDesc.readMode            = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords    = 1;
            texDesc.maxAnisotropy       = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode    = cudaFilterModePoint;
            texDesc.borderColor[0]      = 1.0f;
            texDesc.sRGB                = 1; // needs conversion

            cudaTextureObject_t cudaTex = 0;
            CUDA(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
            emissiveTextureObjects[i] = cudaTex;
        }
    }

    //
    // Sheen colour and roughness
    //

    sheenColourTextureArrays.resize(mesh.numMaterials);
    sheenColourTextureObjects.resize(mesh.numMaterials);
    sheenRoughnessTextureArrays.resize(mesh.numMaterials);
    sheenRoughnessTextureObjects.resize(mesh.numMaterials);

    for (size_t i = 0; i < mesh.numMaterials; ++i) {
        MaterialParams &matParams = mesh.matParams[i];

        if (matParams.sheenColourTexPresent) {
            cudaChannelFormatDesc channelDesc;
            int pitch = matParams.sheenColourTexWidth * matParams.sheenColourTexNumComponents * sizeof(uint8_t);
            channelDesc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = sheenColourTextureArrays[i];
            CUDA(cudaMallocArray(&pixelArray, &channelDesc, matParams.sheenColourTexWidth,
                matParams.sheenColourTexHeight));
            CUDA(cudaMemcpy2DToArray(pixelArray, 0, 0, matParams.sheenColourTexRgba, pitch, pitch,
                matParams.sheenColourTexHeight, cudaMemcpyHostToDevice));

            delete [] matParams.sheenColourTexRgba;
            matParams.sheenColourTexRgba = nullptr;

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = pixelArray;

            cudaTextureDesc texDesc     = {};
            texDesc.addressMode[0]      = cudaAddressModeWrap;
            texDesc.addressMode[1]      = cudaAddressModeWrap;
            texDesc.filterMode          = cudaFilterModeLinear;
            texDesc.readMode            = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords    = 1;
            texDesc.maxAnisotropy       = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode    = cudaFilterModePoint;
            texDesc.borderColor[0]      = 1.0f;
            texDesc.sRGB                = 1; // needs conversion

            cudaTextureObject_t cudaTex = 0;
            CUDA(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
            sheenColourTextureObjects[i] = cudaTex;
        }

        if (matParams.sheenRoughnessTexPresent) {
            cudaChannelFormatDesc channelDesc;
            int pitch = matParams.sheenRoughnessTexWidth * matParams.sheenRoughnessTexNumComponents * sizeof(uint8_t);
            channelDesc = cudaCreateChannelDesc<uchar4>();

            cudaArray_t &pixelArray = sheenRoughnessTextureArrays[i];
            CUDA(cudaMallocArray(&pixelArray, &channelDesc, matParams.sheenRoughnessTexWidth,
                matParams.sheenRoughnessTexHeight));
            CUDA(cudaMemcpy2DToArray(pixelArray, 0, 0, matParams.sheenRoughnessTexRgba, pitch, pitch,
                matParams.sheenRoughnessTexHeight, cudaMemcpyHostToDevice));

            delete [] matParams.sheenRoughnessTexRgba;
            matParams.sheenRoughnessTexRgba = nullptr;

            cudaResourceDesc resDesc = {};
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = pixelArray;

            cudaTextureDesc texDesc     = {};
            texDesc.addressMode[0]      = cudaAddressModeWrap;
            texDesc.addressMode[1]      = cudaAddressModeWrap;
            texDesc.filterMode          = cudaFilterModeLinear;
            texDesc.readMode            = cudaReadModeNormalizedFloat;
            texDesc.normalizedCoords    = 1;
            texDesc.maxAnisotropy       = 1;
            texDesc.maxMipmapLevelClamp = 99;
            texDesc.minMipmapLevelClamp = 0;
            texDesc.mipmapFilterMode    = cudaFilterModePoint;
            texDesc.borderColor[0]      = 1.0f;
            texDesc.sRGB                = 1; // needs conversion (usually the alpha channel in the sheen colour texture)

            cudaTextureObject_t cudaTex = 0;
            CUDA(cudaCreateTextureObject(&cudaTex, &resDesc, &texDesc, nullptr));
            sheenRoughnessTextureObjects[i] = cudaTex;
        }
    }
}

void optixLogCallback(unsigned int level, const char *tag, const char *message, void *data)
{
    std::cout << "Level: " << level << " Tag: " << tag << " Message: " << message << std::endl;
}
