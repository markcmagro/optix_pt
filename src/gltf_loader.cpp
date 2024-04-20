#include <iostream>
#include "gltf_loader.h"
#include "utils.h"

#include <glm/gtx/quaternion.hpp> // toMat4()
#include <glm/gtc/type_ptr.hpp>

#include "stb/stb_image.h"

GltfLoader::GltfLoader(const char *sceneFile, SceneProperties &properties) :
    sceneFile(sceneFile), properties(properties)
{
    scene = nullptr;
    cameraPresent = false;
    camOrigin = glm::vec3(0.0f);
    camTarget = glm::vec3(0.0f, 0.0f, -1.0f);
}

bool GltfLoader::parse()
{
    std::string err;
    std::string warn;

    bool ret = loader.LoadASCIIFromFile(&model, &err, &warn, sceneFile);

    if (!warn.empty())
        std::cout << "Warning! " << warn;

    if (!err.empty())
        std::cerr << "Error! " << err;

    if (!ret) {
        std::cerr << "Failed to load " << sceneFile << std::endl;
        return false;
    }

    // Check glTF version.
    if (model.asset.version != "2.0") {
        std::cerr << "Unsupported glTF version!" << std::endl;
        return false;
    }

    // Info.
    std::printf("Generator: %s\n", model.asset.generator.c_str());
    std::printf("Extensions used:\n");
    for (auto &s : model.extensionsUsed)
        std::printf("  %s\n", s.c_str());
    std::printf("Extensions required:\n");
    for (auto &s : model.extensionsRequired)
        std::printf("  %s\n", s.c_str());

    // Get scene ID.
    int sceneId = model.defaultScene;
    if (sceneId == -1) {
        std::cerr << "No scene specified." << std::endl;
        return false;
    }
    if (model.scenes.size() <= sceneId) {
        std::cerr << "Scene " << sceneId << " not found!" << std::endl;
        return false;
    }

    scene = &model.scenes[sceneId];
    std::cout << "Scene " << sceneId;
    if (!scene->name.empty())
        std::cout << " (" << scene->name << ")";
    std::cout << std::endl;

    return true;
}

void GltfLoader::loadScene()
{
    // Initialise mesh counts and buffers.

    scanScene();

    Mesh &mesh = properties.mesh;

    mesh.numShapes = numMeshes;
    mesh.shapes.resize(numMeshes);

    mesh.numVertices = numVertices;
    mesh.positions.resize(mesh.numVertices);
    mesh.positionsTemp.resize(mesh.numVertices);

    mesh.hasNormals = (numNormals != 0);
    if (mesh.hasNormals) {
        mesh.normals.resize(mesh.numVertices);
        mesh.normalsTemp.resize(mesh.numVertices);
    }

    mesh.hasTexCoords = (numTexCoords != 0);
    if (mesh.hasTexCoords)
        mesh.texCoords.resize(mesh.numVertices);

    mesh.numTriangles = numTriangles;
    mesh.triIndices.resize(numTriangles);
    mesh.matIndices.resize(numTriangles);

    mesh.numMaterials = model.materials.size();
    mesh.matParams.resize(model.materials.size());

    // Materials.
    initMaterialParams(mesh.matParams);

    // Nodes.
    size_t numRootNodes = scene->nodes.size();
    for (size_t i = 0; i < numRootNodes; ++i) {
        glm::mat4 globalTf(1.0f);
        loadNode(scene->nodes[i], properties.lights, globalTf);
    }

    /*
    // Log shapes.
    std::cout << "Shapes" << std::endl;
    for (auto &s : mesh.shapes)
        std::cout << "  triangleOffset = " << s.triangleOffset << " numTriangles = " << s.numTriangles << std::endl;
    */

    // Populate gltf node id to shape index map.
    for (size_t i = 0; i < mesh.numShapes; ++i)
        properties.gltfNodeIdToShapeIndexMap.insert({mesh.shapes[i].nodeId, i});

    // Animations.
    loadAnimations();

    // Emissive triangles.
    for (auto &s : mesh.shapes) {
        for (int i = 0; i < s.numTriangles; ++i) {
            size_t globalTriangleIndex = s.triangleOffset + i;
            int matIndex = mesh.matIndices[globalTriangleIndex];
            if (matIndex != -1) {
                glm::vec3 &emission = mesh.matParams[matIndex].emission;
                if (emission.x > 0.0f || emission.y > 0.0f || emission.z > 0.0f ||
                    mesh.matParams[matIndex].emissiveTexPresent) {
                    s.emissiveTriangles.push_back(i);
                }
            }
        }
    }
}

bool GltfLoader::isCameraPresent()
{
    return cameraPresent;
}

float GltfLoader::getCamAspectRatio()
{
    return camAspectRatio;
}

float GltfLoader::getCamFovYDeg()
{
    return camFovYDeg;
}

glm::vec3 GltfLoader::getCamOrigin()
{
    return camOrigin;
}

glm::vec3 GltfLoader::getCamTarget()
{
    return camTarget;
}

void GltfLoader::printLoaderNode(size_t index)
{
    auto &node = properties.nodes[index];

    std::cout << "Node " << node.id << std::endl;

    std::cout << "  Children : ";
    for (auto &c : node.children)
        std::cout << properties.nodes[c].id << " ";
    std::cout <<std::endl;

    for (auto &c : node.children)
        printLoaderNode(c);
}

void GltfLoader::scanScene()
{
    numMeshes = 0;
    numTriangles = 0;
    numVertices = 0;
    numNormals = 0;
    numTexCoords = 0;
    numIndices = 0;

    verticesIndex = 0;
    normalsIndex = 0;
    texCoordsIndex = 0;
    triIndicesIndex = 0;

    triIndicesBase = 0;
    meshIndex = 0;

    size_t numRootNodes = scene->nodes.size();
    for (size_t i = 0; i < numRootNodes; ++i)
        scanNode(scene->nodes[i], 0, 0);

    markJoints();

    /*
    for (auto &i : rootNodeIds) {
        printLoaderNode(i);
        std::cout << "----" << std::endl;
    }
    */

    /*
    std::cout << "numMeshes    = " << numMeshes << std::endl;
    std::cout << "numTriangles = " << numTriangles << std::endl;
    std::cout << "numVertices  = " << numVertices << std::endl;
    std::cout << "numNormals   = " << numNormals << std::endl;
    std::cout << "numTexCoords = " << numTexCoords << std::endl;
    std::cout << "numIndices   = " << numIndices << std::endl;
    std::cout << "numMaterials = " << model.materials.size() << std::endl; // all scenes
    */
}

void GltfLoader::scanNode(int gltfNodeId, size_t parentNodeIndex, int depth)
{
    tinygltf::Node &node = model.nodes[gltfNodeId];

    SceneNode loaderNode;
    loaderNode.id = gltfNodeId;

    properties.nodes.push_back(loaderNode);
    size_t loaderNodeIndex = properties.nodes.size() - 1;
    properties.gltfNodeIdToNodeIndexMap.insert({gltfNodeId, loaderNodeIndex});

    if (depth == 0)
        properties.rootNodeIndices.push_back(loaderNodeIndex);
    else
        properties.nodes[parentNodeIndex].children.push_back(loaderNodeIndex);

    if (node.mesh != -1) {
        properties.nodes[loaderNodeIndex].isMesh = true;
        if (node.skin != -1)
            properties.nodes[loaderNodeIndex].hasSkin = true;

        numMeshes++;
        tinygltf::Mesh &mesh = model.meshes[node.mesh];

        for (auto &primitive : mesh.primitives) {
            assert(primitive.mode == TINYGLTF_MODE_TRIANGLES);

            size_t localNumVertices = 0;

            std::map<std::string, int>::iterator pos;

            pos = primitive.attributes.find("POSITION");
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                localNumVertices = accessor.count;
                numVertices += accessor.count;
            }

            pos = primitive.attributes.find("NORMAL");
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                numNormals += accessor.count;
            }

            pos = primitive.attributes.find("TEXCOORD_0");
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                numTexCoords += accessor.count;
            }

            if (primitive.indices != -1) {
                tinygltf::Accessor &accessor = model.accessors[primitive.indices];
                numIndices += accessor.count;
                numTriangles += accessor.count / 3;
            }
            else {
                numTriangles += localNumVertices / 3;
            }
        }
    }

    for (auto childId : node.children)
        scanNode(childId, loaderNodeIndex, depth + 1);
}

void GltfLoader::markJoints()
{
    for (auto &s : model.skins) {
        for (auto j : s.joints) {
            size_t nodeIndex = properties.gltfNodeIdToNodeIndexMap.find(j)->second;
            properties.nodes[nodeIndex].isJoint = true;
        }
    }
}

void GltfLoader::parseTextureTransform(tinygltf::Value::Object &tt, TextureTransform *texTf, int *texCoord)
{
    if (texCoord)
        *texCoord = -1;

    auto offsetIt = tt.find("offset");
    if (offsetIt != tt.end() && offsetIt->second.IsArray() && offsetIt->second.ArrayLen() == 2) {
        auto offsetArray = offsetIt->second.Get<tinygltf::Value::Array>();
        glm::vec2 offset;
        offset.x = offsetArray[0].Get<double>();
        offset.y = offsetArray[1].Get<double>();
        std::printf("      offset=(%f, %f)\n", offset.x, offset.y);
        texTf->offset = offset;
    }

    auto rotationIt = tt.find("rotation");
    if (rotationIt != tt.end() && rotationIt->second.IsNumber()) {
        float rotation = rotationIt->second.Get<double>();
        std::printf("      rotation=%f\n", rotation);
        texTf->rotation = rotation;
    }

    auto scaleIt = tt.find("scale");
    if (scaleIt != tt.end() && scaleIt->second.IsArray() && scaleIt->second.ArrayLen() == 2) {
        auto scaleArray = scaleIt->second.Get<tinygltf::Value::Array>();
        glm::vec2 scale;
        scale.x = scaleArray[0].Get<double>();
        scale.y = scaleArray[1].Get<double>();
        std::printf("      scale=(%f, %f)\n", scale.x, scale.y);
        texTf->scale = scale;
    }

    auto texCoordIt = tt.find("texCoord");
    if (texCoordIt != tt.end() && texCoordIt->second.IsInt()) {
        int texCoordTemp = texCoordIt->second.Get<int>();
        std::printf("      texCoord=%d\n", texCoordTemp);
        if (texCoord)
            *texCoord = texCoordTemp;
    }
}

void GltfLoader::parseTextureTransformParent(tinygltf::Value::Object &ttParent, TextureTransform *texTf, int *texCoord)
{
    if (texCoord)
        *texCoord = -1;

    auto it = ttParent.find("KHR_texture_transform");
    if (it == ttParent.end() || !it->second.IsObject())
        return;

    auto &tt = it->second.Get<tinygltf::Value::Object>();

    parseTextureTransform(tt, texTf, texCoord);
}

void GltfLoader::parseSheen(tinygltf::Value::Object obj, SheenData *data)
{
    std::printf("  parseSheen()\n");

    auto colFactorIt = obj.find("sheenColorFactor");
    if (colFactorIt != obj.end() && colFactorIt->second.IsArray() && colFactorIt->second.ArrayLen() == 3) {
        auto factorArray = colFactorIt->second.Get<tinygltf::Value::Array>();
        data->colourFactor = glm::vec3(
            factorArray[0].Get<double>(),
            factorArray[1].Get<double>(),
            factorArray[2].Get<double>());
        std::printf("    sheenColorFactor=(%f, %f, %f)\n", data->colourFactor.x, data->colourFactor.y,
            data->colourFactor.z);

        // textureInfo
        auto colourTexIt = obj.find("sheenColorTexture");
        if (colourTexIt != obj.end() && colourTexIt->second.IsObject()) {
            std::printf("    sheenColorTexture\n");

            auto colourTexObj = colourTexIt->second.Get<tinygltf::Value::Object>();

            // index
            auto it = colourTexObj.find("index");
            if (it != colourTexObj.end() && it->second.IsInt()) {
                data->colourTexIndex = it->second.Get<int>();
                std::printf("      index=%d\n", data->colourTexIndex);
            }

            // texCoord
            it = colourTexObj.find("texCoord");
            if (it != colourTexObj.end() && it->second.IsInt()) {
                data->colourTexCoord = it->second.Get<int>();
                std::printf("      texCoord=%d\n", data->colourTexCoord);
            }

            // extensions
            it = colourTexObj.find("extensions");
            if (it != colourTexObj.end() && it->second.IsObject()) {
                auto extensionsObj = it->second.Get<tinygltf::Value::Object>();

                // KHR_texture_transform
                int texCoord = -1;
                parseTextureTransformParent(extensionsObj, &data->colourTexTf, &texCoord);
                if (texCoord != -1)
                    data->colourTexCoord = texCoord;
            }
        }
    }

    auto roughnessFactorIt = obj.find("sheenRoughnessFactor");
    if (roughnessFactorIt != obj.end() && roughnessFactorIt->second.IsNumber()) {
        data->roughnessFactor = roughnessFactorIt->second.Get<double>();
        std::printf("    sheenRoughnessFactor=%f\n", data->roughnessFactor);

        // textureInfo
        auto roughnessTexIt = obj.find("sheenRoughnessTexture");
        if (roughnessTexIt != obj.end() && roughnessTexIt->second.IsObject()) {
            std::printf("    sheenRoughnessTexture\n");

            auto roughnessTexObj = roughnessTexIt->second.Get<tinygltf::Value::Object>();

            // index
            auto it = roughnessTexObj.find("index");
            if (it != roughnessTexObj.end() && it->second.IsInt()) {
                data->roughnessTexIndex = it->second.Get<int>();
                std::printf("      index=%d\n", data->roughnessTexIndex);
            }

            // texCoord
            it = roughnessTexObj.find("texCoord");
            if (it != roughnessTexObj.end() && it->second.IsInt()) {
                data->roughnessTexCoord = it->second.Get<int>();
                std::printf("      texCoord=%d\n", data->roughnessTexCoord);
            }

            // extensions
            it = roughnessTexObj.find("extensions");
            if (it != roughnessTexObj.end() && it->second.IsObject()) {
                auto extensionsObj = it->second.Get<tinygltf::Value::Object>();

                // KHR_texture_transform
                int texCoord = -1;
                parseTextureTransformParent(extensionsObj, &data->roughnessTexTf, &texCoord);
                if (texCoord != -1)
                    data->roughnessTexCoord = texCoord;
            }
        }
    }
}

void GltfLoader::initMaterialParams(std::vector<MaterialParams> &matParamsColl)
{
    // metallic    : material.pbrMetallicRoughness.metallicFactor or
    //               material.pbrMetallicRoughness.metallicRoughnessTexture
    // base colour : material.pbrMetallicRoughness.baseColorFactor or
    //               material.pbrMetallicRoughness.baseColorTexture
    // roughness   : material.pbrMetallicRoughness.roughnessFactor
    // normal      : material.normalTexture
    // emissive    : material.emissiveFactor or
    //               material.emissiveTexture

    // KHR_materials_specular
    // specular (IOR level) : "specularFactor" (float)
    // specular tint : "specularColorFactor" (3 x float)

    // KHR_materials_transmission
    // transmission : "transmissionFactor" (float) or
    //                "transmissionTexture"
    // transmission roughness : use roughness?
    // tint         : use 1.0 - baseColor

    // KHR_materials_ior
    // IOR          : "ior" (float)

    // KHR_materials_emissive_strength
    // (multiplier) : "emissiveStrength" (float)

    // alphaMode
    // - OPAQUE : ignore base colour alpha (default)
    // - BLEND  : use base colour alpha
    // - MASK   : values below alphaCutoff are not rendered (fully transparent); everything else is opaque
    // alphaCutoff
    // - default is 0.5

    // doublesided : enable/disable backface culling (default is false)
    // occlusionTexture : baked-in ambient occlusion

    for (uint64_t i = 0; i < model.materials.size(); ++i) {
        tinygltf::Material &material = model.materials[i];
        MaterialParams matParams;
        float metallic  = 1.0f; // default is 1.0
        float ior       = 1.5f; // default is 1.5
        float roughness = 1.0f; // default is 1.0

        std::printf("Material %lld: %s\n", i, material.name.c_str());

        std::printf("  Extensions:\n");

        auto specularExt = material.extensions.find("KHR_materials_specular");
        if (specularExt != material.extensions.end())
            std::printf("    KHR_materials_specular\n");

        auto transmissionExt = material.extensions.find("KHR_materials_transmission");
        if (transmissionExt != material.extensions.end())
            std::printf("    KHR_materials_transmission\n");

        auto iorExt = material.extensions.find("KHR_materials_ior");
        if (iorExt != material.extensions.end())
            std::printf("    KHR_materials_ior\n");

        auto emissiveStrengthExt = material.extensions.find("KHR_materials_emissive_strength");
        if (emissiveStrengthExt != material.extensions.end())
            std::printf("    KHR_materials_emissive_strength\n");

        auto clearcoatExt = material.extensions.find("KHR_materials_clearcoat");
        if (clearcoatExt != material.extensions.end())
            std::printf("    KHR_materials_clearcoat\n");

        auto sheenExt = material.extensions.find("KHR_materials_sheen");
        if (sheenExt != material.extensions.end())
            std::printf("    KHR_materials_sheen\n");

        auto textureTransformExt = material.extensions.find("KHR_texture_transform");
        if (textureTransformExt != material.extensions.end())
            std::printf("    KHR_texture_transform\n");

        //
        // Base colour and alpha
        //

        auto &bcf = material.pbrMetallicRoughness.baseColorFactor; // default is [1, 1, 1, 1]
        glm::vec4 baseColour = glm::vec4(bcf[0], bcf[1], bcf[2], bcf[3]);
        std::printf("  base colour=(%f, %f, %f, %f)\n", baseColour.x, baseColour.y, baseColour.z, baseColour.w);

        int baseColourTexIndex = material.pbrMetallicRoughness.baseColorTexture.index;
        if (baseColourTexIndex != -1) {
            tinygltf::Texture &texture = model.textures[baseColourTexIndex];
            tinygltf::Image &image = model.images[texture.source];
            assert(image.component == 4);

            matParams.baseColourTexPresent = true;
            matParams.baseColourTexWidth = image.width;
            matParams.baseColourTexHeight = image.height;
            matParams.baseColourTexNumComponents = image.component;
            matParams.baseColourTexRgba = new unsigned char[image.image.size()];
            if (matParams.baseColourTexRgba)
                memcpy(matParams.baseColourTexRgba, image.image.data(), image.image.size());
            else
                std::cout << "  Failed to load image!" << std::endl;

            if (texture.sampler != -1)
                matParams.baseColourTexSampler = &model.samplers[texture.sampler];

            // extensions
            tinygltf::ExtensionMap &extensionMap = material.pbrMetallicRoughness.baseColorTexture.extensions;
            auto it = extensionMap.find("KHR_texture_transform");
            if (it != extensionMap.end() && it->second.IsObject()) {
                auto tt = it->second.Get<tinygltf::Value::Object>();

                // KHR_texture_transform
                int texCoord = -1;
                parseTextureTransform(tt, &matParams.baseColourTexTf, &texCoord);
                //if (texCoord != -1)
                //    baseColourTexCoord = texCoord;
            }
        }
        else {
            matParams.baseColourTexPresent = false;
        }

        std::printf("  base colour texture=%s\n", matParams.baseColourTexPresent ? "True" : "False");

        AlphaMode alphaMode = AM_OPAQUE;
        if (material.alphaMode == "BLEND")
            alphaMode = AM_BLEND;
        else if (material.alphaMode == "MASK")
            alphaMode = AM_MASK;

        float alphaCutoff = material.alphaCutoff;

        std::printf("  alphaMode=%d\n", alphaMode);
        std::printf("  alphaCutoff=%f\n", alphaCutoff);

        //
        // Normal
        //

        int normalTexIndex = material.normalTexture.index;
        if (normalTexIndex != -1) {
            tinygltf::Texture &texture = model.textures[normalTexIndex];
            tinygltf::Image &image = model.images[texture.source];
            assert(image.component == 4);

            matParams.normalTexPresent = true;
            matParams.normalTexWidth = image.width;
            matParams.normalTexHeight = image.height;
            matParams.normalTexNumComponents = image.component;
            matParams.normalTexRgba = new unsigned char[image.image.size()];
            if (matParams.normalTexRgba)
                memcpy(matParams.normalTexRgba, image.image.data(), image.image.size());
            else
                std::cout << "  Failed to load image!" << std::endl;

            if (texture.sampler != -1)
                matParams.normalTexSampler = &model.samplers[texture.sampler];

            // extensions
            tinygltf::ExtensionMap &extensionMap = material.normalTexture.extensions;
            auto it = extensionMap.find("KHR_texture_transform");
            if (it != extensionMap.end() && it->second.IsObject()) {
                auto tt = it->second.Get<tinygltf::Value::Object>();

                // KHR_texture_transform
                int texCoord = -1;
                parseTextureTransform(tt, &matParams.normalTexTf, &texCoord);
                //if (texCoord != -1)
                //    baseColourTexCoord = texCoord;
            }
        }
        else {
            matParams.normalTexPresent = false;
        }

        std::printf("  normal texture=%s\n", matParams.normalTexPresent ? "True" : "False");

        //
        // Metallic
        //

        // Metal = specular only, tinted with the base colour
        // Non-metal = dielectric i.e. diffuse/transmissive and specular
        // 1.0 = fully specular
        // 0.0 = fully dielectric

        metallic = material.pbrMetallicRoughness.metallicFactor;
        std::printf("  metallicFactor=%f\n", metallic);

        int metallicRoughnessTexIndex = material.pbrMetallicRoughness.metallicRoughnessTexture.index;
        if (metallicRoughnessTexIndex != -1) {
            tinygltf::Texture &texture = model.textures[metallicRoughnessTexIndex];
            tinygltf::Image &image = model.images[texture.source];
            assert(image.component == 4);

            matParams.metallicRoughnessTexPresent = true;
            matParams.metallicRoughnessTexWidth = image.width;
            matParams.metallicRoughnessTexHeight = image.height;
            matParams.metallicRoughnessTexNumComponents = image.component;
            matParams.metallicRoughnessTexRgba = new unsigned char[image.image.size()];
            if (matParams.metallicRoughnessTexRgba)
                memcpy(matParams.metallicRoughnessTexRgba, image.image.data(), image.image.size());
            else
                std::cout << "  Failed to load image!" << std::endl;

            // extensions
            tinygltf::ExtensionMap &extensionMap = material.pbrMetallicRoughness.metallicRoughnessTexture.extensions;
            auto it = extensionMap.find("KHR_texture_transform");
            if (it != extensionMap.end() && it->second.IsObject()) {
                auto tt = it->second.Get<tinygltf::Value::Object>();

                // KHR_texture_transform
                int texCoord = -1;
                parseTextureTransform(tt, &matParams.metallicRoughnessTexTf, &texCoord);
                //if (texCoord != -1)
                //    metallicRoughnessTexCoord = texCoord;
            }
        }
        else {
            matParams.metallicRoughnessTexPresent = false;
        }

        std::printf("  metallic roughness texture=%s\n", matParams.metallicRoughnessTexPresent ? "True" : "False");

        //
        // Roughness
        //

        roughness = material.pbrMetallicRoughness.roughnessFactor;
        std::printf("  roughnessFactor=%f\n", roughness);

        //
        // Reflection
        //

        float specular = 1.0f; // IOR level aka specular factor; default is 1.0
        glm::vec3 specularTint { 1.0f }; // F0 colour aka specular colour factor; default is [1.0, 1.0, 1.0]

        //auto ext = material.extensions.find("KHR_materials_specular");
        if (specularExt != material.extensions.end()) {
            if (specularExt->second.Has("specularFactor"))
                specular = specularExt->second.Get("specularFactor").Get<double>();

            if (specularExt->second.Has("specularColorFactor")) {
                auto factor = specularExt->second.Get("specularColorFactor").Get<tinygltf::Value::Array>();
                if (factor.size() == 3)
                    specularTint = glm::vec3(factor[0].Get<double>(), factor[1].Get<double>(), factor[2].Get<double>());
            }
        }

        //
        // Transmission
        //

        float transmissionFactor = 0.0f; // default is 0.0

        //ext = material.extensions.find("KHR_materials_transmission");
        if (transmissionExt != material.extensions.end()) {
            if (transmissionExt->second.Has("transmissionFactor"))
                transmissionFactor = transmissionExt->second.Get("transmissionFactor").Get<double>();

            /*
            // Currently not using the transmission texture.
            if (transmissionExt->second.Has("transmissionTexture")) {
                auto transmissionTexture = transmissionExt->second.Get("transmissionTexture");
                std::printf("  transmissionTexture=%d\n", transmissionTexture.Get<int>());

                if (transmissionTexture.Has("index")) {
                    auto index = transmissionTexture.Get("index");
                    std::printf("  transmissionTexture index=%d\n", index.Get<int>());
                }
            }
            */
        }

        //ext = material.extensions.find("KHR_materials_ior");
        if (iorExt != material.extensions.end()) {
            if (iorExt->second.Has("ior"))
                ior = iorExt->second.Get("ior").Get<double>();
        }
        
        std::printf("  Reflection\n");
        std::printf("    specular=%f\n", specular);
        std::printf("    specular tint=(%f, %f, %f)\n", specularTint[0], specularTint[1], specularTint[2]);

        std::printf("  Transmission\n");
        std::printf("    transmission factor=%f\n", transmissionFactor);
        std::printf("    ior=%f\n", ior);

        //
        // Emission
        //

        // Emission/emissive strength
        // Point, spot and area lights: watts
        // Sun and mesh lights: watts / m^2.
        // See https://docs.blender.org/manual/en/latest/render/shader_nodes/shader/emission.html

        std::vector<double> &eFactor = material.emissiveFactor; // emission colour, default is [0, 0, 0]
        float emissionStrength = 1.0f; // default is 1.0
        matParams.emissiveTexPresent = false;

        if (glm::fmax(eFactor[0], eFactor[1], eFactor[2]) > 0.0f) {
            std::printf("  Emission\n");
            std::printf("    emission factor=(%f, %f, %f)\n", eFactor[0], eFactor[1], eFactor[2]);

            //ext = material.extensions.find("KHR_materials_emissive_strength");
            if (emissiveStrengthExt != material.extensions.end()) {
                if (emissiveStrengthExt->second.Has("emissiveStrength"))
                    emissionStrength = emissiveStrengthExt->second.Get("emissiveStrength").Get<double>();
            }

            std::printf("    emission strength=%f\n", emissionStrength);

            int emissiveTexIndex = material.emissiveTexture.index;
            if (emissiveTexIndex != -1) {
                tinygltf::Texture &texture = model.textures[emissiveTexIndex];
                tinygltf::Image &image = model.images[texture.source];
                assert(image.component == 4);

                matParams.emissiveTexPresent = true;
                matParams.emissiveTexWidth = image.width;
                matParams.emissiveTexHeight = image.height;
                matParams.emissiveTexNumComponents = image.component;
                matParams.emissiveTexRgba = new unsigned char[image.image.size()];
                if (matParams.emissiveTexRgba)
                    memcpy(matParams.emissiveTexRgba, image.image.data(), image.image.size());
                else
                    std::cout << "  Failed to load image!" << std::endl;

                // extensions
                tinygltf::ExtensionMap &extensionMap = material.emissiveTexture.extensions;
                auto it = extensionMap.find("KHR_texture_transform");
                if (it != extensionMap.end() && it->second.IsObject()) {
                    auto tt = it->second.Get<tinygltf::Value::Object>();

                    // KHR_texture_transform
                    int texCoord = -1;
                    parseTextureTransform(tt, &matParams.emissiveTexTf, &texCoord);
                    //if (texCoord != -1)
                    //    metallicRoughnessTexCoord = texCoord;
                }
            }

            std::printf("  emissive texture=%s\n", matParams.emissiveTexPresent ? "True" : "False");
        }

        //
        // Clearcoat
        //

        float clearcoatFactor = 0.0f;
        float clearcoatRoughnessFactor = 0.0f;

        if (clearcoatExt != material.extensions.end()) {
            if (clearcoatExt->second.Has("clearcoatFactor"))
                clearcoatFactor = clearcoatExt->second.Get("clearcoatFactor").Get<double>();

            if (clearcoatExt->second.Has("clearcoatRoughnessFactor"))
                clearcoatRoughnessFactor = clearcoatExt->second.Get("clearcoatRoughnessFactor").Get<double>();
        }

        std::printf("  Clearcoat\n");
        std::printf("    clearcoatFactor=%f\n", clearcoatFactor);
        std::printf("    clearcoatRoughnessFactor=%f\n", clearcoatRoughnessFactor);

        //
        // Sheen
        //

        SheenData sheenData;
        matParams.sheenColourTexPresent = false;
        matParams.sheenRoughnessTexPresent = false;

        if (sheenExt != material.extensions.end()) {
            parseSheen(sheenExt->second.Get<tinygltf::Value::Object>(), &sheenData);

            // Optimisation (later).
            //int sheenColourTexSource = -1;
            //int sheenRoughnessTexSource = -1;

            if (sheenData.colourTexIndex != -1) {
                tinygltf::Texture &texture = model.textures[sheenData.colourTexIndex];
                tinygltf::Image &image = model.images[texture.source];
                assert(image.component == 4);

                // Optimisation (later).
                //sheenColourTexSource = texture.source;

                matParams.sheenColourTexPresent = true;
                matParams.sheenColourTexWidth = image.width;
                matParams.sheenColourTexHeight = image.height;
                matParams.sheenColourTexNumComponents = image.component;
                matParams.sheenColourTexRgba = new unsigned char[image.image.size()];
                if (matParams.sheenColourTexRgba)
                    memcpy(matParams.sheenColourTexRgba, image.image.data(), image.image.size());
                else
                    std::cout << "  Failed to load image!" << std::endl;
            }

            if (sheenData.roughnessTexIndex != -1) {
                tinygltf::Texture &texture = model.textures[sheenData.roughnessTexIndex];
                tinygltf::Image &image = model.images[texture.source];
                assert(image.component == 4);

                // Optimisation (later).
                //sheenRoughnessTexSource = texture.source;

                matParams.sheenRoughnessTexPresent = true;
                matParams.sheenRoughnessTexWidth = image.width;
                matParams.sheenRoughnessTexHeight = image.height;
                matParams.sheenRoughnessTexNumComponents = image.component;

                /*
                // Optimisation (later) (need to update MaterialData
                // initialisation too).
                if (sheenRoughnessTexSource == sheenColourTexSource) {
                    matParams.sheenRoughnessTexRgba = matParams.sheenColourTexRgba;
                }
                else {
                */
                    matParams.sheenRoughnessTexRgba = new unsigned char[image.image.size()];
                    if (matParams.sheenRoughnessTexRgba)
                        memcpy(matParams.sheenRoughnessTexRgba, image.image.data(), image.image.size());
                    else
                        std::cout << "  Failed to load image!" << std::endl;
                /*
                }
                */
            }
        }

        std::printf("  Sheen\n");
        std::printf("    sheenColourFactor=(%f, %f, %f)\n", sheenData.colourFactor.x, sheenData.colourFactor.y,
            sheenData.colourFactor.z);
        std::printf("    sheenRoughnessFactor=%f\n", sheenData.roughnessFactor);

        //
        // Consolidation
        //

        glm::vec3 emission = glm::vec3(eFactor[0], eFactor[1], eFactor[2]);
        emission *= emissionStrength;
        std::printf("  emission=(%f, %f, %f)\n", emission[0], emission[1], emission[2]);

        matParams.init(
            material.name.c_str(),
            baseColour,
            alphaMode, alphaCutoff,
            glm::vec3(emission[0], emission[1], emission[2]),
            ior,
            roughness,
            metallic,
            specular,
            transmissionFactor,
            clearcoatFactor, clearcoatRoughnessFactor,
            &sheenData);

        matParams.printFlags("  ");
        std::printf("--------\n");

        matParamsColl[i] = matParams;
    }
}

void GltfLoader::loadNode(int nodeId, std::vector<Light> &lights, glm::mat4 &globalTf)
{
    // nodeId is the tinygltf node ID.

    tinygltf::Node &node = model.nodes[nodeId];
    updateGlobalTransform(node, nodeId, globalTf);

    if (node.extensions.size() != 0) {
        tinygltf::Value::Object::iterator pos = node.extensions.find("KHR_lights_punctual");
        if (pos != node.extensions.end())
            loadLight(pos, lights, globalTf);
    }

    if (node.camera != -1 && !cameraPresent)
        loadCamera(node.camera, globalTf);

    if (node.mesh != -1) {
        tinygltf::Mesh &mesh = model.meshes[node.mesh];
        //std::cout << mesh.name << std::endl;

        Shape &shape = properties.mesh.shapes[meshIndex];

        shape.nodeId = nodeId;
        shape.triangleOffset = triIndicesIndex;
        shape.numVertices = 0;
        shape.numNormals = 0;
        shape.numTexCoords = 0;
        shape.numTriIndices = 0;

        shape.verticesIndex = verticesIndex;
        shape.normalsIndex = normalsIndex;

        shape.hasSkin = false;
        shape.skinIndex = std::numeric_limits<size_t>::max();

        size_t primitivesTriIndicesBase = 0;

        for (auto &primitive : mesh.primitives) {
            assert(primitive.mode == TINYGLTF_MODE_TRIANGLES);

            size_t primitiveNumVertices = 0;
            std::map<std::string, int>::iterator pos;

            pos = primitive.attributes.find("POSITION");
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC3);

                tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
                unsigned char *ptrByteOffset = &buffer.data[byteOffset];

                primitiveNumVertices = accessor.count;
                shape.numVertices += accessor.count;

                for (size_t i = 0; i < accessor.count; ++i) {
                    glm::vec3 v = *(glm::vec3 *)ptrByteOffset;
                    // Log
                    //std::cout << std::fixed << "Raw : pos = (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
                    //printf("Raw : pos = (% .2f, % .2f, % .2f)\n", v.x, v.y, v.z);

                    // The global transform should not be applied here. It is applied in updateScene().
                    properties.mesh.positions[verticesIndex++] = glm::vec3(glm::vec4(v, 1.0f));
                    //properties.mesh.positions[verticesIndex++] = glm::vec3(globalTf * glm::vec4(v, 1.0f));

                    //v = properties.mesh.positions[verticesIndex - 1];
                    //std::cout << std::fixed << "Tf  : pos = (" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
                    //printf("Tf  : pos = (% .2f, % .2f, % .2f)\n", v.x, v.y, v.z);

                    ptrByteOffset += accessor.ByteStride(bufferView);
                }
            }

            pos = primitive.attributes.find("NORMAL");
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC3);

                tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
                unsigned char *ptrByteOffset = &buffer.data[byteOffset];

                shape.numNormals += accessor.count;

                for (size_t i = 0; i < accessor.count; ++i) {
                    glm::vec3 n = *(glm::vec3 *)ptrByteOffset;
                    // Log
                    //std::cout << std::fixed << "(" << n.x << ", " << n.y << ", " << n.z << ")" << std::endl;

                    // Normals need to be transformed by the transpose of the inverse of the matrix mat so that they are
                    // unaffected by non-uniform scaling. See
                    // "https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/
                    // transforming-normals" for a detailed explanation.
                    glm::mat4 modifiedGlobalTf = glm::transpose(glm::inverse(globalTf));

                    // The global transform should not be applied here. It is applied in updateScene().
                    properties.mesh.normals[normalsIndex++] = glm::vec3(glm::vec4(n, 1.0f));
                    //properties.mesh.normals[normalsIndex++] = glm::vec3(modifiedGlobalTf * glm::vec4(n, 1.0f));

                    ptrByteOffset += accessor.ByteStride(bufferView);
                }
            }

            pos = primitive.attributes.find("TEXCOORD_0");
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC2);

                tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
                unsigned char *ptrByteOffset = &buffer.data[byteOffset];

                shape.numTexCoords += accessor.count;

                for (size_t i = 0; i < accessor.count; ++i) {
                    // Log
                    //glm::vec2 *t = (glm::vec2 *)ptrByteOffset;
                    //std::cout << std::fixed << "(" << t->x << ", " << t->y << ")" << std::endl;
                    properties.mesh.texCoords[texCoordsIndex++] = *(glm::vec2 *)ptrByteOffset;
                    ptrByteOffset += accessor.ByteStride(bufferView);
                }
            }
            else if (properties.mesh.hasTexCoords) {
                // Create dummy texture coordinates.
                shape.numTexCoords += shape.numVertices;
                for (size_t i = 0; i < shape.numVertices; ++i)
                    properties.mesh.texCoords[texCoordsIndex++] = glm::vec2(0.0f);
            }

            if (primitive.indices != -1) {
                tinygltf::Accessor &accessor = model.accessors[primitive.indices];
                // Assume unsigned int indices do not exceed the range of signed int. 
                assert(
                    accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ||
                    accessor.componentType == TINYGLTF_COMPONENT_TYPE_INT ||
                    accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT);
                assert(accessor.type == TINYGLTF_TYPE_SCALAR);

                tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
                unsigned char *ptrByteOffset = &buffer.data[byteOffset];

                shape.numTriIndices += accessor.count;

                for (size_t i = 0; i < accessor.count; i += 3) {
                    glm::ivec3 idx;

                    // Every mesh's indices are zero-based.
                    // Every mesh primitive's indices are zero-based.

                    if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT) {
                        idx.x = *(unsigned short *)ptrByteOffset + static_cast<int>(primitivesTriIndicesBase);
                        ptrByteOffset += accessor.ByteStride(bufferView);
                        idx.y = *(unsigned short *)ptrByteOffset + static_cast<int>(primitivesTriIndicesBase);
                        ptrByteOffset += accessor.ByteStride(bufferView);
                        idx.z = *(unsigned short *)ptrByteOffset + static_cast<int>(primitivesTriIndicesBase);
                        ptrByteOffset += accessor.ByteStride(bufferView);
                    }
                    else if (accessor.componentType == TINYGLTF_COMPONENT_TYPE_INT ||
                        accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT) {
                        idx.x = *(int *)ptrByteOffset + static_cast<int>(primitivesTriIndicesBase);
                        ptrByteOffset += accessor.ByteStride(bufferView);
                        idx.y = *(int *)ptrByteOffset + static_cast<int>(primitivesTriIndicesBase);
                        ptrByteOffset += accessor.ByteStride(bufferView);
                        idx.z = *(int *)ptrByteOffset + static_cast<int>(primitivesTriIndicesBase);
                        ptrByteOffset += accessor.ByteStride(bufferView);
                    }

                    // Log
                    //std::cout << "i = (" << idx.x << ", " << idx.y << ", " << idx.z << ")" << std::endl;
                    properties.mesh.triIndices[triIndicesIndex] = idx;
                    properties.mesh.matIndices[triIndicesIndex] = primitive.material;
                    triIndicesIndex++;
                }

                primitivesTriIndicesBase += primitiveNumVertices;
                triIndicesBase = verticesIndex;
            }
            else {
                // Generate indices, assuming that every 3 consecutive vertices
                // make up a triangle.

                size_t numTriangles = primitiveNumVertices / 3;
                shape.numTriIndices += numTriangles;

                for (size_t i = 0; i < numTriangles; ++i) {
                    properties.mesh.triIndices[triIndicesIndex] = glm::ivec3(
                        primitivesTriIndicesBase + i * 3 + 0,
                        primitivesTriIndicesBase + i * 3 + 1,
                        primitivesTriIndicesBase + i * 3 + 2);
                    properties.mesh.matIndices[triIndicesIndex] = primitive.material;
                    triIndicesIndex++;
                }

                primitivesTriIndicesBase += primitiveNumVertices;
                triIndicesBase = verticesIndex;
            }

            pos = primitive.attributes.find("JOINTS_0"); // node indices
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT);
                assert(accessor.type == TINYGLTF_TYPE_VEC4);

                tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
                unsigned char *ptrByteOffset = &buffer.data[byteOffset];

                //shape.joints.resize(accessor.count);

                for (size_t i = 0; i < accessor.count; ++i) {
                    glm::u16vec4 v = *(glm::u16vec4 *)ptrByteOffset;
                    // Log
                    //std::cout << "vertex " << i << " : joints = (" << v.x << ", " << v.y << ", " << v.z << ", " <<
                    //    v.w << ")" << std::endl;
                    //shape.joints[i] = v;
                    shape.joints.push_back(v);

                    ptrByteOffset += accessor.ByteStride(bufferView);
                }
            }

            pos = primitive.attributes.find("WEIGHTS_0");
            if (pos != primitive.attributes.end()) {
                tinygltf::Accessor &accessor = model.accessors[pos->second];
                assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
                assert(accessor.type == TINYGLTF_TYPE_VEC4);

                tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
                tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

                size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
                unsigned char *ptrByteOffset = &buffer.data[byteOffset];

                //shape.weights.resize(accessor.count);

                for (size_t i = 0; i < accessor.count; ++i) {
                    glm::vec4 v = *(glm::vec4 *)ptrByteOffset;
                    // Log
                    //std::cout << "vertex " << i << std::fixed << " : weights = (" << v.x << ", " << v.y << ", " <<
                    //    v.z << ", " << v.w << ")" << std::endl;
                    //shape.weights[i] = v;
                    shape.weights.push_back(v);

                    ptrByteOffset += accessor.ByteStride(bufferView);
                }
            }
        }

        shape.numTriangles = triIndicesIndex - shape.triangleOffset;

        if (node.skin != -1) {
            Skin loaderSkin;
            properties.skins.push_back(loaderSkin);
            size_t loaderSkinIndex = properties.skins.size() - 1;

            tinygltf::Skin skin = model.skins[node.skin];

            tinygltf::Accessor &accessor = model.accessors[skin.inverseBindMatrices];
            assert(accessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
            assert(accessor.type == TINYGLTF_TYPE_MAT4);

            tinygltf::BufferView &bufferView = model.bufferViews[accessor.bufferView];
            tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

            size_t byteOffset = accessor.byteOffset + bufferView.byteOffset;
            unsigned char *ptrByteOffset = &buffer.data[byteOffset];

            assert(accessor.count == skin.joints.size());

            for (size_t i = 0; i < accessor.count; ++i) {
                glm::mat4 mat = *(glm::mat4 *)ptrByteOffset;
                // Log
                //std::cout << "<InverseBindMatrix>" << std::endl;

                //float *matData = glm::value_ptr(mat);
                //std::cout << i << " :" << std::endl;
                //std::cout << std::fixed << matData[0] << " " << matData[4] << " " << matData[ 8] << " " << matData[12] << std::endl;
                //std::cout << std::fixed << matData[1] << " " << matData[5] << " " << matData[ 9] << " " << matData[13] << std::endl;
                //std::cout << std::fixed << matData[2] << " " << matData[6] << " " << matData[10] << " " << matData[14] << std::endl;
                //std::cout << std::fixed << matData[3] << " " << matData[7] << " " << matData[11] << " " << matData[15] << std::endl;

                properties.skins[loaderSkinIndex].inverseBindMatrices.push_back(mat);
                ptrByteOffset += accessor.ByteStride(bufferView);
            }

            //std::cout << "Joints (nodes):" << std::endl;
            for (auto &j : skin.joints) {
                //std::cout << j << std::endl;
                properties.skins[loaderSkinIndex].joints.push_back(j);
            }

            properties.skins[loaderSkinIndex].jointMatrices.resize(skin.joints.size());

            shape.hasSkin = true;
            shape.skinIndex = loaderSkinIndex;
        }

        meshIndex++;
    }

    for (auto childId : node.children) {
        glm::mat4 tempGlobalTf = globalTf;
        loadNode(childId, lights, tempGlobalTf);
    }
}

void GltfLoader::updateGlobalTransform(tinygltf::Node &node, int nodeId, glm::mat4 &globalTf)
{
    size_t loaderNodeIndex = properties.gltfNodeIdToNodeIndexMap.find(nodeId)->second;
    SceneNode &loaderNode = properties.nodes[loaderNodeIndex];

    glm::mat4 localTf = glm::mat4(1.0f); // identity matrix

    if (node.matrix.size() != 0) {
        // Note: glTF and glm both use column-major matrices.
        float *localTfData = glm::value_ptr(localTf);
        for (int i = 0; i < 16; ++i)
            localTfData[i] = static_cast<float>(node.matrix[i]);

        loaderNode.hasMatrix = true;
    }
    else {
        glm::mat4 translate = glm::mat4(1.0f);
        glm::mat4 rotate = glm::mat4(1.0f);
        glm::mat4 scale = glm::mat4(1.0f);

        if (node.translation.size() != 0) {
            loaderNode.translation = glm::vec3(node.translation[0], node.translation[1], node.translation[2]);
            translate = glm::translate(glm::mat4(1.0f), loaderNode.translation);
        }

        if (node.rotation.size() != 0) {
            // node.rotation is a quaternion in the order x, y, z, w
            // glm::quat constructor parameter order is w, x, y, z

            loaderNode.rotation = glm::quat(
                static_cast<float>(node.rotation[3]),
                static_cast<float>(node.rotation[0]),
                static_cast<float>(node.rotation[1]),
                static_cast<float>(node.rotation[2]));

            rotate = glm::toMat4(loaderNode.rotation);
        }

        if (node.scale.size() != 0) {
            loaderNode.scale = glm::vec3(node.scale[0], node.scale[1], node.scale[2]);
            scale = glm::scale(glm::mat4(1.0f), loaderNode.scale);
        }

        localTf = translate * rotate * scale;
    }

    loaderNode.localTf = localTf;
    loaderNode.globalTf = globalTf * localTf;

    globalTf = loaderNode.globalTf;
}

void GltfLoader::loadLight(tinygltf::Value::Object::iterator &pos, std::vector<Light> &lights, glm::mat4 &globalTf)
{
    // Get light index.

    //tinygltf::Value &val = pos->second;
    //assert(val.IsObject());
    //assert(val.Has("light"));
    //assert(val.Get("light").IsInt());
    //int lightIndex = val.Get("light").Get<int>();

    int lightIndex = pos->second.Get("light").Get<int>();
    //std::cout << "lightIndex = " << lightIndex << std::endl;

    // Get light parameters from model.extensions.

    const tinygltf::Value &val = model.extensions["KHR_lights_punctual"].Get("lights").Get(lightIndex);
    assert(val.IsObject());
    tinygltf::Value::Object obj = val.Get<tinygltf::Value::Object>();
    //std::cout << obj["name"].Get<std::string>() << std::endl;
    std::string &type = obj["type"].Get<std::string>();
    //std::cout << type << std::endl;

    float intensity = static_cast<float>(obj["intensity"].GetNumberAsDouble());
    //std::cout << "intensity = " << std::fixed << intensity << " candela (lm/sr)" << std::endl;
    float wattage = intensity / 683.0f * 4.0f * M_PIf;

    glm::vec3 colour;
    tinygltf::Value::Array &arr = obj["color"].Get<tinygltf::Value::Array>();
    colour.r = static_cast<float>(arr[0].GetNumberAsDouble()); // arr[0] == obj["color"].Get(0)
    colour.g = static_cast<float>(arr[1].GetNumberAsDouble());
    colour.b = static_cast<float>(arr[2].GetNumberAsDouble());
    //std::cout << "colour = (" << colour.r << ", " << colour.g << ", " << colour.b << ")" << std::endl;

    if (type == "point") {
        glm::vec3 position = globalTf * glm::vec4(glm::vec3(0.0f), 1.0f);

        Light light;
        light.createPointLight(
            make_float3(position.x, position.y, position.z),
            wattage,
            make_float3(colour.r, colour.g, colour.b));
        lights.push_back(light);
    }
    else if (type == "directional") {
        //std::cout << "intensity = " << std::fixed << intensity << " lux (lm/m^2)" << std::endl;

        glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
        direction = glm::normalize(globalTf * glm::vec4(direction, 0.0f));

        Light light;
        light.createDirectionalLight(
            make_float3(0.0f),
            make_float3(0.0f),
            make_float3(0.0f),
            make_float3(direction.x, direction.y, direction.z),
            1000.0f,
            wattage,
            make_float3(colour.r, colour.g, colour.b));
        lights.push_back(light);
    }
    else if (type == "spot") {
        tinygltf::Value::Object &spotObj = obj["spot"].Get<tinygltf::Value::Object>();
        float innerConeAngle = static_cast<float>(spotObj["innerConeAngle"].GetNumberAsDouble());
        float outerConeAngle = static_cast<float>(spotObj["outerConeAngle"].GetNumberAsDouble());

        glm::vec3 position = globalTf * glm::vec4(glm::vec3(0.0f), 1.0f);

        glm::vec3 direction = glm::vec3(0.0f, 0.0f, -1.0f);
        direction = glm::normalize(globalTf * glm::vec4(direction, 0.0f));

        Light light;
        light.createSpotLight(
            make_float3(position.x, position.y, position.z),
            make_float3(direction.x, direction.y, direction.z),
            glm::degrees(outerConeAngle),
            glm::degrees(innerConeAngle),
            wattage,
            make_float3(colour.r, colour.g, colour.b));
        lights.push_back(light);
    }
}

void GltfLoader::loadCamera(int cameraId, glm::mat4 &globalTf)
{
    tinygltf::Camera &camera = model.cameras[cameraId];
    if (camera.type == "perspective") {
        cameraPresent = true;
        camAspectRatio = static_cast<float>(camera.perspective.aspectRatio);
        camFovYDeg = static_cast<float>(glm::degrees(camera.perspective.yfov));

        // Unlike vertices and normals, the global transform should be applied to the camera here.
        camOrigin = globalTf * glm::vec4(glm::vec3(0.0f), 1.0f);
        camTarget = globalTf * glm::vec4(glm::vec3(0.0f, 0.0f, -1.0f), 1.0f); // original

        /*
        // + This is a breakdown of the calculation above.
        glm::vec3 camLookAt = glm::vec3(0.0f, 0.0f, -1.0f);
        //camLookAt = globalTf * glm::vec4(camLookAt, 0.0f); // TO DO: Why does this have length 1?
        camLookAt = glm::normalize(globalTf * glm::vec4(camLookAt, 0.0f));
        //std::cout << "length = " << glm::length(camLookAt) << std::endl;
        camTarget = camOrigin + camLookAt;
        // -
        */
    }
    else {
        std::cout << "Unsupported camera type!" << std::endl;
    }
}

// Load a specific animation.
void GltfLoader::loadAnimations()
{
    //std::cout << model.animations.size() << " animation(s)" << std::endl;

    size_t numAnimations = model.animations.size();
    if (numAnimations == 0)
        return;

    // Select animation.
    int gltfAnimId = 2; // example
    if (gltfAnimId >= numAnimations)
        gltfAnimId = 0;

    auto &gltfAnim = model.animations[gltfAnimId];
    std::vector<AnimShape> &animShapes = properties.animShapes;

    int channelId = 0;
    for (auto &channel : gltfAnim.channels) {
        AnimShape animShape;

        //std::cout << "  Channel #" << channelId++ << std::endl;

        //std::cout << "    Target node = " << channel.target_node << std::endl;
        std::map<int, size_t>::iterator pos = properties.gltfNodeIdToShapeIndexMap.find(channel.target_node);
        if (pos != properties.gltfNodeIdToShapeIndexMap.end()) {
            //std::cout << "Target node found in map! " << pos->second << std::endl;
            animShape.shapeId = pos->second;
        }
        else {
            //std::cout << "    Target node not found in map!" << std::endl;
            animShape.shapeId = std::numeric_limits<size_t>::max();
        }

        animShape.nodeId = properties.gltfNodeIdToNodeIndexMap.find(channel.target_node)->second;

        //std::cout << "    Target path = " << channel.target_path << std::endl;

        //std::cout << "    Sampler (" << channel.sampler << ")" << std::endl;
        tinygltf::AnimationSampler &sampler = model.animations[gltfAnimId].samplers[channel.sampler];

        //std::cout << "      Interpolation = " << sampler.interpolation << std::endl;
        if (sampler.interpolation == "STEP")
            animShape.interpolation = 0;
        else if (sampler.interpolation == "LINEAR")
            animShape.interpolation = 1;
        else if (sampler.interpolation == "CUBIC")
            animShape.interpolation = 2;
        else
            animShape.interpolation = -1;

        //std::cout << "      Input (" << sampler.input << ")" << std::endl;
        //std::cout << "      Output (" << sampler.output << ")" << std::endl;
        tinygltf::Accessor &iAccessor = model.accessors[sampler.input];
        tinygltf::Accessor &oAccessor = model.accessors[sampler.output];
        assert(iAccessor.count == oAccessor.count);

        animShape.data.resize(iAccessor.count);

        {
            assert(iAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
            assert(iAccessor.type == TINYGLTF_TYPE_SCALAR);

            tinygltf::BufferView &bufferView = model.bufferViews[iAccessor.bufferView];
            tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

            size_t byteOffset = iAccessor.byteOffset + bufferView.byteOffset;
            unsigned char *ptrByteOffset = &buffer.data[byteOffset];
            for (size_t i = 0; i < iAccessor.count; ++i) {
                float t = *(float * )ptrByteOffset;
                ptrByteOffset += iAccessor.ByteStride(bufferView);
                // Log
                //std::cout << std::fixed << "        t = " << t << std::endl;
                animShape.data[i].t = t;
            }
        }

        {
            assert(oAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT);
            assert(channel.target_path == "translation" ? (oAccessor.type == TINYGLTF_TYPE_VEC3) : true);
            assert(channel.target_path == "rotation" ? (oAccessor.type == TINYGLTF_TYPE_VEC4) : true);
            assert(channel.target_path == "scale" ? (oAccessor.type == TINYGLTF_TYPE_VEC3) : true);

            if (channel.target_path == "translation")
                animShape.type = 0;
            else if (channel.target_path == "rotation")
                animShape.type = 1;
            else if (channel.target_path == "scale")
                animShape.type = 2;
            else
                animShape.type = -1; // not supporting morph targets for the moment

            tinygltf::BufferView &bufferView = model.bufferViews[oAccessor.bufferView];
            tinygltf::Buffer &buffer = model.buffers[bufferView.buffer];

            size_t byteOffset = oAccessor.byteOffset + bufferView.byteOffset;
            unsigned char *ptrByteOffset = &buffer.data[byteOffset];
            for (size_t i = 0; i < oAccessor.count; ++i) {
                if (animShape.type == 0 || animShape.type == 2) {
                    glm::vec3 v = *(glm::vec3 *)ptrByteOffset;
                    animShape.data[i].v3 = v;
                }
                else if (animShape.type == 1) {
                    glm::vec4 v = *(glm::vec4 *)ptrByteOffset;

                    // Log
                    //std::cout << std::fixed << "        vec4 = (" << v.x << ", " << v.y << ", " << v.z << ", " <<
                    //    v.w << ")" << std::endl;

                    // In gltf the order of the quaternion components is x, y, z, w.
                    // In glm the order is w, x, y, z (glm::quat(w, x, y, z)).
                    animShape.data[i].q = glm::quat(v.w, v.x, v.y, v.z);
                }

                ptrByteOffset += oAccessor.ByteStride(bufferView);
            }
        }

        animShapes.push_back(animShape);
    }
}
