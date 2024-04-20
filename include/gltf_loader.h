#pragma once

#define TINYGLTF_NO_INCLUDE_STB_IMAGE
#define TINYGLTF_NO_INCLUDE_STB_IMAGE_WRITE
#include <tiny_gltf.h>

#include "light.h"
#include "scene_properties.h"
#include "sheen_data.h"
#include "texture_transform.h"

class GltfLoader
{
    std::string sceneFile;

    tinygltf::TinyGLTF loader;
    tinygltf::Model model;
    tinygltf::Scene *scene;

    size_t numMeshes;
    size_t numTriangles;
    size_t numVertices;
    size_t numNormals;
    size_t numTexCoords;
    size_t numIndices;

    size_t verticesIndex;
    size_t normalsIndex;
    size_t texCoordsIndex;
    size_t triIndicesIndex;

    size_t triIndicesBase;
    size_t meshIndex;

    bool cameraPresent;
    float camAspectRatio;
    float camFovYDeg;
    glm::vec3 camOrigin;
    glm::vec3 camTarget;

    SceneProperties &properties;

public:
    GltfLoader(const char *sceneFile, SceneProperties &properties);

    bool parse();
    void loadScene();

    bool isCameraPresent();
    float getCamAspectRatio();
    float getCamFovYDeg();
    glm::vec3 getCamOrigin();
    glm::vec3 getCamTarget();

private:
    const int numSpacesPerIndentLevel = 2;

    void printLoaderNode(size_t index);

    // Construct the node hierarchy and count the number of meshes, triangles, vertices, etc.
    void scanScene();
    void scanNode(int gltfNodeId, size_t parentNodeIndex, int depth);
    void markJoints();

    void parseTextureTransform(tinygltf::Value::Object &tt, TextureTransform *texTf, int *texCoord);
    void parseTextureTransformParent(tinygltf::Value::Object &ttParent, TextureTransform *texTf, int *texCoord);

    void parseSheen(tinygltf::Value::Object obj, SheenData *data);
    void initMaterialParams(std::vector<MaterialParams> &matParamsColl);

    void loadNode(int nodeId, std::vector<Light> &lights, glm::mat4 &globalTf);

    void updateGlobalTransform(tinygltf::Node &node, int nodeId, glm::mat4 &globalTf);

    void loadLight(tinygltf::Value::Object::iterator &pos, std::vector<Light> &lights, glm::mat4 &globalTf);
    void loadCamera(int cameraId, glm::mat4 &globalTf);

    void loadAnimations();
};
