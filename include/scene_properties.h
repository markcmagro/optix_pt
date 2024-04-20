#pragma once

#include <map>
#include "anim.h"
#include "camera.h"
#include "light.h"
#include "mesh.h"
#include "scene_node.h"
#include "skin.h"

struct SceneProperties
{
    Mesh mesh;
    Camera cam;

    // Emissive vertex indices into mesh.positions.
    // Each set of 3 indices makes up an emissive triangle.
    // These indices are used to copy the vertices into emissivePositions.
    std::vector<int> emissiveVertexIndices;

    // The vertex data corresponding to the above indices.
    // Each set of 3 vertices makes up an emissive triangle.
    std::vector<glm::vec3> emissiveVertices;

    // The texture coordinates corresponding to the above indices.
    std::vector<glm::vec2> emissiveTexCoords;

    // The emission values corresponding to each emissive triangle.
    std::vector<glm::vec3> emissiveTriRadiances;

    // The material indices corresponding to each emissive triangle.
    std::vector<int> emissiveTriMatIndices;

    std::vector<Light> lights;
    std::vector<AnimShape> animShapes;
    std::vector<Skin> skins;

    std::vector<size_t> rootNodeIndices; // indices into the nodes collection
    std::vector<SceneNode> nodes;

    std::map<int, size_t> gltfNodeIdToShapeIndexMap;
    std::map<int, size_t> gltfNodeIdToNodeIndexMap;
};
