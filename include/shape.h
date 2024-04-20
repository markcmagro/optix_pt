#pragma once

#include <string>
#include <glm/glm.hpp>

struct Shape
{
    int nodeId; // tinygltf node id

    size_t triangleOffset; // offset into the triIndices field of a mesh
    size_t numTriangles;
    size_t numVertices;
    size_t numNormals;
    size_t numTexCoords;
    size_t numTriIndices;

    size_t verticesIndex;
    size_t normalsIndex;

    std::vector<glm::u16vec4> joints; // gltf node indices, per vertex
    std::vector<glm::vec4> weights; // per vertex

    bool hasSkin; // to be deprecated (see scene_node.h)
    size_t skinIndex; // to be deprecated (see scene_node.h)

    // Emissive triangle indices, relative to this shape.
    std::vector<int> emissiveTriangles;
};
