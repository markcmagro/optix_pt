#pragma once

#include <string>
#include <vector>
#include <glm/glm.hpp>
#include "material_params.h"
#include "shape.h"

class Mesh
{
public:
    Mesh();
    ~Mesh();

    size_t numShapes;
    std::vector<Shape> shapes;

    size_t numVertices;                    // Number of triangle vertices
    std::vector<glm::vec3> positions;      // Triangle vertex positions (len numVertices)
    std::vector<glm::vec3> positionsTemp;  // Triangle vertex positions (len numVertices)

    bool hasNormals;
    std::vector<glm::vec3> normals;        // Triangle normals (len 0 or numVertices)
    std::vector<glm::vec3> normalsTemp;    // Triangle normals (len 0 or numVertices)

    bool hasTexCoords;
    std::vector<glm::vec2> texCoords;      // Triangle UVs (len 0 or numVertices)

    size_t numTriangles;                   // Number of triangles
    std::vector<glm::ivec3> triIndices;    // Indices into positions, normals, texcoords
    std::vector<int> matIndices;           // Indices into matParams (len numTriangles)

    size_t numMaterials;
    std::vector<MaterialParams> matParams;

    glm::vec3 bboxMin;                     // Scene bounding box
    glm::vec3 bboxMax;

    void clear();
    bool isValid();
    void applyTransform(const float *tf);

    void print();
};
