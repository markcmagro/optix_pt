#include <algorithm>
#include <iostream>
#include "cuda_helpers.h"
#include <glm/gtc/type_ptr.hpp>
#include "mesh.h"

Mesh::Mesh()
{
    numShapes = 0;
    numVertices = 0;
    hasNormals = false;
    hasTexCoords = false;
    numTriangles = 0;
    numMaterials = 0;
    bboxMin = glm::vec3( 1e16f);
    bboxMax = glm::vec3(-1e16f);
}

Mesh::~Mesh()
{
    positions.clear();
    normals.clear();
    texCoords.clear();
    triIndices.clear();
    matIndices.clear();
    matParams.clear();
}

void Mesh::clear()
{
    numShapes = 0;

    numVertices = 0;
    positions.clear();

    hasNormals = false;
    normals.clear();

    hasTexCoords = false;
    texCoords.clear();

    numTriangles = 0;
    triIndices.clear();
    matIndices.clear();

    numMaterials = 0;
    matParams.clear();

    bboxMin = glm::vec3( 1e16f);
    bboxMax = glm::vec3(-1e16f);
}

bool Mesh::isValid()
{
    if (numVertices == 0)
    {
        std::cerr << "Mesh not valid: numVertices == 0" << std::endl;
        return false;
    }

    if (positions.empty())
    {
        std::cerr << "Mesh not valid: positions is empty" << std::endl;
        return false;
    }

    if (numTriangles == 0)
    {
        std::cerr << "Mesh not valid: numTriangles == 0" << std::endl;
        return false;
    }

    if (triIndices.empty())
    {
        std::cerr << "Mesh not valid: triIndices is empty" << std::endl;
        return false;
    }

    if (matIndices.empty())
    {
        std::cerr << "Mesh not valid: mat_indices is empty" << std::endl;
        return false;
    }

    if (hasNormals && normals.empty())
    {
        std::cerr << "Mesh has normals, but normals is empty" << std::endl;
        return false;
    }

    if (hasTexCoords && texCoords.empty())
    {
        std::cerr << "Mesh has texcoords, but texcoords is empty" << std::endl;
        return false;
    }

    if (numMaterials == 0)
    {
        std::cerr << "Mesh not valid: numMaterials == 0" << std::endl;
        return false;
    }

    if (matParams.empty())
    {
        std::cerr << "Mesh not valid: matParams is empty" << std::endl;
        return false;
    }

    return true;
}

void Mesh::applyTransform(const float *tf)
{
    if (!tf)
        return;

    bool haveMatrix = false;
    for (int i = 0; i < 16; ++i) {
        if (tf[i] != 0.0f) {
            haveMatrix = true;
            break;
        }
    }

    if (haveMatrix)
    {
        bboxMin = glm::vec3( 1e16f);
        bboxMax = glm::vec3(-1e16f);

        glm::mat4 mat = glm::make_mat4(tf);

        for (size_t i = 0; i < numVertices; ++i) {
            const glm::vec3 v = glm::vec3(mat * glm::vec4(positions[i], 1.0f));
            positions[i] = v;

            bboxMin = glm::min(bboxMin, v);
            bboxMax = glm::max(bboxMax, v);
        }

        if (hasNormals) {
            // Normals need to be transformed by the transpose of the inverse of
            // the matrix mat so that they are unaffected by non-uniform scaling.
            // For a detailed explanation see
            // https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/geometry/
            //   transforming-normals.

            mat = glm::transpose(glm::inverse(mat));

            for (size_t i = 0; i < numVertices; ++i)
                normals[i] = glm::vec3(mat * glm::vec4(normals[i], 1.0f));
        }
    }
}

void Mesh::print()
{
    std::printf("Num vertices  = %zd\n", numVertices);
    std::printf("Num triangles = %zd\n", numTriangles);
    std::printf("Num shapes    = %zd\n", numShapes);
    std::printf("\n");

    std::printf("Vertices\n");
    int index = 0;
    for (auto v : positions) {
        std::printf(" %5d : % 9.2f, % 9.2f, % 9.2f\n", index, v.x, v.y, v.z);
        index++;
    }
    std::printf("\n");

    std::printf("Triangles\n");
    index = 0;
    for (auto t : triIndices) {
        std::printf(" %5d : %5d, %5d, %5d\n", index, t.x, t.y, t.z);
        index++;
    }
    std::printf("\n");

    std::printf("Shapes\n");
    index = 0;
    for (auto s : shapes) {
        std::printf("Shape %d\n", index);
        std::printf("    Num triangles    = %zd\n", s.numTriangles);
        std::printf("    Triangles offset = %zd\n", s.triangleOffset);
        std::printf("    Vertices index   = %zd\n", s.verticesIndex);
        index++;
    }
    std::printf("\n");
}
