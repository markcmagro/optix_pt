#pragma once

#include <vector>
#include <glm/glm.hpp>

struct Skin
{
    std::vector<glm::mat4> inverseBindMatrices;
    std::vector<int> joints; // gltf node IDs

    // The joint matrices are recomputed every time the skinned mesh is
    // rendered.
    std::vector<glm::mat4> jointMatrices;
};
