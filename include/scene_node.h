#pragma once

#include <vector>

struct SceneNode
{
    int id; // gltf ID

    bool isMesh;
    bool isJoint;

    bool hasSkin;
    // To do: Need skin index, or else keep using the one in Shape.

    // Contains a full matrix instead of TRS; this implies that transforms are
    // not animatable for this node. The matrix is stored in localTf.
    bool hasMatrix;

    // The combination of these transforms is also computed and stored in localTf.
    glm::vec3 translation; // T is constructed from this
    glm::quat rotation;    // R is constructed from this
    glm::vec3 scale;       // S is constructed from this

    glm::mat4 localTf;    // the original matrix if hasMatrix is true; otherwise T * R * S
    glm::mat4 globalTf;   // parentTf * localTf

    std::vector<size_t> children;

    SceneNode()
    {
        id = -1;
        isMesh = false;
        isJoint = false;
        hasSkin = false;
        hasMatrix = false;

        translation = glm::vec3(0.0f, 0.0f, 0.0f);
        rotation = glm::quat(1.0f, 0.0f, 0.0f, 0.0f);
        scale = glm::vec3(1.0f, 1.0f, 1.0f);

        localTf = glm::mat4(1.0f);
    }
};
