#pragma once

#include <vector>

#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include <glm/gtx/quaternion.hpp>

struct AnimShapeData
{
    float t;

    union {
        glm::vec3 v3;
        glm::quat q;
    };
};

struct AnimShape
{
    size_t shapeId; // index into mesh.shapes
    size_t nodeId;  // index into GltfLoader::nodes

    int type; // 0 => translation; 1 => rotation; 2 => scale
    int interpolation; // 0 => step; 1 => linear; 2 => cubic
    std::vector<AnimShapeData> data;

    AnimShape()
    {
    }
};

struct AnimJointData
{
};

struct AnimJoint
{
    std::vector<AnimJointData> data;
};

struct Anim
{
    std::vector<AnimShape> animShapes;
};
