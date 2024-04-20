#pragma once

#include "texture_transform.h"

struct SheenData
{
    glm::vec3 colourFactor { 0.0f };
    int colourTexIndex = -1;
    int colourTexCoord = 0;
    TextureTransform colourTexTf;

    float roughnessFactor = 0.0f;
    int roughnessTexIndex = -1;
    int roughnessTexCoord = 0;
    TextureTransform roughnessTexTf;
};
