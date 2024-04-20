#define _USE_MATH_DEFINES
#include <cmath>

#include <fstream>
#include <iostream>
#include <sstream>
#include "utils.h"

#include <glm/gtc/type_ptr.hpp>

std::string readFile(const char *fileName)
{
    std::ifstream is;

    is.open(fileName, std::ios::binary);
    if (is.is_open()) {
        std::stringstream ss;
        ss << is.rdbuf();
        return ss.str();
    }

    return std::string("");
}

std::string directoryOfFilePath(const std::string &filePath)
{
    size_t slashPos, backslashPos;
    slashPos = filePath.find_last_of('/');
    backslashPos = filePath.find_last_of('\\');

    size_t breakPos;
    if (slashPos == std::string::npos && backslashPos == std::string::npos) {
        return std::string();
    } else if (slashPos == std::string::npos) {
        breakPos = backslashPos;
    } else if (backslashPos == std::string::npos) {
        breakPos = slashPos;
    } else {
        breakPos = std::max(slashPos, backslashPos);
    }

    // Include the final slash.
    return filePath.substr(0, breakPos + 1);
}

void printMatrix(glm::mat4 &m)
{
    float *data = glm::value_ptr(m);
    for (int i = 0; i < 16; ++i)
        std::cout << data[i] << " ";
    std::cout << std::endl;
}

/*
glm::quat slerp(glm::quat prevQuat, glm::quat nextQuat, float interpolationValue)
{
    // References:
    // - https://github.com/KhronosGroup/glTF-Tutorials/blob/master/gltfTutorial/gltfTutorial_007_Animations.md.

    float dotProduct = glm::dot(prevQuat, nextQuat);

    // Make sure we take the shortest path in case dotProduct is negative.
    if(dotProduct < 0.0) {
        nextQuat = -nextQuat;
        dotProduct = -dotProduct;
    }

    // If the two quaternions are too close to each other, just linear interpolate between the 4D vectors.
    if (dotProduct > 0.9995f)
        return glm::normalize(prevQuat + interpolationValue * (nextQuat - prevQuat));

    // Perform the spherical linear interpolation.
    float theta_0 = acos(dotProduct);
    float theta = interpolationValue * theta_0;
    float sinTheta = sin(theta);
    float sinTheta_0 = sin(theta_0);

    float scalePrevQuat = cos(theta) - dotProduct * sinTheta / sinTheta_0;
    float scaleNextQuat = sinTheta / sinTheta_0;
    return scalePrevQuat * prevQuat + scaleNextQuat * nextQuat;
}
*/
