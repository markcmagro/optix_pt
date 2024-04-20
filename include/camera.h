#pragma once

#include <glm/glm.hpp>

class Camera
{
public:
    glm::vec3 eye;
    glm::vec3 target;
    glm::vec3 upHint;

    glm::vec3 lookDir;

    glm::vec3 U;
    glm::vec3 V;
    glm::vec3 W;

    float aspectRatio;
    float vertFovDeg;

    void setParams(uint32_t resX, uint32_t resY, glm::vec3 eye, glm::vec3 target, glm::vec3 upHint, float vertFovDeg);

    void setupUvwCoordinatSystem();

    void incrementPositionAlongDirection(float moveLeftRight, float moveUpDown, float moveBackwardForward);
    void incrementDirection(float rotationLeftRight, float rotationUpDown);
};
