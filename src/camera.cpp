#include "camera.h"
#include "utils.h"

#include <glm/gtx/transform.hpp>

void Camera::setParams(uint32_t resX, uint32_t resY, glm::vec3 eye, glm::vec3 target, glm::vec3 upHint,
    float vertFovDeg)
{
    this->eye = eye;
    this->target = target;
    this->upHint = glm::normalize(upHint);

    lookDir = glm::normalize(target - eye);

    aspectRatio = (float)resX / resY;
    this->vertFovDeg = vertFovDeg;
}

void Camera::setupUvwCoordinatSystem()
{
    // Create left-handed coordinate system.

    W = lookDir; // forward

    // Sometimes, the length of W is 1; sometimes it incorporates the focal
    // length (the distance from the eye to the image plane) so it is not 1.
    float wLen = glm::length(W);

    // The cross product of unit vectors may not be a unit vector, therefore
    // normalisation is necessary.
    U = glm::normalize(glm::cross(upHint, -W)); // right
    V = glm::normalize(glm::cross(-W, U)); // up

    // U and V are multiplied by uLen and vLen respectively to make the
    // calculations for determining the pixel on the image plane more efficient.
    float vLen = wLen * tanf(0.5f * glm::radians(vertFovDeg));
    V *= vLen;
    float uLen = vLen * aspectRatio;
    U *= uLen;
}

void Camera::incrementPositionAlongDirection(float moveLeftRight, float moveUpDown, float moveBackwardForward)
{
    glm::vec3 xAxisVector = glm::normalize(glm::cross(lookDir, upHint));
    eye += moveLeftRight * xAxisVector + moveBackwardForward * lookDir;
}

void Camera::incrementDirection(float rotationLeftRight, float rotationUpDown)
{
    glm::vec3 xAxisVector = glm::normalize(glm::cross(lookDir, upHint));
    glm::mat4 rotMat = glm::rotate(rotationLeftRight, upHint) * glm::rotate(rotationUpDown, xAxisVector);

    glm::vec4 lookDir4 = glm::vec4(lookDir, 1.0f);
    lookDir4 = rotMat * lookDir4;
    lookDir = glm::normalize(glm::vec3(lookDir4));
}
