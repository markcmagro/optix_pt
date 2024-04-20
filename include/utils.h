#pragma once

#include <string>

#include <glm/glm.hpp>

std::string readFile(const char *fileName);

std::string directoryOfFilePath(const std::string &filePath);

void printMatrix(glm::mat4 &m);

//glm::quat slerp(glm::quat prevQuat, glm::quat nextQuat, float interpolationValue);
