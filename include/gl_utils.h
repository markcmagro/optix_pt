#pragma once

#include <iostream>
#include <string>

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GLFW/glfw3.h>

#ifdef _DEBUG
#define GL(func) \
{ \
    func; \
    GLenum error; \
    while ((error = glGetError()) != GL_NO_ERROR) \
        std::cerr << #func << " failed. Error " << error << ". Line " << __LINE__ << "." << std::endl; \
}
#else
#define GL(func) func
#endif

bool checkCompileStatus(unsigned int shader);
bool checkLinkStatus(unsigned int program);
std::string loadFile(const char *path);
unsigned int loadShader(std::string &source, GLenum type);
unsigned int loadShader(const char *path, GLenum type);
unsigned int createProgram(unsigned int vertexShader, unsigned int fragmentShader, unsigned int geometryShader = -1);
