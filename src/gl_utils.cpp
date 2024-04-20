#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include "gl_utils.h"

bool checkCompileStatus(unsigned int shader)
{
    char infoLog[1024];
    int success;

    GL(glGetShaderiv(shader, GL_COMPILE_STATUS, &success));
    if (!success)
    {
        GL(glGetShaderInfoLog(shader, sizeof(infoLog), NULL, infoLog));
        std::cerr << "Compiler error! " << infoLog << std::endl;
        return false;
    }

    return true;
}

bool checkLinkStatus(unsigned int program)
{
    char infoLog[1024];
    int success;

    GL(glGetProgramiv(program, GL_LINK_STATUS, &success));
    if (!success)
    {
        GL(glGetProgramInfoLog(program, sizeof(infoLog), NULL, infoLog));
        std::cerr << "Linker error! " << infoLog << std::endl;
        return false;
    }

    return true;
}

std::string loadFile(const char *path)
{
    std::ifstream file;

    file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

    try
    {
        file.open(path);
        std::stringstream stream;
        stream << file.rdbuf();
        file.close();
        return stream.str();
    }
    catch (std::ifstream::failure e)
    {
        std::cout << "Error reading file " << path << "!" << std::endl;
    }

    return "";
}

unsigned int loadShader(std::string &source, GLenum type)
{
    const char *str = source.c_str();

    unsigned int shader = glCreateShader(type);
    GLenum error = glGetError();
    if (error != GL_NO_ERROR)
        std::cerr << "glCreateShader failed. Error " << error << ". Line " << __LINE__ << "." << std::endl;

    GL(glShaderSource(shader, 1, &str, nullptr));
    GL(glCompileShader(shader));
    if (!checkCompileStatus(shader)) {
        GL(glDeleteShader(shader));
        shader = 0;
    }
    return shader;
}

unsigned int loadShader(const char *path, GLenum type)
{
    std::string shaderStr = loadFile(path);
    if (shaderStr == "")
        return 0;
    return loadShader(shaderStr, type);
}

unsigned int createProgram(unsigned int vertexShader,
    unsigned int fragmentShader, unsigned int geometryShader)
{
    unsigned int program = glCreateProgram();
    if (vertexShader != -1)
        GL(glAttachShader(program, vertexShader));
    if (fragmentShader != -1)
        GL(glAttachShader(program, fragmentShader));
    if (geometryShader != -1)
        GL(glAttachShader(program, geometryShader));
    GL(glLinkProgram(program));
    if (!checkLinkStatus(program)) {
        GL(glDeleteProgram(program));
        program = 0;
    }
    return program;
}
