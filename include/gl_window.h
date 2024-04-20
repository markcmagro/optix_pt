#pragma once

#include <GL/glew.h>
#include <GL/wglew.h>
#include <GLFW/glfw3.h>

class GLWindow
{
protected:
    GLFWwindow *window;

public:
    GLWindow(const char *title, int posX, int posY, int resX, int resY, GLFWwindow *share);
    virtual ~GLWindow();

    GLFWwindow *getWindow();
};
