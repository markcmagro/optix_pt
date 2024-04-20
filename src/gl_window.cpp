#include <exception>
#include "gl_window.h"
#include "gl_utils.h"

GLWindow::GLWindow(const char *title, int posX, int posY, int resX, int resY, GLFWwindow *share)
{
    window = glfwCreateWindow(resX, resY, title, nullptr, share);
    if (!window)
        throw std::exception("GLFW window or GL context creation failed.");

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0); // 0 => disable vsync; 1 => clamp to 60 FPS

    int oldWidth, oldHeight;
    glfwGetWindowSize(window, &oldWidth, &oldHeight);
    //std::printf("%d x %d\n", oldWidth, oldHeight);

    glfwSetWindowPos(window, posX, posY);

    //int newWidth, newHeight;
    //glfwGetWindowSize(window, &newWidth, &newHeight);
    //std::printf("%d x %d\n", newWidth, newHeight);

    // After glfwSetWindowPos, the window may move to another display with
    // different scaling (e.g. the primary display is set to 125% and the second
    // display is set to 150%) causing the window size to appear incorrectly.
    // Set the window size to the original values to fix the problem.
    glfwSetWindowSize(window, oldWidth, oldHeight);

    //glfwGetWindowSize(window, &newWidth, &newHeight);
    //std::printf("%d x %d\n", newWidth, newHeight);

    glfwSetWindowAttrib(window, GLFW_RESIZABLE, GLFW_FALSE);
}

GLWindow::~GLWindow()
{
    glfwDestroyWindow(window);
}

GLFWwindow *GLWindow::getWindow()
{
    return window;
}
